import math
import re
from pathlib import Path
from typing import Tuple, Optional, List, Any, Dict

from metahtr.util import (
    TrainMode,
    decode_prediction,
    EOS_TOKEN,
    split_batch_train_mode,
    split_batch_test_mode,
    load_best_pl_checkpoint,
)

from htr.data import IAMDataset
from htr.util import LabelEncoder, matplotlib_imshow

import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from torch import Tensor
from torch.utils.data import DataLoader
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import ModelCheckpoint


PREDICTIONS_TO_LOG = {
    "word": 10,
    "line": 6,
    "form": 1,
}


class LogLearnableInnerLoopLearningRates(Callback):
    """Logs the learnable inner loop learning rates in a bar plot."""

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ):
        # Collect all inner loop learning rates.
        lrs = []
        for n, p in pl_module.state_dict().items():
            if n.startswith("model.gbml.compute_update"):
                ix = int(re.search(r"[0-9]+", n).group(0))
                lrs.append((ix, p.item()))
            elif n.startswith("model.gbml.lrs"):
                lrs.extend(list(enumerate(p.squeeze().tolist())))
        assert lrs != []

        # Plot the learning rates.
        xs, ys = zip(*lrs)
        fig = plt.figure()
        plt.bar(xs, ys, align="edge", alpha=0.5)
        plt.grid(True)
        plt.ylabel("learning rate")

        # Log to Tensorboard.
        tensorboard = trainer.logger.experiment
        tensorboard.add_figure(f"inner loop learning rates", fig, trainer.global_step)
        plt.close(fig)


class LogInstanceSpecificWeights(Callback):
    """Logs the average instance specific weights per ASCII character in a bar plot."""

    def __init__(self, label_encoder: LabelEncoder):
        self.label_encoder = label_encoder

    def on_train_epoch_start(self, trainer, pl_module):
        pl_module.char_to_avg_inst_weight = None

    def on_validation_epoch_start(self, trainer, pl_module):
        pl_module.char_to_avg_inst_weight = None

    def on_test_epoch_start(self, trainer, pl_module):
        pl_module.char_to_avg_inst_weight = None

    def on_train_epoch_end(self, trainer, pl_module):
        if pl_module.use_instance_weights:
            char_to_avg_weight = pl_module.char_to_avg_inst_weight
            assert char_to_avg_weight is not None
            self.log_instance_weights(trainer, char_to_avg_weight, "train")

    def on_validation_epoch_end(self, trainer, pl_module):
        if pl_module.use_instance_weights:
            char_to_avg_weight = pl_module.char_to_avg_inst_weight
            assert char_to_avg_weight is not None
            self.log_instance_weights(trainer, char_to_avg_weight, "val")

    def on_test_epoch_end(self, trainer, pl_module):
        if pl_module.use_instance_weights:
            char_to_avg_weight = pl_module.char_to_avg_inst_weight
            assert char_to_avg_weight is not None
            self.log_instance_weights(trainer, char_to_avg_weight, "test")

    def log_instance_weights(
        self,
        trainer: "pl.Trainer",
        char_to_avg_weight: Dict[int, float],
        mode: str = "train",
    ):
        # Decode the characters.
        chars, ws = zip(
            *sorted(char_to_avg_weight.items(), key=lambda kv: kv[1], reverse=True)
        )
        chars = self.label_encoder.inverse_transform(chars)

        # Replace special tokens with shorter names to make the plot more readable.
        _chars = []
        _tkn_abbrevs = {"<EOS>": "eos", "<PAD>": "pad", "<SOS>": "sos"}
        for i, c in enumerate(chars):
            if c in _tkn_abbrevs.keys():
                _chars.append(_tkn_abbrevs[c])
            else:
                _chars.append(c)
        chars = _chars

        # Plot the average instance-specific weight per character.
        to_plot = 10
        fig = plt.figure()
        plt.subplot(1, 2, 1)
        plt.bar(chars[:to_plot], ws[:to_plot], align="edge", alpha=0.5)
        plt.grid(True)
        plt.title("Highest")

        plt.subplot(1, 2, 2)
        plt.bar(chars[-to_plot:], ws[-to_plot:], align="edge", alpha=0.5)
        plt.grid(True)
        plt.title("Lowest")

        # Log to Tensorboard.
        tensorboard = trainer.logger.experiment
        tensorboard.add_figure(
            f"{mode}: average instance-specific weights", fig, trainer.global_step
        )
        plt.close(fig)


class LogWorstPredictions(Callback):
    """Logs the N worst model predictions based on character error rate."""

    def __init__(
        self,
        shots: int,
        ways: int,
        label_encoder: LabelEncoder,
        train_dataloader: Optional[DataLoader] = None,
        val_dataloader: Optional[DataLoader] = None,
        test_dataloader: Optional[DataLoader] = None,
        training_skipped: bool = False,
    ):
        self.label_encoder = label_encoder
        self.shots = shots
        self.ways = ways
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.training_skipped = training_skipped

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        if self.training_skipped and self.val_dataloader is not None:
            self.log_worst_predictions(
                self.val_dataloader, trainer, pl_module, mode=TrainMode.VAL
            )

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        if self.test_dataloader is not None:
            self.log_worst_predictions(
                self.test_dataloader, trainer, pl_module, mode=TrainMode.TEST
            )

    def on_fit_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        if self.train_dataloader is not None:
            self.log_worst_predictions(
                self.train_dataloader, trainer, pl_module, mode=TrainMode.TRAIN
            )
        if self.val_dataloader is not None:
            self.log_worst_predictions(
                self.val_dataloader, trainer, pl_module, mode=TrainMode.VAL
            )

    def log_worst_predictions(
        self,
        dataloader: DataLoader,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        mode: TrainMode = TrainMode.TRAIN,
        forward_mode: Optional[str] = None,
    ):
        img_cers = []
        device = "cuda:0" if pl_module.on_gpu else "cpu"
        if not self.training_skipped:
            load_best_pl_checkpoint(trainer, pl_module, self.label_encoder)
            pl_module = trainer.model

        print(f"Running {mode.name.lower()} inference on best model...")

        # Run inference on the validation set.
        torch.set_grad_enabled(True)
        for imgs, targets, writer_ids in dataloader:

            if mode is TrainMode.TRAIN:
                writer_batches = split_batch_train_mode(
                    [imgs, targets, writer_ids], self.ways, self.shots
                )
            else:
                writerid_to_splits = (
                    pl_module.val_writerid_to_splits
                    if mode is TrainMode.VAL
                    else pl_module.test_writerid_to_splits
                )
                writer_batches = split_batch_test_mode(
                    [imgs, targets, writer_ids], self.shots, writerid_to_splits
                )

            for adapt_imgs, adapt_tgts, query_imgs, query_tgts in writer_batches:
                args = [t.to(device) for t in [adapt_imgs, adapt_tgts, query_imgs]]
                _, preds, *_ = (
                    pl_module(*args, mode=forward_mode)
                    if forward_mode
                    else pl_module(*args)
                )

                cer_metric = pl_module.cer_metric
                for prd, tgt, im in zip(preds, query_tgts, query_imgs):
                    with torch.inference_mode():
                        cer_metric.reset()
                        cer = cer_metric(prd.unsqueeze(0), tgt.unsqueeze(0)).item()
                        img_cers.append((im, cer, prd.cpu(), tgt))
        torch.set_grad_enabled(False)
        self.log_worst_predictions_tensorboard(trainer, img_cers, mode)
        print("Done.")

    def log_worst_predictions_tensorboard(
        self,
        trainer: "pl.Trainer",
        img_cers: List[Any],
        mode: TrainMode = TrainMode.TRAIN,
    ):
        # Log the worst k predictions.
        eos_tkn_idx = self.label_encoder.transform([EOS_TOKEN])[0]
        to_log = PREDICTIONS_TO_LOG["word"] * 2
        img_cers.sort(key=lambda x: x[1], reverse=True)  # sort by CER
        img_cers = img_cers[:to_log]
        fig = plt.figure(figsize=(24, 16))
        for i, (im, cer, prd, tgt) in enumerate(img_cers):
            pred_str = decode_prediction(prd, self.label_encoder, eos_tkn_idx)
            target_str = decode_prediction(tgt, self.label_encoder, eos_tkn_idx)

            # Create plot.
            ncols = 4
            nrows = math.ceil(to_log / ncols)
            ax = fig.add_subplot(nrows, ncols, i + 1, xticks=[], yticks=[])
            matplotlib_imshow(im, IAMDataset.MEAN, IAMDataset.STD)
            ax.set_title(f"Pred: {pred_str} (CER: {cer:.2f})\nTarget: {target_str}")

        # Log the results to Tensorboard.
        tensorboard = trainer.logger.experiment
        tensorboard.add_figure(
            f"{mode.name.lower()}: worst predictions", fig, trainer.global_step
        )
        plt.close(fig)


class LogModelPredictions(Callback):
    """
    Use a fixed test batch to monitor model predictions at the end of every epoch.

    Specifically: it generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's prediction alongside the actual target.
    """

    def __init__(
        self,
        label_encoder: LabelEncoder,
        val_batch: Optional[Tuple[Tensor, Tensor, Tensor, Tensor, str]] = None,
        train_batch: Optional[Tuple[Tensor, Tensor, Tensor, Tensor, str]] = None,
        data_format: str = "word",
        predict_on_train_start: bool = False,
    ):
        self.label_encoder = label_encoder
        self.val_batch = val_batch
        self.train_batch = train_batch
        self.data_format = data_format
        self.predict_on_train_start = predict_on_train_start

    def on_validation_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ):
        if self.val_batch is not None:
            self._predict_intermediate(trainer, pl_module, split=TrainMode.VAL)

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ):
        if self.train_batch is not None:
            self._predict_intermediate(trainer, pl_module, split=TrainMode.TRAIN)

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        if self.predict_on_train_start:
            if self.val_batch is not None:
                self._predict_intermediate(trainer, pl_module, split=TrainMode.VAL)
            if self.train_batch is not None:
                self._predict_intermediate(trainer, pl_module, split=TrainMode.TRAIN)

        # Log the support images once at the start of training.
        if self.val_batch is not None:
            support_imgs, support_targets, _, _, writer = self.val_batch
            self._log_intermediate(
                trainer,
                support_imgs,
                support_targets,
                split=TrainMode.VAL,
                plot_title="support batch",
                plot_suptitle=f"Writer id: {writer}",
            )
        if self.train_batch is not None:
            support_imgs, support_targets, _, _, writer = self.train_batch
            self._log_intermediate(
                trainer,
                support_imgs,
                support_targets,
                split=TrainMode.TRAIN,
                plot_title="support batch",
                plot_suptitle=f"Writer id: {writer}",
            )

    def _predict_intermediate(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        split: TrainMode = TrainMode.VAL,
    ):
        """Make predictions on a fixed batch of data; log the results to Tensorboard."""

        device = "cuda:0" if pl_module.on_gpu else "cpu"
        if split == TrainMode.TRAIN:
            batch = self.train_batch
        else:
            batch = self.val_batch

        # Make predictions.
        support_imgs, support_tgts, query_imgs, query_tgts, writer = batch
        torch.set_grad_enabled(True)
        _, preds, *_ = pl_module(
            *[t.to(device) for t in [support_imgs, support_tgts, query_imgs]]
        )
        torch.set_grad_enabled(False)

        # Log the results.
        self._log_intermediate(
            trainer,
            query_imgs,
            query_tgts,
            preds,
            split=split,
            plot_suptitle=f"Writer id: {writer}",
        )

    def _log_intermediate(
        self,
        trainer: "pl.Trainer",
        imgs: Tensor,
        targets: Tensor,
        preds: Optional[Tensor] = None,
        split: TrainMode = TrainMode.VAL,
        plot_suptitle: Optional[str] = None,
        plot_title: str = "query predictions vs targets",
    ):
        """Log a batch of images along with their targets to Tensorboard."""

        assert imgs.shape[0] == targets.shape[0]

        eos_tkn_idx = self.label_encoder.transform([EOS_TOKEN])[0]
        # Generate plot.
        fig = plt.figure(figsize=(12, 16))
        for i, (tgt, im) in enumerate(zip(targets, imgs)):
            # Decode predictions and targets.
            pred_str = None
            if preds is not None:
                pred_str = decode_prediction(preds[i], self.label_encoder, eos_tkn_idx)
            target_str = decode_prediction(tgt, self.label_encoder, eos_tkn_idx)

            # Create plot.
            ncols = 2 if self.data_format == "word" else 1
            nrows = math.ceil(targets.size(0) / ncols)
            ax = fig.add_subplot(nrows, ncols, i + 1, xticks=[], yticks=[])
            matplotlib_imshow(im, IAMDataset.MEAN, IAMDataset.STD)
            ttl = f"Target: {target_str}"
            if pred_str is not None:
                ttl = f"Pred: {pred_str}\n" + ttl
            ax.set_title(ttl)
        if plot_suptitle is not None:
            fig.suptitle(plot_suptitle)

        # Log the results to Tensorboard.
        tensorboard = trainer.logger.experiment
        tensorboard.add_figure(
            f"{split.name.lower()}: {plot_title}", fig, trainer.global_step
        )
        plt.close(fig)
