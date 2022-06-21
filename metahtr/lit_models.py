from pathlib import Path
from collections import defaultdict
from typing import Optional, Dict, Union, Tuple, Any, List, Sequence

from metahtr.data import WriterDataset
from metahtr.models import MAMLInterface, MetaHTR, MAMLHTR
from metahtr.util import (
    identity_collate_fn,
    TrainMode,
    PREDICTIONS_TO_LOG,
    prepare_writer_splits,
    BaseModelArch,
    MainModelArch,
)
from metahtr.lit_callbacks import (
    LogLearnableInnerLoopLearningRates,
    LogInstanceSpecificWeights,
    LogModelPredictions,
    LogWorstPredictions,
)

from htr.metrics import WordErrorRate, CharacterErrorRate
from htr.models.fphtr.fphtr import FullPageHTREncoderDecoder
from htr.models.sar.sar import ShowAttendRead
from htr.models.lit_models import LitShowAttendRead, LitFullPageHTREncoderDecoder
from htr.util import LabelEncoder

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import pytorch_lightning as pl
import learn2learn as l2l
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import Callback


class LitBaseEpisodic(pl.LightningModule):
    """Base class for models trained using episodic learning."""

    def __init__(
        self,
        base_model_arch: str,
        main_model_arch: str,
        taskset_train: Optional[Union[l2l.data.TaskDataset, Dataset]] = None,
        taskset_val: Optional[WriterDataset] = None,
        taskset_test: Optional[WriterDataset] = None,
        learning_rate: float = 0.0001,
        weight_decay: float = 0.0,
        max_val_batch_size: int = 128,
        num_workers: int = 0,
        grad_clip: Optional[float] = None,
        max_epochs: Optional[int] = None,
        use_cosine_lr_scheduler: bool = False,
        prms_to_log: Optional[Dict[str, Union[str, float, int]]] = None,
        **kwargs,
    ):
        """
        Args:
            base_model_arch (str): base model architecture descriptor
            main_model_arch (str): main model architecture descriptor
            taskset_train (Optional[Union[l2l.data.TaskDataset, Dataset]]):
                learn2learn train taskset
            taskset_val (Optional[WriterDataset]): val taskset
            taskset_test (Optional[WriterDataset]): test taskset
            learning_rate (float): learning rate
            weight_decay (float): weight decay
            max_val_batch_size (int): batch size used in val loop (test time)
            num_workers (int): how many sub-processes to use for data loading
            grad_clip (int): maximum L2-norm of gradients before clipping occurs
            max_epochs (Optional[int]): number of epochs the model will be trained.
                This is only used if `use_cosine_lr_scheduler` is set to True
            use_cosine_lr_scheduler (bool): whether to use a cosine annealing
                scheduler to decay the learning rate from its initial value
            prms_to_log (Optional[Dict[str, Union[str, float, int]]]): additional
                parameters to log in the Pytorch Lightning hparams file
        """
        super().__init__()

        assert not (use_cosine_lr_scheduler and max_epochs is None), (
            "When using cosine learning rate scheduler, specify `max_epochs` to "
            "configure the learning rate decay properly."
        )

        self.base_model_arch = base_model_arch
        self.main_model_arch = main_model_arch
        self.taskset_train = taskset_train
        self.taskset_val = taskset_val
        self.taskset_test = taskset_test
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_val_batch_size = max_val_batch_size
        self.num_workers = num_workers
        self.grad_clip = grad_clip
        self.max_epochs = max_epochs
        self.use_cosine_lr_scheduler = use_cosine_lr_scheduler

        # Initialize k-fold splits for validation and testing.
        self.val_writerid_to_splits = prepare_writer_splits(self.taskset_val)
        self.test_writerid_to_splits = prepare_writer_splits(self.taskset_test)

        # Save the parameters to log here so they can be accessed later by a subclass
        # that actually logs them. We don't log them here because Pytorch Lightning
        # doesn't like it if you save hyperparameters multiple times in multiple PL
        # modules.
        self.hparams_to_log = dict()
        if prms_to_log is not None:
            self.hparams_to_log.update(prms_to_log)

    def train_dataloader(self):
        # Since we are using a l2l TaskDataset which already batches the data,
        # using a PyTorch DataLoader is effectively redundant. However, Pytorch
        # Lightning requires the use of a proper DataLoader. Therefore, we pass a
        # DataLoader that acts as an identity function, simply passing along a single
        # batch of data prepared by the TaskDataset.
        return DataLoader(
            self.taskset_train,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=identity_collate_fn,
            pin_memory=False,
        )

    def val_dataloader(self):
        if self.taskset_val is not None:
            return DataLoader(
                self.taskset_val,
                batch_size=1,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=identity_collate_fn,
                pin_memory=False,
            )

    def test_dataloader(self):
        if self.taskset_test is not None:
            return DataLoader(
                self.taskset_test,
                batch_size=1,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=identity_collate_fn,
                pin_memory=False,
            )

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        if self.use_cosine_lr_scheduler:
            max_epochs = self.max_epochs or 20
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max_epochs,
                eta_min=1e-06,  # final learning rate
                verbose=True,
            )
            return [optimizer], [lr_scheduler]
        return optimizer

    def add_model_specific_callbacks(self, callbacks: List[Callback], *args, **kwargs):
        return callbacks

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument("--learning_rate", type=float, default=0.0001)
        parser.add_argument("--weight_decay", type=float, default=0.0)
        parser.add_argument(
            "--max_val_batch_size",
            type=int,
            default=128,
            help="Batch size for validation. If the number of samples for a "
            "writer exceeds this value, the data is split into chunks of size `max_val_batch_size`.",
        )
        parser.add_argument(
            "--grad_clip", type=float, default=None, help="Max gradient norm."
        )
        parser.add_argument("--num_workers", type=int, default=0)
        parser.add_argument(
            "--use_cosine_lr_scheduler",
            action="store_true",
            default=False,
            help="Use a cosine annealing scheduler to decay the learning rate.",
        )
        return parser


class LitMAMLLearner(LitBaseEpisodic):
    """Pytorch Lightning wrapper for the MAML model."""

    def __init__(
        self,
        cer_metric: CharacterErrorRate,
        wer_metric: WordErrorRate,
        base_model: Optional[nn.Module] = None,
        save_hparams: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.cer_metric = cer_metric
        self.wer_metric = wer_metric

        # For sub-classes of the current class, self.model can be defined in the
        # sub-class itself after __init__, or initialized here by passing
        # the `base_model` argument.
        self.model = None
        if base_model is not None:
            self.model = MAMLHTR(
                base_model=base_model,
                val_writerid_to_splits=self.val_writerid_to_splits,
                test_writerid_to_splits=self.test_writerid_to_splits,
                **kwargs,
            )

        self.automatic_optimization = False
        self.use_instance_weights = False
        self.char_to_avg_inst_weight = None

        if save_hparams:
            self.save_hyperparameters(
                "ways",
                "shots",
                "inner_lr",
                "use_instance_weights",
                "use_batch_stats_for_batchnorm",
                "use_dropout",
                "num_inner_steps",
                "learning_rate",
                "grad_clip",
                "max_epochs",
                "use_cosine_lr_scheduler",
            )
            self.save_hyperparameters(self.hparams_to_log)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)
        opt = self.optimizers()
        opt.zero_grad()
        outer_loss, inner_loss, inst_ws = self.model.meta_learn(
            batch, mode=TrainMode.TRAIN
        )
        if self.grad_clip is not None:
            self.clip_gradients(opt, self.grad_clip, "norm")
        opt.step()

        self.log("train_loss_outer", outer_loss, sync_dist=True, prog_bar=False)
        self.log(f"train_loss_inner", inner_loss, sync_dist=True, prog_bar=False)

        return {"loss": outer_loss, "char_to_inst_weights": inst_ws}

    def training_epoch_end(self, epoch_outputs):
        sch = self.lr_schedulers()
        if sch is not None:
            sch.step()

    def validation_step(self, batch, batch_idx):
        return self.val_or_test_step(batch, mode=TrainMode.VAL)

    def test_step(self, batch, batch_idx):
        return self.val_or_test_step(batch, mode=TrainMode.TEST)

    def val_or_test_step(
        self, batch: Tuple[Tensor, Tensor, Tensor], mode: TrainMode = TrainMode.VAL
    ):
        # val/test requires finetuning a model in the inner loop, hence we need to
        # enable gradients.
        torch.set_grad_enabled(True)
        outer_loss, inner_loss, inst_ws = self.model.meta_learn(batch, mode=mode)
        torch.set_grad_enabled(False)

        self.log(
            f"{mode.name.lower()}_loss_outer", outer_loss, sync_dist=True, prog_bar=True
        )
        self.log(
            f"{mode.name.lower()}_loss_inner",
            inner_loss,
            sync_dist=True,
            prog_bar=False,
        )
        self.log("char_error_rate", self.cer_metric, prog_bar=False)
        self.log("word_error_rate", self.wer_metric, prog_bar=True)

        return {"loss": outer_loss, "char_to_inst_weights": inst_ws}

    def add_model_specific_callbacks(
        self,
        callbacks: List[Callback],
        shots: int,
        ways: int,
        label_encoder: LabelEncoder,
        is_train: bool = True,
    ) -> List[Callback]:
        # Prepare fixed batches used for monitoring model predictions during training.
        im, t, wrtrs = next(iter(self.val_dataloader()))
        # Select the first writer in the batch.
        val_batch = (im[: shots * 2], t[: shots * 2], wrtrs[: shots * 2])
        im, t, wrtrs = next(iter(self.train_dataloader()))
        train_batch = (im[: shots * 2], t[: shots * 2], wrtrs[: shots * 2])

        assert (
            val_batch[-1].unique().numel() == 1 and val_batch[-1].unique().numel() == 1
        ), "Only one writer should be in the batch for logging."

        val_batch, train_batch = [
            (
                im[:shots],
                t[:shots],
                im[shots : shots + PREDICTIONS_TO_LOG["word"]],
                t[shots : shots + PREDICTIONS_TO_LOG["word"]],
                wrtrs[0],
            )
            for (im, t, wrtrs) in [val_batch, train_batch]
        ]

        callbacks.extend(
            [
                LogModelPredictions(
                    label_encoder=label_encoder,
                    val_batch=val_batch,
                    train_batch=train_batch,
                    predict_on_train_start=True,
                ),
                LogWorstPredictions(
                    label_encoder=label_encoder,
                    train_dataloader=self.train_dataloader(),
                    val_dataloader=self.val_dataloader(),
                    test_dataloader=self.test_dataloader(),
                    shots=shots,
                    ways=ways,
                    training_skipped=not is_train,
                ),
            ]
        )
        return callbacks

    @staticmethod
    def init_with_base_model_from_checkpoint(
        base_model_arch: str,
        checkpoint_path: Union[str, Path],
        model_hparams_file: Union[str, Path],
        label_encoder: LabelEncoder,
        maml_arch: str = "maml",
        model_params_to_log: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        base_model_arch = BaseModelArch.from_string(base_model_arch)
        maml_arch = MainModelArch.from_string(maml_arch)

        # Initialize base model.
        if base_model_arch is BaseModelArch.FPHTR:
            # Load FPHTR model.
            base_model = LitFullPageHTREncoderDecoder.load_from_checkpoint(
                checkpoint_path,
                hparams_file=str(model_hparams_file),
                strict=False,
                label_encoder=label_encoder,
                params_to_log=model_params_to_log,
                loss_reduction="none",  # necessary for instance-specific loss weights
            )
        elif base_model_arch is BaseModelArch.SAR:
            base_model = LitShowAttendRead.load_from_checkpoint(
                checkpoint_path,
                hparams_file=str(model_hparams_file),
                strict=False,
                label_encoder=label_encoder,
                params_to_log=model_params_to_log,
                loss_reduction="none",  # necessary for instance-specific loss weights
            )
        else:
            raise ValueError(base_model_arch)

        num_clf_weights = (
            base_model.model.clf_layer.in_features
            * base_model.model.clf_layer.out_features
        )

        # Initialize meta-model.
        if maml_arch is MainModelArch.MAML:
            model = LitMAMLLearner.load_from_checkpoint(
                checkpoint_path,
                strict=False,
                cer_metric=base_model.model.cer_metric,
                wer_metric=base_model.model.wer_metric,
                base_model=base_model.model,
                base_model_arch=base_model_arch,
                **kwargs,
            )
        elif maml_arch is MainModelArch.MetaHTR:
            model = LitMetaHTR.load_from_checkpoint(
                checkpoint_path,
                strict=False,
                cer_metric=base_model.model.cer_metric,
                wer_metric=base_model.model.wer_metric,
                base_model=base_model.model,
                base_model_arch=base_model_arch,
                num_clf_weights=num_clf_weights,
                **kwargs,
            )
        else:
            raise ValueError(maml_arch)

        return model

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("MAML")
        parser.add_argument("--shots", type=int, default=16)
        parser.add_argument("--ways", type=int, default=8)
        parser.add_argument("--num_inner_steps", type=int, default=1)
        parser.add_argument("--inner_lr", type=float, default=0.001)
        parser.add_argument(
            "--use_batch_stats_for_batchnorm",
            action="store_true",
            default=False,
            help="Use batch statistics over stored statistics for batchnorm layers.",
        )
        parser.add_argument(
            "--use_dropout",
            action="store_true",
            default=False,
            help="Use dropout in the outer loop.",
        )
        parser.add_argument(
            "--freeze_batchnorm_gamma",
            action="store_true",
            default=False,
            help="Freeze gamma (scaling factor) for all batchnorm layers.",
        )
        return parent_parser


class LitMetaHTR(LitMAMLLearner):
    """Pytorch Lightning wrapper for the MetaHTR model."""

    def __init__(
        self,
        base_model: nn.Module,
        num_clf_weights: int,
        inst_mlp_hidden_size: int,
        initial_inner_lr: float,
        use_instance_weights: bool,
        **kwargs,
    ):
        super().__init__(**kwargs, save_hparams=False)

        self.num_clf_weights = num_clf_weights
        self.inst_mlp_hidden_size = inst_mlp_hidden_size
        self.initial_inner_lr = initial_inner_lr
        self.use_instance_weights = use_instance_weights

        self.model = MetaHTR(
            base_model=base_model,
            val_writerid_to_splits=self.val_writerid_to_splits,
            test_writerid_to_splits=self.test_writerid_to_splits,
            num_clf_weights=num_clf_weights,
            inst_mlp_hidden_size=inst_mlp_hidden_size,
            initial_inner_lr=initial_inner_lr,
            use_instance_weights=use_instance_weights,
            **kwargs,
        )

        self.save_hyperparameters(
            "inst_mlp_hidden_size",
            "initial_inner_lr",
            "use_instance_weights",
            "ways",
            "shots",
            "inner_lr",
            "use_instance_weights",
            "use_batch_stats_for_batchnorm",
            "use_dropout",
            "num_inner_steps",
            "learning_rate",
            "grad_clip",
            "max_epochs",
            "use_cosine_lr_scheduler",
        )
        self.save_hyperparameters(self.hparams_to_log)

    def training_epoch_end(self, epoch_outputs):
        sch = self.lr_schedulers()
        if sch is not None:
            sch.step()
        if self.use_instance_weights:
            self.aggregate_epoch_instance_weights(epoch_outputs)

    def validation_epoch_end(self, epoch_outputs):
        if self.use_instance_weights:
            self.aggregate_epoch_instance_weights(epoch_outputs)

    def test_epoch_end(self, epoch_outputs):
        if self.use_instance_weights:
            self.aggregate_epoch_instance_weights(epoch_outputs)

    def aggregate_epoch_instance_weights(
        self, training_epoch_outputs: Sequence[Dict[Any, Any]]
    ):
        char_to_weights, char_to_avg_weight = defaultdict(list), defaultdict(float)
        # Aggregate all instance-specific weights.
        for dct in training_epoch_outputs:
            char_to_ws = dct["char_to_inst_weights"]
            for c_idx, ws in char_to_ws.items():
                char_to_weights[c_idx].extend(ws)
        # Calculate average instance weight per class for logging purposes.
        for c_idx, ws in char_to_weights.items():
            char_to_avg_weight[c_idx] = np.mean(ws)
        self.char_to_avg_inst_weight = char_to_avg_weight

    def add_model_specific_callbacks(
        self, callbacks: List[Callback], label_encoder: LabelEncoder, **kwargs
    ) -> List[Callback]:
        callbacks = super().add_model_specific_callbacks(
            callbacks,
            label_encoder=label_encoder,
            **kwargs,
        )
        callbacks.extend(
            [
                LogLearnableInnerLoopLearningRates(),
                LogInstanceSpecificWeights(label_encoder),
            ]
        )
        return callbacks

    @staticmethod
    def init_with_base_model_from_checkpoint(*args, **kwargs):
        return LitMAMLLearner.init_with_base_model_from_checkpoint(
            maml_arch="metahtr", *args, **kwargs
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("MetaHTR")
        parser.add_argument("--inst_mlp_hidden_size", type=int, default=8)
        parser.add_argument("--initial_inner_lr", type=float, default=0.0001)
        parser.add_argument(
            "--use_instance_weights",
            action="store_true",
            default=False,
            help="Use instance-specific weights proposed in the MetaHTR paper.",
        )
        return parent_parser
