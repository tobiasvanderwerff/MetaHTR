from copy import copy
from enum import Enum
from functools import partial
import math
from pathlib import Path
import random
import shutil
from typing import Optional, Tuple, List, Sequence, Any, Union, Dict
from collections import OrderedDict

from metahtr.data import WriterDataset, PtTaskDataset

from htr.data import IAMDataset
from htr.util import LabelEncoder

import torch
import learn2learn as l2l
import torch.nn as nn
import numpy as np
import pandas as pd
from torch import Tensor
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.plugins import TorchCheckpointIO
from pytorch_lightning.utilities.types import _PATH
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.saving import load_hparams_from_yaml


PREDICTIONS_TO_LOG = {
    "word": 10,
    "line": 6,
    "form": 1,
}
EOS_TOKEN = "<EOS>"
SOS_TOKEN = "<SOS>"
PAD_TOKEN = "<PAD>"


class ExtendedEnum(Enum):
    @classmethod
    def from_string(cls, s: str):
        s = s.lower()
        for el in cls:  # iterate over all enum values
            if s == el.name.lower():
                return el
        raise ValueError(f"{s} is not a valid enum specifier.")

    @classmethod
    def get_vals(cls) -> List[str]:
        return [el.name.lower() for el in cls]


class TrainMode(ExtendedEnum):
    TRAIN = 1
    VAL = 2
    TEST = 3


class BaseModelArch(ExtendedEnum):
    FPHTR = 1
    SAR = 2


class MainModelArch(ExtendedEnum):
    MAML = 1
    MetaHTR = 2


def main_lit_models():
    from metahtr.lit_models import LitMAMLLearner, LitMetaHTR

    return {
        "MAML": LitMAMLLearner,
        "MetaHTR": LitMetaHTR,
    }


def get_label_encoder(trained_model_path: Union[str, Path]) -> LabelEncoder:
    """Load a stored label encoder originating from a trained model."""
    assert Path(
        trained_model_path
    ).is_file(), f"{trained_model_path} does not point to a file."
    model_path = Path(trained_model_path).resolve()
    le_path_1 = model_path.parent.parent / "label_encoding.txt"
    le_path_2 = model_path.parent.parent / "label_encoder.pkl"
    assert le_path_1.is_file() or le_path_2.is_file(), (
        f"Label encoder file not found at {le_path_1} or {le_path_2}. "
        f"Make sure 'label_encoding.txt' exists in the lightning_logs directory."
    )
    le_path = le_path_2 if le_path_2.is_file() else le_path_1
    return LabelEncoder().read_encoding(le_path)


def save_label_encoder(label_encoder: LabelEncoder, path: Union[str, Path]):
    """Save a label encoder to a specified path."""
    Path(path).mkdir(exist_ok=True, parents=True)
    label_encoder.dump(path)


def get_pl_tb_logger(log_dir: Union[str, Path], version: Optional[str] = None):
    """Get a Pytorch Lightning Tensorboard logger."""
    return pl_loggers.TensorBoardLogger(str(Path(log_dir)), name="", version=version)


def copy_hyperparameters_to_logging_dir(
    trained_model_path: Union[str, Path], log_dir: Union[str, Path]
) -> Tuple[str, Dict[str, Any]]:
    """
    Copy hyperparameters to logging directory. The loaded base model has an associated
    `hparams.yaml` file, which is copied to the current logging directory so that the
    base model can be loaded later using the saved hyper parameters.

    Returns:
        - path to hparams file
        - loaded hyperparameters as a dictionary
    """
    model_path = Path(trained_model_path)
    model_path_1 = model_path.parent.parent / "model_hparams.yaml"
    model_path_2 = model_path.parent.parent / "hparams.yaml"
    model_hparams_file = model_path_1 if model_path_1.is_file() else model_path_2
    model_hparams_file = str(model_hparams_file)
    shutil.copy(model_hparams_file, Path(log_dir) / "model_hparams.yaml")
    return model_hparams_file, load_hparams_from_yaml(model_hparams_file)


def prepare_iam_splits(dataset: IAMDataset, aachen_splits_path: Union[str, Path]):
    """Prepare IAM dataset train/val/(test) splits.

    The Aachen splits are used for the IAM dataset. It should be noted that these
    splits do not encompass the complete IAM dataset. Also worth noting is that in the
    Aachen splits, the writers present in train/val/test are disjoint.
    """
    train_splits = (aachen_splits_path / "train.uttlist").read_text().splitlines()
    validation_splits = (
        (aachen_splits_path / "validation.uttlist").read_text().splitlines()
    )
    test_splits = (aachen_splits_path / "test.uttlist").read_text().splitlines()

    data_train = dataset.data[dataset.data["img_id"].isin(train_splits)]
    data_val = dataset.data[dataset.data["img_id"].isin(validation_splits)]
    data_test = dataset.data[dataset.data["img_id"].isin(test_splits)]

    ds_train = copy(dataset)
    ds_train.data = data_train

    ds_val = copy(dataset)
    ds_val.data = data_val

    ds_test = copy(dataset)
    ds_test.data = data_test

    return ds_train, ds_val, ds_test


def prepare_writer_splits(dataset: WriterDataset):
    """Initialize k-fold splits for validation and testing."""
    writerid_to_splits = dict()
    for wid, img_idxs in dataset.writerid_to_img_idxs.items():
        splits = []
        idxs = list(range(len(img_idxs)))
        for _ in range(10):  # create 10 random orderings
            splits.append(random.sample(idxs, len(idxs)))
        writerid_to_splits[wid] = splits
    return writerid_to_splits


def prepare_test_taskset(dataset: IAMDataset):
    eos_tkn_idx, sos_tkn_idx, pad_tkn_idx = dataset.label_enc.transform(
        [EOS_TOKEN, SOS_TOKEN, PAD_TOKEN]
    )
    collate_fn = partial(
        IAMDataset.collate_fn,
        pad_val=pad_tkn_idx,
        eos_tkn_idx=eos_tkn_idx,
        dataset_returns_writer_id=True,
    )
    return WriterDataset(dataset, collate_fn)


def prepare_train_taskset(
    dataset: IAMDataset,
    ways: int,
    bookkeeping_path: Union[str, Path],
    shots: Optional[int] = None,
):
    eos_tkn_idx, sos_tkn_idx, pad_tkn_idx = dataset.label_enc.transform(
        [EOS_TOKEN, SOS_TOKEN, PAD_TOKEN]
    )
    collate_fn = partial(
        IAMDataset.collate_fn,
        pad_val=pad_tkn_idx,
        eos_tkn_idx=eos_tkn_idx,
        dataset_returns_writer_id=True,
    )
    # Setting the _bookkeeping_path attribute will make the MetaDataset instance
    # load its label-index mapping from a file, rather than creating it (which takes a
    # long time). If the path does not exists, the bookkeeping will be created and
    # stored on disk afterwards. Number of shots is stored along with the mapping,
    # because due to filtering of writers with less than `shots * 2` examples,
    # the mapping can change with the number of shots.
    dataset._bookkeeping_path = bookkeeping_path
    dataset_meta = l2l.data.MetaDataset(dataset)

    # Define learn2learn task transforms.
    task_trnsf = [
        # Nways picks N random labels (writers in this case)
        l2l.data.transforms.NWays(dataset_meta, n=ways),
    ]
    if shots is not None:
        # Keep K samples for each present writer.
        task_trnsf.append(l2l.data.transforms.KShots(dataset_meta, k=shots))
    task_trnsf.append(l2l.data.transforms.LoadData(dataset_meta))

    taskset = l2l.data.TaskDataset(
        dataset_meta,
        task_trnsf,
        num_tasks=-1,
        task_collate=collate_fn,
    )

    sample_per_epoch = int(len(dataset.writer_ids) / ways)

    # Wrap the task datasets into a simple class that sets a length for the dataset
    # (other than 1, which is the default if setting num_tasks=-1).
    # This is necessary because the dataset length is used by Pytorch dataloaders to
    # determine how many batches are in the dataset per epoch.
    return PtTaskDataset(taskset, epoch_length=sample_per_epoch)


def identity_collate_fn(x: Sequence[Any]) -> Any:
    """
    This function can be used for PyTorch dataloaders that return batches of size
    1 and do not require any collation of samples in the batch. This is useful if a
    batch of data is already prepared when it is passed to the dataloader.
    """
    assert len(x) == 1
    return x[0]


def decode_prediction(
    pred: Tensor, label_encoder: LabelEncoder, eos_tkn_idx: int
) -> str:
    """Convert a sequence of indices to a sequence of letter codes."""
    eos_idx = (pred == eos_tkn_idx).float().argmax().item()
    res = pred[:eos_idx] if eos_idx != 0 else pred
    return "".join(label_encoder.inverse_transform(res.tolist()))


def freeze(model: nn.Module):
    """Freeze the weights of a Pytorch module."""
    for p in model.parameters():
        p.requires_grad = False


def get_parameter_names(checkpoint_path: Union[str, Path]):
    """Return all parameter names in a state dict in a checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    return [wn for wn in ckpt["state_dict"].keys()]


def chunk_batch(
    imgs: Tensor, targets: Tensor, max_batch_size: int
) -> Tuple[Sequence[Tensor], Sequence[Tensor]]:
    """Chunk a batch of data into a sequence of smaller batches."""
    n_chunks = math.ceil(imgs.size(0) / max_batch_size)
    query_img_chunks = torch.chunk(imgs, n_chunks)
    query_tgt_chunks = torch.chunk(targets, n_chunks)
    return query_img_chunks, query_tgt_chunks


def split_batch_test_mode(
    batch, shots: int, writerid_to_splits: Dict[int, List[List[int]]]
):
    """
    For validation/testing, an incoming batch consists of all the examples
    for a particular writer. This functions splits the batch several times using
    given support/query splits.
    """
    imgs, target, writer_ids = batch
    writer_ids_uniq = writer_ids.unique().tolist()
    assert len(writer_ids_uniq) == 1, "For val/test, supply only one writer per batch"

    splits = writerid_to_splits[writer_ids_uniq[0]]
    batches = []
    for split in splits:
        supp_idxs = split[:shots]
        query_idxs = split[shots:]
        supp_imgs, supp_tgts = imgs[supp_idxs], target[supp_idxs]
        query_imgs, query_tgts = imgs[query_idxs], target[query_idxs]
        batches.append((supp_imgs, supp_tgts, query_imgs, query_tgts))
    return batches


def split_batch_train_mode(
    batch, ways: int, shots: int, limit_num_samples_per_task: Optional[int] = None
) -> List[Tuple[Tensor, Tensor, Tensor, Tensor]]:
    """
    Split a batch of data for adaptation.

    Based on a batch of the form (imgs, target, writer_ids), split the batch based on
    the writers in the batch, and then split each writer batch into a single
    adaptation and query batch.

    Returns:
        Adaptation/query 4-tuples for each writer, of the form
            (adaptation_imgs, adaptation_tgts, query_imgs, query_tgts)
    """
    imgs, target, writer_ids = batch
    writer_ids_uniq = writer_ids.unique().tolist()

    assert len(writer_ids_uniq) == ways, f"{len(writer_ids_uniq)} vs {ways}"

    # Split the batch into N different writers, where N = ways.
    writer_batches = []
    for task in range(ways):  # tasks correspond to different writers
        wrtr_id = writer_ids_uniq[task]
        task_slice = writer_ids == wrtr_id
        imgs_, target_ = (
            imgs[task_slice],
            target[task_slice],
        )
        if limit_num_samples_per_task is not None:
            imgs_, target_ = (
                imgs[:limit_num_samples_per_task],
                target[:limit_num_samples_per_task],
            )

        # Separate data into support/query set.
        adaptation_indices = np.zeros(imgs_.size(0), dtype=bool)
        # Select first k even indices for adaptation set.
        adaptation_indices[np.arange(shots) * 2] = True
        # Select remaining indices for query set.
        query_indices = torch.from_numpy(~adaptation_indices)
        adaptation_indices = torch.from_numpy(adaptation_indices)
        adaptation_imgs, adaptation_tgts = (
            imgs_[adaptation_indices],
            target_[adaptation_indices],
        )
        query_imgs, query_tgts = imgs_[query_indices], target_[query_indices]
        writer_batches.append(
            (adaptation_imgs, adaptation_tgts, query_imgs, query_tgts)
        )

    return writer_batches


def filter_df_by_freq(df: pd.DataFrame, column: str, min_freq: int) -> pd.DataFrame:
    """
    Filters a pandas DataFrame based on the value frequency in the specified column.

    Taken from https://stackoverflow.com/questions/30485151/python-pandas-exclude
    -rows-below-a-certain-frequency-count#answer-58809668.

    :param df: DataFrame to be filtered.
    :param column: Column name that should be frequency filtered.
    :param min_freq: Minimal value frequency for the row to be accepted.
    :return: Frequency filtered DataFrame.
    """
    # Frequencies of each value in the column.
    freq = df[column].value_counts()
    # Select frequent values. Value is in the index.
    frequent_values = freq[freq >= min_freq].index
    # Return only rows with value frequency above threshold.
    return df[df[column].isin(frequent_values)]


def set_batchnorm_layers_train(model: nn.Module, training: bool = True):
    """Set the `training` attribute of all Batch Norm layers in a Pytorch module.

    Setting it to True/False has the following effect:
    training=True:
        - running statistics of activiations are collected
        - batch statistics are used for normalization
    training=False:
        - running statistics of activiations are *not* collected
        - saved running statistics are used for normalization
    """
    _batchnorm_layers = (nn.BatchNorm1d, nn.BatchNorm2d)
    for m in model.modules():
        if isinstance(m, _batchnorm_layers):
            m.training = training


@torch.no_grad()
def batchnorm_reset_running_stats(model: nn.Module):
    """Reset running statistics in Batch Norm layers."""
    _batchnorm_layers = (nn.BatchNorm1d, nn.BatchNorm2d)
    for m in model.modules():
        if isinstance(m, _batchnorm_layers):
            m.reset_running_stats()


def set_dropout_layers_train(model: nn.Module, training: bool = True):
    """
    Set `training` attribute of Dropout layers. Setting it to False disables
    Dropout, setting it to True enables it.
    """
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.training = training


def reset_all_weights(module: nn.Module):
    """Randomize all weights in a Pytorch module."""
    if hasattr(module, "reset_parameters"):
        module.reset_parameters()
    elif hasattr(module, "_reset_parameters"):
        module._reset_parameters()
    for m in module.children():  # recurse through children modules
        reset_all_weights(m)


def load_best_pl_checkpoint(
    trainer: "pl.Trainer",
    pl_module: "pl.LightningModule",
    label_encoder: LabelEncoder,
    verbose: bool = True,
):
    """
    Load the best checkpoint for a given PL module, and set it as the new model for
    the given trainer.
    """
    ckpt_callback = None
    for cb in trainer.callbacks:
        if isinstance(cb, ModelCheckpoint):
            ckpt_callback = cb
            break
    assert ckpt_callback is not None, "ModelCheckpoint not found in callbacks."
    best_model_path = ckpt_callback.best_model_path

    if verbose:
        print(f"Loading best model at {best_model_path}")

    model_hparams_file = Path(best_model_path).parent.parent / "model_hparams.yaml"

    # Prepare model arguments.
    args = dict(
        base_model_arch=pl_module.base_model_arch,
        main_model_arch=pl_module.main_model_arch,
        checkpoint_path=best_model_path,
        model_hparams_file=model_hparams_file,
        label_encoder=label_encoder,
        taskset_train=pl_module.taskset_train,
        taskset_val=pl_module.taskset_val,
        taskset_test=pl_module.taskset_test,
        num_workers=pl_module.num_workers,
    )
    args.update(pl_module.hparams)

    # Load the model checkpoint.
    cls = pl_module.__class__
    best_model = cls.init_with_base_model_from_checkpoint(**args)
    trainer.model = best_model  # set new model as trainer attribute


class LayerWiseLRTransform:
    """
    A modified version of the l2l.optim.ModuleTransform class to use
    per-layer learning rates in the MAML framework.
    """

    def __init__(self, initial_lr: float = 0.0001):
        self.initial_lr = initial_lr

    def __call__(self, parameter):
        # In combination with `GBML` class, `l2l.nn.Scale` will scale the gradient for
        # each layer in a model with an adaptable learning rate.
        transform = l2l.nn.Scale(shape=1, alpha=self.initial_lr)
        numel = parameter.numel()
        flat_shape = (1, numel)
        return l2l.optim.ReshapedTransform(
            transform=transform,
            shape=flat_shape,
        )


class MAMLHTRCheckpointIO(TorchCheckpointIO):
    """
    Class that modifies the default behavior of Pytorch Lightning checkpoint
    saving.

    This is used to modify the hierarchy of attributes in the PL module under
    consideration, which makes sure the base model weights are saved under the same
    name as they were originally loaded.
    """

    def __init__(self, base_model_params: List[str]):
        self.base_model_params = base_model_params
        self.new_to_old = None

    def save_checkpoint(
        self,
        checkpoint: Dict[str, Any],
        path: _PATH,
        storage_options: Optional[Any] = None,
    ) -> None:
        if self.new_to_old is None:
            self._init_new_to_old(checkpoint)
        checkpoint = self._correct_base_model_weight_names(checkpoint)
        super().save_checkpoint(checkpoint, path, storage_options)

    def _init_new_to_old(self, checkpoint: Dict[str, Any]):
        """
        Map newly loaded base model weights, which have an additional prefix
        because they are loaded as part of a new model, to their original weight
        names.
        """
        new_to_old = dict()
        state_dict = checkpoint["state_dict"]
        for new in state_dict.keys():
            for old in self.base_model_params:
                if new.endswith(old) or new.endswith(".".join(old.split(".")[1:])):
                    new_to_old[new] = old
                    break
        assert len(new_to_old) == len(self.base_model_params)
        self.new_to_old = new_to_old

    def _correct_base_model_weight_names(self, checkpoint: Dict[str, Any]):
        """
        For all base model weights, save them under the same name as they were
        originally loaded. This makes it easier to load the weights from their
        original classes later on.
        """
        new_dict = OrderedDict()
        state_dict = checkpoint["state_dict"]
        n_corrected = 0
        while len(state_dict) > 0:
            wn, w = state_dict.popitem()
            old = self.new_to_old.get(wn)
            if old is not None:
                wn = old
                n_corrected += 1
            new_dict[wn] = w
        assert n_corrected == len(self.base_model_params), (
            f"Not all base model parameters were found: {n_corrected} found, "
            f"whereas {len(self.base_model_params)} should be present."
        )
        checkpoint["state_dict"] = new_dict
        return checkpoint
