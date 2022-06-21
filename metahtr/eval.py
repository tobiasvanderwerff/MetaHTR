"""Run test/validation on a trained model."""

import argparse
from pathlib import Path

from pytorch_lightning.core.saving import load_hparams_from_yaml

from metahtr.util import (
    filter_df_by_freq,
    get_label_encoder,
    get_pl_tb_logger,
    prepare_iam_splits,
    main_lit_models,
    prepare_test_taskset,
)

from htr.data import IAMDataset
from htr.util import LitProgressBar

from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelSummary


def main(args):

    print(f"Running validation on {'TEST' if args.test else 'VAL'} set.")

    model_path = Path(args.trained_model_path)
    hparams_path = model_path.parent.parent / "hparams.yaml"
    base_hparams_path = model_path.parent.parent / "model_hparams.yaml"
    hparams = load_hparams_from_yaml(str(hparams_path))
    base_hparams = load_hparams_from_yaml(str(base_hparams_path))

    seed_everything(args.seed)

    # Initalize logging/cache directories.
    log_dir = args.log_dir
    if log_dir is None:
        log_dir = Path(__file__).parent.parent.resolve() / "lightning_logs"
    tb_logger = get_pl_tb_logger(log_dir, args.experiment_name)
    log_dir = tb_logger.log_dir
    cache_dir = (
        Path(args.cache_dir)
        if args.cache_dir
        else Path(log_dir).parent.parent.parent / "cache"
    )
    cache_dir.mkdir(exist_ok=True, parents=True)

    label_enc = get_label_encoder(args.trained_model_path)

    ds = IAMDataset(
        args.iam_dir,
        "word",
        "train",
        label_enc=label_enc,
        return_writer_id=True,
        return_writer_id_as_idx=True,
        only_lowercase=base_hparams["only_lowercase"],
    )

    _, ds_val, ds_test = prepare_iam_splits(
        ds, Path(__file__).resolve().parent.parent / "htr/aachen_splits"
    )

    # Exclude writers from the dataset that do not have sufficiently many samples.
    # For the WriterCodeAdaptive model, there is no support/query split performed.
    # Therefore, limit the size of the train batch to half of what if would be if
    # this split was done.
    shots, ways = map(int, (hparams["shots"], hparams["ways"]))
    ds_val.data = filter_df_by_freq(ds_val.data, "writer_id", shots * 2)
    ds_test.data = filter_df_by_freq(ds_test.data, "writer_id", shots * 2)

    # Set image transforms.
    ds_val.set_transforms_for_split("val")
    ds_test.set_transforms_for_split("test")

    # Initialize learn2learn tasksets.
    taskset_val = prepare_test_taskset(ds_val)
    taskset_test = prepare_test_taskset(ds_test)

    # Remove overlap with PL arguments.
    hparams.pop("max_epochs")
    hparams.pop("seed")
    assert "inner_lr" in hparams  # sanity check

    # Define model arguments.
    args_ = dict(
        checkpoint_path=args.trained_model_path,
        model_hparams_file=base_hparams_path,
        label_encoder=ds_val.label_enc,
        load_meta_weights=True,
        num_writers=len(ds_val.writer_ids),
        taskset_val=taskset_val,
        taskset_test=taskset_test,
        **vars(args),
        **hparams,
    )
    # Initialize with a trained base model.
    cls = main_lit_models()[hparams["main_model_arch"]]
    learner = cls.init_with_base_model_from_checkpoint(**args_)

    callbacks = [
        ModelSummary(max_depth=3),
        LitProgressBar(),
    ]

    trainer = Trainer.from_argparse_args(
        args,
        logger=tb_logger,
        gpus=(0 if args.use_cpu else 1),
        log_every_n_steps=10,
        enable_model_summary=False,
        callbacks=callbacks,
    )

    if args.test:  # run on test set
        trainer.test(learner)
    else:  # run on validation set
        trainer.validate(learner)


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser()

    parser.add_argument("--trained_model_path", type=str, required=True, metavar="PATH",
                        help="Path to a base model checkpoint")
    parser.add_argument("--iam_dir", type=str, required=True, metavar="PATH",
                        help="IAM dataset root folder.")
    parser.add_argument("--cache_dir", type=str, metavar="PATH")
    parser.add_argument("--log_dir", type=str, metavar="PATH",
                        help="Directory where the lighning logs will be stored.")
    parser.add_argument("--test", action="store_true", default=False,
                        help="Use the test set rather than the val set.")
    parser.add_argument("--use_cpu", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Experiment name, used as the name of the folder in "
                             "which logs are stored.")
    # fmt: on

    parser = Trainer.add_argparse_args(parser)  # adds Pytorch Lightning arguments

    args = parser.parse_args()
    main(args)
