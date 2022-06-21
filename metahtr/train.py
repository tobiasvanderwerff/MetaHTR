import argparse
from pathlib import Path

from metahtr.lit_models import (
    LitMAMLLearner,
    LitBaseEpisodic,
    LitMetaHTR,
)
from metahtr.util import (
    filter_df_by_freq,
    get_label_encoder,
    save_label_encoder,
    get_pl_tb_logger,
    copy_hyperparameters_to_logging_dir,
    prepare_iam_splits,
    prepare_train_taskset,
    main_lit_models,
    get_parameter_names,
    prepare_test_taskset,
    load_best_pl_checkpoint,
    MAMLHTRCheckpointIO,
    MainModelArch,
    BaseModelArch,
)

from htr.data import IAMDataset
from htr.util import LitProgressBar

import torch
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, ModelSummary
from pytorch_lightning.plugins import DDPPlugin


def main(args: argparse.Namespace):

    print(f"Main model used: {str(args.main_model_arch).upper()}")
    print(f"Base model used: {str(args.base_model_arch).upper()}")

    if args.base_model_arch == "sar":
        # Disable CuDDN when using LSTM base model (SAR). This is necessary because
        # CuDNN does not support calculating second derivatives for RNNs.
        torch.backends.cudnn.enabled = False

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
    save_label_encoder(label_enc, log_dir)
    model_hparams_file, hparams = copy_hyperparameters_to_logging_dir(
        args.trained_model_path, log_dir
    )

    only_lowercase = hparams["only_lowercase"]
    augmentations = "train" if args.use_image_augmentations else "val"

    # Initialize IAM dataset.
    ds = IAMDataset(
        args.iam_dir,
        parse_method="word",  # use word images
        split="train",
        label_enc=label_enc,
        return_writer_id=True,
        return_writer_id_as_idx=True,
        only_lowercase=only_lowercase,
    )

    # Get Aachen splits.
    ds_train, ds_val, ds_test = prepare_iam_splits(
        ds, Path(__file__).resolve().parent.parent / "htr/aachen_splits"
    )

    # Exclude writers from the dataset that do not have sufficiently many samples for
    # meta-learning.
    ds_train.data = filter_df_by_freq(ds_train.data, "writer_id", args.shots * 2)
    ds_val.data = filter_df_by_freq(ds_val.data, "writer_id", args.shots * 2)
    ds_test.data = filter_df_by_freq(ds_test.data, "writer_id", args.shots * 2)

    # Set image transforms.
    ds_train.set_transforms_for_split(augmentations)
    ds_val.set_transforms_for_split("val")
    ds_test.set_transforms_for_split("test")

    print("Dataset split sizes:")
    print(f"train:\t{len(ds_train)}")
    print(f"val:\t{len(ds_val)}")
    print(f"test:\t{len(ds_test)}")

    # Initialize learn2learn tasksets.
    print("Initializing tasksets. This can take some time the first time it is done.")
    shots, ways = args.shots, args.ways
    taskset_train = prepare_train_taskset(
        dataset=ds_train,
        ways=ways,
        bookkeeping_path=cache_dir / f"train_l2l_bookkeeping_shots={shots}.pkl",
        shots=shots * 2,
    )
    taskset_val = prepare_test_taskset(ds_val)
    taskset_test = prepare_test_taskset(ds_test)

    # Define model arguments.
    args_ = dict(
        checkpoint_path=args.trained_model_path,
        model_hparams_file=model_hparams_file,
        label_encoder=ds_train.label_enc,
        model_params_to_log={"only_lowercase": only_lowercase},
        num_writers=len(ds_train.writer_ids),
        taskset_train=taskset_train,
        taskset_val=taskset_val,
        taskset_test=taskset_test,
        # allow_nograd=args.freeze_batchnorm_gamma,
        prms_to_log={
            "main_model_arch": args.main_model_arch,
            "base_model_arch": args.base_model_arch,
            "seed": args.seed,
            "model_path": args.trained_model_path,
            "early_stopping_patience": args.early_stopping_patience,
            "use_image_augmentations": args.use_image_augmentations,
            "weight_decay": args.weight_decay,
        },
        **vars(args),
    )

    # Initialize MetaHTR/MAML with a trained base model.
    cls = main_lit_models()[args.main_model_arch]
    learner = cls.init_with_base_model_from_checkpoint(**args_)

    # Configure checkpoint saving.
    base_model_params = get_parameter_names(args.trained_model_path)
    plugins = [MAMLHTRCheckpointIO(base_model_params)]

    # Set callbacks.
    callbacks = [
        ModelSummary(max_depth=3),
        LitProgressBar(),
        ModelCheckpoint(
            save_top_k=(-1 if args.save_all_checkpoints else 3),
            mode="min",
            monitor="word_error_rate",
            filename=args.main_model_arch
            + "-{epoch}-{char_error_rate:.4f}-{word_error_rate:.4f}",
            save_weights_only=True,
        ),
    ]
    callbacks = learner.add_model_specific_callbacks(
        callbacks,
        shots=args.shots,
        ways=args.ways,
        label_encoder=ds_train.label_enc,
        is_train=not (args.validate or args.test),
    )
    if args.early_stopping_patience != -1:
        callbacks.append(
            EarlyStopping(
                monitor="word_error_rate",
                patience=args.early_stopping_patience,
                verbose=True,
                mode="min",
            )
        )

    trainer = Trainer.from_argparse_args(
        args,
        logger=tb_logger,
        plugins=plugins,
        strategy=(
            DDPPlugin(find_unused_parameters=False) if args.num_nodes != 1 else None
        ),  # ddp = Distributed Data Parallel
        gpus=(0 if args.use_cpu else 1),
        log_every_n_steps=10,
        enable_model_summary=False,
        callbacks=callbacks,
    )

    if args.validate:  # validate a trained model
        trainer.validate(learner)
    elif args.test:  # test a trained model
        trainer.test(learner)
    else:  # train a model
        trainer.fit(learner)

    if args.test_on_fit_end:
        load_best_pl_checkpoint(trainer, learner, ds_train.label_enc)
        trainer.test(learner)


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser()

    parser.add_argument("--trained_model_path", type=str, required=True, metavar="PATH",
                        help="Path to a base model checkpoint")
    parser.add_argument("--iam_dir", type=str, required=True, metavar="PATH",
                        help="IAM dataset root folder.")
    parser.add_argument("--main_model_arch", type=str, required=True,
                        choices=MainModelArch.get_vals(), default="metahtr")
    parser.add_argument("--base_model_arch", type=str, required=True,
                        choices=BaseModelArch.get_vals(), default="fphtr")
    parser.add_argument("--cache_dir", type=str, metavar="PATH")
    parser.add_argument("--log_dir", type=str, metavar="PATH",
                        help="Directory where the lighning logs will be stored.")
    parser.add_argument("--validate", action="store_true", default=False)
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--early_stopping_patience", type=int, default=10,
                        help="Number of checks with no improvement after which "
                             "training will be stopped. Setting this to -1 will disable "
                             "early stopping.")
    parser.add_argument("--use_image_augmentations", action="store_true", default=False,
                        help="Whether to use image augmentations during training.")
    parser.add_argument("--save_all_checkpoints", action="store_true", default=False)
    parser.add_argument("--test_on_fit_end", action="store_true", default=False,
                        help="Run test on the best model checkpoint after training.")
    parser.add_argument("--use_cpu", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Experiment name, used as the name of the folder in "
                             "which logs are stored.")
    # fmt: on

    parser = LitBaseEpisodic.add_model_specific_args(parser)
    parser = LitMAMLLearner.add_model_specific_args(parser)
    parser = LitMetaHTR.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)  # adds Pytorch Lightning arguments

    args = parser.parse_args()
    main(args)
