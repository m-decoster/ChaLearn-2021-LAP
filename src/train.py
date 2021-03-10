"""Train a neural network on a given dataset with fixed hyperparameters.
For tuning of the hyperparameters, see tune.py."""

import importlib
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from models import module

if __name__ == '__main__':
    # -------------------------------- #
    # ARGUMENT PARSING
    # -------------------------------- #
    parser = ArgumentParser()

    # Program specific
    parser.add_argument('--log_dir', type=str, help='Directory to which experiment logs will be written', required=True)
    parser.add_argument('--seed', type=int, help='Random seed', default=42)
    parser.add_argument('--dataset', type=str, help='Dataset module', required=True)

    program_args, _ = parser.parse_known_args()

    # Model specific
    parser = module.get_model_def().add_model_specific_args(parser)

    # Data module specific
    data_module = importlib.import_module(f'datasets.{program_args.dataset}')
    parser = data_module.get_datamodule_def().add_datamodule_specific_args(parser)

    # Trainer specific
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    # -------------------------------- #
    # SETUP
    # -------------------------------- #
    pl.seed_everything(args.seed)

    trainer = pl.Trainer(callbacks=[
        EarlyStopping(monitor='val_loss', mode='min', verbose=True, patience=10),
        LearningRateMonitor(logging_interval='epoch')
    ], logger=TensorBoardLogger(args.log_dir, name=args.model),
        fast_dev_run=args.fast_dev_run,
        track_grad_norm=args.track_grad_norm,
        gradient_clip_val=args.gradient_clip_val,
        log_gpu_memory=args.log_gpu_memory,
        overfit_batches=args.overfit_batches,
        accumulate_grad_batches=args.accumulate_grad_batches,
        val_check_interval=args.val_check_interval,
        profiler=args.profiler,
        progress_bar_refresh_rate=args.progress_bar_refresh_rate,
        gpus=args.gpus)

    # -------------------------------- #
    # FITTING THE MODEL
    # -------------------------------- #
    dict_args = vars(args)

    model = module.get_model(**dict_args)
    dm = data_module.get_datamodule(**dict_args)

    train_results = trainer.fit(model, dm)
