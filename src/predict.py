"""Use a trained neural network to predict on a data set."""
import csv
import importlib
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch

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
    parser.add_argument('--checkpoint', type=str, help='Path to the checkpoint', required=True)
    parser.add_argument('--submission_template', type=str, help='Path to the submission template', required=True)
    parser.add_argument('--out', type=str, help='Output file path', required=True)

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

    dict_args = vars(args)

    model = module.get_model_def().load_from_checkpoint(args.checkpoint)
    dm = data_module.get_datamodule(**dict_args)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.to(device)
    model.eval()

    # -------------------------------- #
    # USING MODEL TO PREDICT
    # -------------------------------- #
    submission = dict()

    dataloader = dm.test_dataloader()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            x, paths = batch
            if isinstance(x, list):
                logits = model([e.to(device) for e in x]).cpu()
            else:
                logits = model(x.to(device)).cpu()

            predictions = torch.argmax(logits, dim=1)
            for j in range(logits.size(0)):
                submission[paths[j]] = predictions[j].item()

    with open(args.submission_template) as stf:
        reader = csv.reader(stf)
        with open(args.out, 'w') as of:
            writer = csv.writer(of)
            for row in reader:
                sample = row[0]
                print(f'Predicting {sample}', end=' ')
                print(f'as {submission[sample]}')
                writer.writerow([sample, submission[sample]])

    print(f'Wrote submission to {args.out}')
