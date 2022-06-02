import json
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from os import cpu_count, environ
from datetime import datetime
from pathlib import Path
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from data import Dataset
from models import SequenceClassifier

environ["TOKENIZERS_PARALLELISM"] = "false"


if __name__ == "__main__":

    arg_parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    arg_parser.add_argument(
        "-d",
        "--dataset",
        default="ag_news",
        choices=["ag_news", "banking77"],
        type=str,
        help="The dataset ID of a sequence classification dataset hosted inside a dataset repo on huggingface.co",
    )
    arg_parser.add_argument(
        "-m",
        "--model-name-or-path",
        default="distilbert-base-uncased",
        type=str,
        help="The model ID of a pretrained model hosted inside a model repo on huggingface.co",
    )
    arg_parser.add_argument(
        "-g", "--gpus", default="[0]", type=str, help="Which GPUs to train on"
    )
    arg_parser.add_argument(
        "-l",
        "--log-dir",
        default="sequence_classification_logs",
        type=str,
        help="Default path for logs and weights",
    )
    arg_parser.add_argument(
        "-b",
        "--batch-size",
        default=8,
        type=int,
        help="How many samples per batch to load",
    )
    arg_parser.add_argument(
        "-e",
        "--max-epochs",
        default=100,
        type=int,
        help="Stop training once this number of epochs is reached",
    )
    arg_parser.add_argument(
        "-r", "--learning-rate", default=2.0e-5, type=float, help="Learning rate"
    )
    arg_parser.add_argument(
        "--auto-lr-find",
        default=False,
        action="store_true",
        help="Try to optimize initial learning rate for faster convergence",
    )
    arg_parser.add_argument(
        "-w",
        "--num-workers",
        default=cpu_count(),
        type=int,
        help="Number of subprocesses to use for data loading",
    )
    arg_parser.add_argument(
        "-s",
        "--seed",
        default=42,
        type=int,
        help="The integer value seed for global random state",
    )
    arg_parser.add_argument(
        "-p",
        "--precision",
        default=16,
        choices=[64, 32, 16],
        type=int,
        help="Use double precision (64), full precision (32) or half precision (16)",
    )
    args = arg_parser.parse_args()
    args.gpus = json.loads(args.gpus)
    args.log_dir = Path(args.log_dir).expanduser().resolve()
    logger = TensorBoardLogger(save_dir=args.log_dir, name=str(args.dataset))
    seed_everything(args.seed)

    # Preparing the model, data, and trainer:
    dataset = Dataset(
        args.dataset, args.model_name_or_path, args.batch_size, args.num_workers
    )
    model = SequenceClassifier(
        args.learning_rate, args.model_name_or_path, dataset.label_names
    )
    trainer = Trainer(
        accelerator="gpu",
        gpus=args.gpus,
        max_epochs=args.max_epochs,
        precision=args.precision,
        default_root_dir=args.log_dir,
        auto_lr_find=args.auto_lr_find,
        logger=logger,
        callbacks=[EarlyStopping(monitor="val_loss", mode="min")],
    )

    if args.auto_lr_find:
        trainer.tune(model, datamodule=dataset)
    else:
        dataset.setup()

    # Document hyper-parameters:
    args.learning_rate = model.learning_rate
    hparams = {k: vars(args)[k] for k in vars(args) if k != "log_dir"}
    logger.log_hyperparams(hparams)

    # Start training the model:
    training_start_time = datetime.now()
    trainer.fit(model, datamodule=dataset)
    print(f"Training completed! Duration:{datetime.now() - training_start_time}")

    print("Evaluating the best model...")
    trainer.test(ckpt_path="best", datamodule=dataset)
