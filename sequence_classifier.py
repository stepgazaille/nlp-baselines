from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from os import cpu_count, environ
from datetime import datetime
from pathlib import Path
import json
import torch
from torch import Tensor, argmax
from torch.utils.data import DataLoader, random_split
from torch.nn import CrossEntropyLoss
from torch.optim import Optimizer
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from datasets import load_dataset, load_metric
from datasets.arrow_dataset import Batch
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.preprocessing import LabelBinarizer
environ['TOKENIZERS_PARALLELISM'] = 'false'


class Dataset(LightningDataModule):

	def __init__(self, dataset_name_or_path: str,
					   model_name_or_path: str,
					   batch_size: int) -> None:
		super().__init__()
		self.save_hyperparameters()
		self.dataset = load_dataset(dataset_name_or_path)		
		self.label_names = self.dataset['train'].features['label'].names
		self.batch_size = batch_size
		self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
		self.label_binarizer = LabelBinarizer()

	def setup(self, stage: str = None) -> None:		
		self.label_binarizer.fit(range(len(self.label_names)))
		self.dataset = self.dataset.map(self.preproc,
										batched=True,
										batch_size=self.batch_size,
										remove_columns=['text', 'label'])
		self.dataset.set_format(type='torch')
		train_split_size = len(self.dataset['train']) - len(self.dataset['test'])
		val_split_size = len(self.dataset['test'])
		self.dataset['train'], self.dataset['val'] = random_split(self.dataset['train'], [train_split_size, val_split_size])		

	def preproc(self, batch: Batch) -> dict:
		input_ids = self.tokenizer(batch['text'], padding=True)['input_ids']
		labels = self.label_binarizer.transform(batch['label']).astype(float)		
		return {'input_ids': input_ids, 'labels': labels}
	
	def train_dataloader(self) -> DataLoader:
		return DataLoader(self.dataset['train'], batch_size=self.batch_size, num_workers=args.num_workers, shuffle=True)
	
	def val_dataloader(self) -> DataLoader:
		return DataLoader(self.dataset['val'], batch_size=self.batch_size, num_workers=args.num_workers)
		
	def test_dataloader(self) -> DataLoader:
		return DataLoader(self.dataset['test'], batch_size=self.batch_size, num_workers=args.num_workers)


class SequenceClassifier(LightningModule):
	
	def __init__(self, model_name_or_path: str,
					   label_names: list[str]) -> None:
		super().__init__()
		self.save_hyperparameters()
		self.label_names = label_names
		self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=len(self.label_names))
		self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=self.config)
		self.loss_function = CrossEntropyLoss()
		experiment_id = datetime.now()
		self.f1 = load_metric('f1', experiment_id=experiment_id)
		self.accuracy = load_metric('accuracy', experiment_id=experiment_id)
							
	def forward(self, input_ids: Tensor) -> SequenceClassifierOutput:
		return self.model(input_ids)

	def configure_optimizers(self) -> Optimizer:
		return  torch.optim.Adam(self.parameters(), lr=.0002)

	def training_step(self, batch: dict, batch_idx: int) -> Tensor:
		logits  = self(batch['input_ids'])['logits']
		loss = self.loss_function(logits, batch['labels'])		
		self.log('train_loss', loss)
		return loss

	def validation_step(self, batch: dict, batch_idx: int) -> Tensor:
		logits  = self(batch['input_ids'])['logits']
		loss = self.loss_function(logits, batch['labels'])		
		self.log('val_loss', loss)	
		return loss

	def on_test_epoch_start(self) -> None:
		self.test_preds = []
		self.test_labels = []

	def test_step(self, batch: dict, batch_idx: int) -> None:
		logits  = self(batch['input_ids'])['logits']
		self.test_preds += argmax(logits, axis=1).tolist()
		self.test_labels += argmax(batch['labels'], axis=1).tolist()
	
	def on_test_epoch_end(self) -> None:
		self.log_dict(self.f1.compute(predictions=self.test_preds, references=self.test_labels, average='micro'))
		self.log_dict(self.f1.compute(predictions=self.test_preds, references=self.test_labels, average=None, labels=self.label_names))
		self.log_dict(self.accuracy.compute(predictions=self.test_preds, references=self.test_labels))


if __name__ == '__main__':

	arg_parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
	arg_parser.add_argument('-d', '--dataset', default='ag_news', choices=['ag_news'], type=str, help="The dataset ID of a sequence classification dataset hosted inside a dataset repo on huggingface.co")
	arg_parser.add_argument('-m', '--model-name-or-path', default='distilbert-base-uncased', type=str, help="The model ID of a pretrained model hosted inside a model repo on huggingface.co")
	arg_parser.add_argument('-g', '--gpus', default='[0]', type=str, help="Which GPUs to train on.")
	arg_parser.add_argument('-l', '--log-dir', default='./', type=str, help="Default path for logs and weights ")
	arg_parser.add_argument('-b', '--batch-size', default=2, type=int, help="How many samples per batch to load.")
	arg_parser.add_argument('-e', '--max-epochs', default=5, type=int, help="Stop training once this number of epochs is reached.")
	arg_parser.add_argument('-w', '--num-workers', default=cpu_count(), type=int, help="Number of subprocesses to use for data loading.")
	arg_parser.add_argument('-s', '--seed', default=42, type=int, help="The integer value seed for global random state.")
	arg_parser.add_argument('-p', '--precision', default=16, choices=[64, 32, 16], type=int, help="Use double precision (64), full precision (32) or half precision (16).")
	args = arg_parser.parse_args()
	args.log_dir = Path(args.log_dir).expanduser().resolve()
	args.gpus = json.loads(args.gpus)
	seed_everything(args.seed)

	print("Preparing the data...")
	dataset = Dataset(args.dataset, args.model_name_or_path, args.batch_size)
	dataset.setup()
	print(my_batch)

	print("Preparing the model...")
	model = SequenceClassifier(args.model_name_or_path, dataset.label_names)
	trainer = Trainer(accelerator='gpu',
					  gpus=args.gpus,
					  max_epochs=args.max_epochs,
					  precision=args.precision,					  
					  default_root_dir=args.log_dir,
					  callbacks=[EarlyStopping(monitor='val_loss', mode='min')])

	print("Start training the model...")
	training_start_time = datetime.now()
	trainer.fit(model, dataset.train_dataloader(), dataset.val_dataloader())
	print(f"Training completed! Duration:{datetime.now() - training_start_time}")

	print("Evaluating the best model...")
	trainer.test(ckpt_path='best', dataloaders=dataset.test_dataloader())
