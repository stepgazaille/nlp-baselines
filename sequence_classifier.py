
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from os import cpu_count
from datetime import datetime

import torch
from torch.utils.data import DataLoader, random_split
from torch.nn import CrossEntropyLoss
from torch.nn.functional import one_hot
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from datasets import load_dataset, load_metric
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification


class Dataset(LightningDataModule):
	def __init__(self, dataset_name_or_path = 'ag_news', batch_size = 1):
		super().__init__()
		self.dataset_name_or_path = dataset_name_or_path
		self.batch_size = batch_size
		
	def prepare_data(self):
		# Download dataset if not already done:
		load_dataset(self.dataset_name_or_path)
	
	
	def setup(self, stage):
		
		self.dataset = load_dataset(self.dataset_name_or_path)
		if stage in (None, 'fit'):
			self.train_split, self.val_split = random_split(self.dataset['train'], [120000 - 7600, 7600])
		
		if stage in (None, 'test'):
			self.test_split = self.dataset['test']
		
	
	def train_dataloader(self):
		return DataLoader(self.train_split, num_workers=args.num_workers)
		
	def val_dataloader(self):
		return DataLoader(self.val_split, num_workers=args.num_workers)
		
	def test_dataloader(self):
		return DataLoader(self.test_split, num_workers=args.num_workers)


class SequenceClassifier(LightningModule):
	
	def __init__(self):
		super().__init__()
		self.save_hyperparameters()
		model_name_or_path='distilbert-base-uncased'
		self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=4)
		self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
		self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=self.config)
		self.loss_function = CrossEntropyLoss()
		self.metric = load_metric('f1', experiment_id=datetime.now())
							
	def forward(self, x):
		ids = self.tokenizer(x)['input_ids']
		ids = torch.LongTensor(ids).to(device='cuda')
		outputs = self.model(ids)
		return outputs['logits']

	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=.0005)
		return optimizer

	def training_step(self, batch, batch_idx):
		x, y = batch['text'], batch['label']
		y_hat  = self(x)
		one_hot_y = one_hot(y, num_classes=4).double()
		loss = self.loss_function(y_hat, one_hot_y)		
		self.log('train_loss', loss)
		return loss

	def validation_step(self, batch, batch_idx):
		x, y = batch['text'], batch['label']
		y_hat  = self(x)
		one_hot_y = one_hot(y, num_classes=4).double()
		loss = self.loss_function(y_hat, one_hot_y)		
		self.log('val_loss', loss)	
		preds = torch.argmax(y_hat, axis=1)
		return {'loss': loss, 'preds': preds, 'labels': y}


if __name__ == '__main__':

	arg_parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
	arg_parser.add_argument('-e', '--max-epochs', default=10, type=int, help="Number of epochs")
	arg_parser.add_argument('-w', '--num-workers', default=cpu_count(), type=int, help="Number of workers")
	args = arg_parser.parse_args()
	seed_everything(42)

	dataset = Dataset(dataset_name_or_path = 'ag_news', batch_size = 1)
	model = SequenceClassifier()
	trainer = Trainer(accelerator='gpu',
					  auto_select_gpus=True,
					  precision=16,
					  max_epochs=args.max_epochs,
					  default_root_dir='./',
					  callbacks=[EarlyStopping(monitor='val_loss', mode='min')])
	
	
	training_start_time = datetime.now()
	trainer.fit(model, datamodule=dataset)
	training_end_time = datetime.now()
	print("Training comleted!")
	print(f"Training time: {training_end_time - training_start_time}")
