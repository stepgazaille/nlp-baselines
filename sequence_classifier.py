
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from os import cpu_count
from datetime import datetime

import torch
from torch.utils.data import DataLoader, random_split
from torch.nn import CrossEntropyLoss
from torch.nn.functional import one_hot
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from datasets import load_dataset, load_metric
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification


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

	dataset_name_or_path = 'ag_news'
	dataset = load_dataset(dataset_name_or_path)
	train_split, val_split = random_split(dataset['train'], [120000 - 7600, 7600])
	test_split = dataset['test']

	train_loader = DataLoader(train_split, num_workers=args.num_workers)
	val_loader = DataLoader(val_split, num_workers=args.num_workers)
	test_loader = DataLoader(test_split, num_workers=args.num_workers)

	model = SequenceClassifier()
	trainer = Trainer(accelerator='gpu',
					  auto_select_gpus=True,
					  precision=16,
					  auto_scale_batch_size='binsearch',
					  max_epochs=args.max_epochs,
					  default_root_dir='./',
					  callbacks=[EarlyStopping(monitor='val_loss', mode='min')])
	
	
	training_start_time = datetime.now()
	trainer.fit(model, train_loader, test_loader)
	training_end_time = datetime.now()
	print("Training comleted!")
	print(f"Training time: {training_end_time - training_start_time}")
