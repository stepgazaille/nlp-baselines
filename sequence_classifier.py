
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from os import cpu_count, environ
from datetime import datetime
import torch
from torch.utils.data import DataLoader, random_split
from torch.nn import CrossEntropyLoss
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from datasets import load_dataset, load_metric
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelBinarizer
environ["TOKENIZERS_PARALLELISM"] = 'false'


class Dataset(LightningDataModule):

	def __init__(self, dataset_name_or_path, model_name_or_path, batch_size):
		super().__init__()
		self.save_hyperparameters()
		self.dataset_name_or_path = dataset_name_or_path
		self.batch_size = batch_size
		self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
		self.label_binarizer = LabelBinarizer()

	def setup(self, stage):
		self.dataset = load_dataset(self.dataset_name_or_path)
		self.label_binarizer.fit(range(4))
		self.dataset = self.dataset.map(self.convert_to_features,
										batched=True,
										batch_size=self.batch_size,
										remove_columns=['text', 'label'])
		self.dataset.set_format(type='torch')
		self.train_split, self.val_split = random_split(self.dataset['train'], [120000 - 7600, 7600])
		self.test_split = self.dataset['test']

	def convert_to_features(self,  dataset):
		input_ids = self.tokenizer(dataset['text'], padding=True)['input_ids']
		labels = self.label_binarizer.transform(dataset['label']).astype(float)
		return {'input_ids': input_ids, 'labels': labels}
	
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
		self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=self.config)
		self.loss_function = CrossEntropyLoss()
		self.metric = load_metric('f1', experiment_id=datetime.now())
							
	def forward(self, input_ids):
		return self.model(input_ids)

	def configure_optimizers(self):
		return torch.optim.Adam(self.parameters(), lr=.0005)

	def training_step(self, batch, batch_idx):
		logits  = self(batch['input_ids'])['logits']
		loss = self.loss_function(logits, batch['labels'])		
		self.log('train_loss', loss)
		return loss

	def validation_step(self, batch, batch_idx):
		logits  = self(batch['input_ids'])['logits']
		loss = self.loss_function(logits, batch['labels'])		
		self.log('val_loss', loss)	
		return loss

	def test_step(self, batch, batch_idx):
		# TODO
		logits  = self(batch['input_ids'])['logits']	
		preds = torch.argmax(logits, axis=1)
		return preds


if __name__ == '__main__':

	arg_parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
	arg_parser.add_argument('-d', '--dataset', default='ag_news', type=str, help="Dataset on which to train the classifier on")
	arg_parser.add_argument('-m', '--model_name_or_path', default='distilbert-base-uncased', type=str, help="Pretrained model to use in the classifier")
	arg_parser.add_argument('-b', '--batch_size', default=1, type=int, help="Batch size")
	arg_parser.add_argument('-e', '--max-epochs', default=100, type=int, help="Number of epochs")
	arg_parser.add_argument('-w', '--num-workers', default=cpu_count(), type=int, help="Number of workers")
	args = arg_parser.parse_args()
	seed_everything(42)

	dataset = Dataset(args.dataset, args.model_name_or_path, args.batch_size)
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
