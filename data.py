from datasets import DatasetDict, load_dataset
from datasets.arrow_dataset import Batch
from pytorch_lightning import LightningDataModule
from sklearn.preprocessing import LabelBinarizer
from transformers import AutoTokenizer
from torch.utils.data import DataLoader


class Dataset(LightningDataModule):

	def __init__(self, dataset_name_or_path: str,
					   model_name_or_path: str,
					   batch_size: int,
					   num_workers: int) -> None:
		super().__init__()
		self.dataset = load_dataset(dataset_name_or_path)		
		self.label_names = self.dataset['train'].features['label'].names
		self.batch_size = batch_size
		self.num_workers = num_workers
		self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
		self.label_binarizer = LabelBinarizer()

	def setup(self, stage: str = None) -> None:		
		self.label_binarizer.fit(range(len(self.label_names)))
		val_split_size = len(self.dataset['test'])
		new_train_split_size = len(self.dataset['train']) - val_split_size	
		new_splits = self.dataset['train'].train_test_split(val_split_size, new_train_split_size)
		self.dataset = DatasetDict({'train': new_splits['train'],
									'val': new_splits['test'],
									'test': self.dataset['test']})
		self.dataset = self.dataset.map(self.preproc,
										batched=True,
										batch_size=self.batch_size,
										remove_columns=['text', 'label'])
		self.dataset.set_format(type='torch')			

	def preproc(self, batch: Batch) -> dict:
		input_ids = self.tokenizer(batch['text'], padding=True)['input_ids']
		labels = self.label_binarizer.transform(batch['label']).astype(float)		
		return {'input_ids': input_ids, 'labels': labels}
	
	def train_dataloader(self) -> DataLoader:
		return DataLoader(self.dataset['train'], batch_size=self.batch_size, num_workers=self.num_workers)
	
	def val_dataloader(self) -> DataLoader:
		return DataLoader(self.dataset['val'], batch_size=self.batch_size, num_workers=self.num_workers)
		
	def test_dataloader(self) -> DataLoader:
		return DataLoader(self.dataset['test'], batch_size=self.batch_size, num_workers=self.num_workers)
