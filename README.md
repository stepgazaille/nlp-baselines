# NLP baselines
A collection of baselines for various NLP tasks.

## Installation
1. Download and install the latest [Anaconda Python distribution](https://www.anaconda.com/distribution/#download-section)
2. Execute the following commands to install all software requirements:
```
cd nlp_baselines
conda env create
```

## Usage
```
$ python sequence_classifier.py -h
usage: sequence_classifier.py [-h] [-d DATASET] [-m MODEL_NAME_OR_PATH] [-b BATCH_SIZE] [-e MAX_EPOCHS] [-w NUM_WORKERS]

optional arguments:
  -h, --help            show this help message and exit
  -d DATASET, --dataset DATASET
                        Dataset on which to train the classifier on (default: ag_news)
  -m MODEL_NAME_OR_PATH, --model_name_or_path MODEL_NAME_OR_PATH
                        Pretrained model to use in the classifier (default: distilbert-base-uncased)
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size (default: 1)
  -e MAX_EPOCHS, --max-epochs MAX_EPOCHS
                        Number of epochs (default: 10)
  -w NUM_WORKERS, --num-workers NUM_WORKERS
                        Number of workers (default: 4)
```
