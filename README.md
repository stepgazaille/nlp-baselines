# NLP baselines
A collection of baselines for various NLP tasks.

[EarlyStopping](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.callbacks.EarlyStopping.html?highlight=EarlyStopping) is used by default.

## Installation
1. Download and install the latest [Anaconda Python distribution](https://www.anaconda.com/distribution/#download-section)
2. Execute the following commands to install all software requirements:
```
cd nlp_baselines
conda env create
```

## Usage
### Sequence classification
```
$ python train_sequence_classifier.py -h
usage: train_sequence_classifier.py [-h] [-d {ag_news,banking77}] [-m MODEL_NAME_OR_PATH] [-g GPUS] [-l LOG_DIR] [-b BATCH_SIZE] [-e MAX_EPOCHS] [-r LEARNING_RATE] [--auto-lr-find] [-w NUM_WORKERS] [-s SEED]
                                    [-p {64,32,16}]

optional arguments:
  -h, --help            show this help message and exit
  -d {ag_news,banking77}, --dataset {ag_news,banking77}
                        The dataset ID of a sequence classification dataset hosted inside a dataset repo on huggingface.co (default: ag_news)
  -m MODEL_NAME_OR_PATH, --model-name-or-path MODEL_NAME_OR_PATH
                        The model ID of a pretrained model hosted inside a model repo on huggingface.co (default: distilbert-base-uncased)
  -g GPUS, --gpus GPUS  Which GPUs to train on (default: [0])
  -l LOG_DIR, --log-dir LOG_DIR
                        Default path for logs and weights (default: sequence_classification_logs)
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        How many samples per batch to load (default: 8)
  -e MAX_EPOCHS, --max-epochs MAX_EPOCHS
                        Stop training once this number of epochs is reached (default: 100)
  -r LEARNING_RATE, --learning-rate LEARNING_RATE
                        Learning rate (default: 2e-05)
  --auto-lr-find        Try to optimize initial learning rate for faster convergence (default: False)
  -w NUM_WORKERS, --num-workers NUM_WORKERS
                        Number of subprocesses to use for data loading (default: 40)
  -s SEED, --seed SEED  The integer value seed for global random state (default: 42)
  -p {64,32,16}, --precision {64,32,16}
                        Use double precision (64), full precision (32) or half precision (16) (default: 16)
```
To visualize logs (evaluation results can be found under the TEXT tab):
```
tensorboard --logdir [LOG_DIR]
# For example:
tensorboard --logdir sequence_classification_logs
```