# Model Bibimbap Ensemble Framework

This is a Pytorch implementation of the following paper: 

Jinho Kang, Taero Kim, Jiyoung Jung, Rakwoo Chang, Kyungwoo Song. Bibimbap : Ensembling Diverse Pre-trained Models for Domain Generalization in  Domain Shifted Task. PRL 2023.
[arXiv]() [OpenReview]() 

If you make use of the code/experiment in your work, please cite our paper (Bibtex below).

```
Bibtex here
```

## Installation
We used the following Python packages for core development. We tested on `Python 3.8.15`.
```
pytorch                   1.1.0
torch-cluster             1.6.0             
torch-geometric           2.2.0
torch-scatter             2.0.9
torch-sparse              0.6.13
torch-spline-conv         1.2.1
rdkit                     2022.9.3
tqdm                      4.62.3
tensorboardx              2.5.1
```

## Dataset download
All the necessary data files can be downloaded from the following links.

download from [chem data](http://snap.stanford.edu/gnn-pretrain/data/chem_dataset.zip) (2.5GB), unzip it, and put it under `dataset/`.

## Pre-training and fine-tuning
In each directory, we have three kinds of files used to train GNNs.

#### 1. Self-supervised pre-training
```
python pretrain_contextpred.py --output_model_file OUTPUT_MODEL_PATH
python pretrain_masking.py --output_model_file OUTPUT_MODEL_PATH
python pretrain_edgepred.py --output_model_file OUTPUT_MODEL_PATH
python pretrain_deepgraphinfomax.py --output_model_file OUTPUT_MODEL_PATH
```
This will save the resulting pre-trained model to `OUTPUT_MODEL_PATH`.

#### 2. Supervised pre-training
```
python pretrain_supervised.py --output_model_file OUTPUT_MODEL_PATH --input_model_file INPUT_MODEL_PATH
```
This will load the pre-trained model in `INPUT_MODEL_PATH`, further pre-train it using supervised pre-training, and then save the resulting pre-trained model to `OUTPUT_MODEL_PATH`.

#### 3. Fine-tuning
```
python finetune.py --model_file INPUT_MODEL_PATH --dataset DOWNSTREAM_DATASET --filename OUTPUT_FILE_PATH
```
This will finetune pre-trained model specified in `INPUT_MODEL_PATH` using dataset `DOWNSTREAM_DATASET.` The result of fine-tuning will be saved to `OUTPUT_FILE_PATH.`

## Saved pre-trained models
We release pre-trained models in `model_gin/` and `model_architecture/` for both biology (`bio/`) and chemistry (`chem/`) applications. Feel free to take the models and use them in your applications!

## Reproducing results in the paper
Our results in the paper can be reproduced by running `sh finetune_tune.sh SEED DEVICE`, where `SEED` is a random seed ranging from 0 to 9, and `DEVICE` specifies the GPU ID to run the script. This script will finetune our saved pre-trained models on each downstream dataset.
