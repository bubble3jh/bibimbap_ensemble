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

## Model Bibimbap strategy
Model Bibimbap ensembles fine-tuned models in three steps.

#### 1. Load pre-trained model and fine-tune
Under 'model_gin/', there are pre-trained models that are trained in different ways. In INPUT_MODEL_PATH, put the path to the selected model.

```
python finetune.py --input_model_file PRETRAINED_MODEL_PATH --filename OUTPUT_FILENAME --dataset DATASET --epochs EPOCHS
```

We can give as a hyperparameter whether to freeze the batch normalization layer when training the model with `--freeze_bn`.
This step creates a vanilla fine-tuned model from the pretrained model for each datasets.

#### 2. Linear probing for target dataset
```
python finetune.py --input_model_file PRETRAINED_MODEL_PATH --filename OUTPUT_FILENAME --dataset DATASET --epochs EPOCHS --freeze_gnn
```
In this step, load the pre-trained model in `PRETRAINED_MODEL_PATH` further we run linear probing on the target dataset to obtain weights to replace the linear classifier in the auxiliary model.

#### 3. Auxiliary fine-tune
```
python3 finetune.py --input_model_file FINETUNED_MODEL_PATH --replace_classifier LP_MODEL_PATH --dataset DATASET --filename OUTPUT_FILENAME --epochs EPOCHS 

```

At this stage, you can also use `--freeze_bn` to decide whether to learn or not.
Also, for ease of experimentation, we used the following form of OUTPUT_FILENAME. `target_${target}_aux_${aux}_${model_ver}`
For ${target}, we wrote the name of the target dataset, for ${aux}, the name of the auxiliary dataset, and ${model_ver} the string such as 'fr_nfr', which summarizes whether the model was trained by running freeze_bn in the previous step and the current step.

#### 4. Ensemble and final fine-tune
```
python finetune.py --averaging_target DATASET --dataset DATASET --averaging_aux AUX_MODELS --epochs EPOCHS --model_ver MODEL_VER --ensemble_method uniform
```

`DATASET` should be target dataset, and `AUX_MODELS` should be auxiliary model's names. We added `model_ver` argument parameter for find auxiliary model's directory more easier.

At this stage, we tried a combination of `--freeze_gnn` freezing the features, `--freeze_lc` freezing the classifier, and `--freeze_bn` to find a best performing model for each dataset with hyperparameter tuning.

Furthermroe, we can select ensemble method with `--ensemble_method uniform` or `--ensemble_method dirichelt` and can select ensemble weight manualy with `ens_weight` argument.
