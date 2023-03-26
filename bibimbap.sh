#!/bin/bash
FILE="done"
epochs=500
# First fine-tuning step
for dataset in "tox21" "hiv" "muv" "bace" "bbbp" "toxcast" "sider" "clintox"
do
    dir=PATH
    if [ ! -d $dir ]; then 
        mkdir $dir
    fi
    if  [ ! -e $FILE ]; then
       python finetune.py --input_model_file "/model_gin/contextpred.pth" --dataset ${dataset} --epochs ${epochs} # --freeze_bn
    fi

done

#Linear probing for target dataset
dir=LP_PATH
if [ ! -d $dir ]; then 
    mkdir $dir
fi
if  [ ! -e $FILE ]; then
    python finetune.py --input_model_file "/model_gin/contextpred.pth" --dataset ${dataset} --epochs ${epochs}
fi

# Second fine-tuning step for make auxiliary model
datasets=("tox21" "hiv" "muv" "bace" "bbbp" "toxcast" "sider" "clintox")
target=TARGET
model_ver=MODEL_VER
for aux in ${datasets[@]/${target}}
do
    dir=FINETUNE_PATH
    if [ ! -d $dir ]; then 
        mkdir $dir
    fi
    if  [ ! -e $FILE ]; then
        python finetune.py --input_model_file AUX_MODEL_PATH --replace_classifier ${target}_LP_PATH --dataset ${target} --filename target_${target}_aux_${aux}_${model_ver} --epochs ${epochs} # --freeze_bn
    fi
done

# Bibimbap final fine-tune
python finetune.py --averaging_target ${target} --averaging_aux ${datasets[@]/${target}} --dataset ${target} --epochs ${epochs} --model_ver ${model_ver} # --freeze_bn

