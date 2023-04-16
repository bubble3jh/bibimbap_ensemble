#!/bin/bash
target=clintox
FILE="done"
epochs=500
datasets=("tox21" "hiv" "muv" "bace" "bbbp" "toxcast" "sider" "clintox")

case "$target" in
  "clintox")
    model_ver="fr_fr"
    bibim_fr="--freeze_lc"
    ;;
  "tox21")
    model_ver="fr_fr"
    bibim_fr="--freeze_lc"
    ;;
  "toxcast")
    model_ver="fr_fr"
    bibim_fr="--freeze_lc"
    ;;
  "sider")
    model_ver="fr_nfr"
    bibim_fr="--freeze_gnn"
    ;;
  "bace")
    model_ver="fr_fr"
    bibim_fr="--freeze_gnn"
    ;;
  "bbbp")
    model_ver="fr_nfr"
    bibim_fr="--freeze_lc"
    ;;
  "hiv")
    model_ver="fr_fr"
    bibim_fr="--freeze_lc"
    ;;
  "muv")
    model_ver="fr_fr"
    bibim_fr="--freeze_lc"
    ;;
  *)
    echo "Unknown target: $target"
    exit 1
    ;;
esac

# First fine-tuning step
for dataset in "${datasets[@]}"
do
    # The Path is example path with runseed 0. you can make it specific path for your own.
    dir="./runs/finetune_cls_runseed0/_${target}_fbn_${epochs}"
    if [ ! -d $dir ]; then 
        mkdir $dir
    fi
    if  [ ! -e $FILE ]; then
        python finetune.py --input_model_file "./model_gin/contextpred.pth" --dataset ${target} --epochs ${epochs} --freeze_bn 
    fi
done

#Linear probing for target dataset
dir="./runs/finetune_cls_runseed0/_${target}_fgnn_fbn_${epochs}"
if [ ! -d $dir ]; then 
    mkdir $dir
fi
if  [ ! -e $FILE ]; then
    python finetune.py --input_model_file "./model_gin/contextpred.pth" --dataset ${target} --epochs ${epochs} --freeze_gnn --freeze_bn
fi

# Second fine-tuning step for make auxiliary model
for aux in ${datasets[@]/${target}}
do
    AUX_MODEL_PATH="./runs/finetune_cls_runseed0/_${aux}_fbn_${epochs}/best_model.pth"
    TARGET_LP_PATH="./runs/finetune_cls_runseed0/_${target}_fgnn_fbn_${epochs}/best_model.pth"
    if [ "$model_ver" == "fr_fr" ]; then
        dir="./runs/finetune_cls_runseed0/_fbn_target_${target}_aux_${aux}__${epochs}"
            if [ ! -d $dir ]; then 
                mkdir $dir
            fi
            if  [ ! -e $FILE ]; then
                python finetune.py --input_model_file ${AUX_MODEL_PATH} --replace_classifier ${TARGET_LP_PATH} --dataset ${target} --filename _target_${target}_aux_${aux}_${model_ver} --epochs ${epochs} --freeze_bn
            fi
    elif [ "$model_ver" == "fr_nfr" ]; then
        dir="./runs/finetune_cls_runseed0/_target_${target}_aux_${aux}__${epochs}"
            if [ ! -d $dir ]; then 
                mkdir $dir
            fi
            if  [ ! -e $FILE ]; then
                python finetune.py --input_model_file ${AUX_MODEL_PATH} --replace_classifier ${TARGET_LP_PATH} --dataset ${target} --filename _target_${target}_aux_${aux}_${model_ver} --epochs ${epochs} # --freeze_bn
            fi
    else
        echo "model_ver is not valid"
    fi

done

# Bibimbap final fine-tune
python finetune.py --averaging_aux ${datasets[@]/${target}} --dataset ${target} --epochs ${epochs} --model_ver ${model_ver} --freeze_bn

# To reproduce paper's best models. use codes below.
# muv
python finetune.py --averaging_aux ${datasets[@]/${target}} --dataset ${target} --epochs ${epochs} --model_ver ${model_ver} --freeze_bn --freeze_lc --ens_weight 0.0127 0.2558 0.0913 0.0848 0.2740 0.0097 0.1378 0.1339
# tox21
python finetune.py --averaging_aux ${datasets[@]/${target}} --dataset ${target} --epochs ${epochs} --model_ver ${model_ver} --freeze_bn --freeze_lc --ens_weight 0.0853 0.2055 0.0528 0.1843 0.0386 0.2727 0.1289 0.0320
# toxcast
python finetune.py --averaging_aux ${datasets[@]/${target}} --dataset ${target} --epochs ${epochs} --model_ver ${model_ver} --freeze_bn --freeze_lc --ens_weight 0.0319 0.3417 0.0056 0.2945 0.1618 0.1133 0.0081 0.0431
# bace
python finetune.py --averaging_aux ${datasets[@]/${target}} --dataset ${target} --epochs ${epochs} --model_ver ${model_ver} --freeze_bn --freeze_gnn --ens_weight 0.0463 0.0822 0.0449 0.2312 0.1655 0.0912 0.0216 0.3170
# bbbp
python finetune.py --averaging_aux ${datasets[@]/${target}} --dataset ${target} --epochs ${epochs} --model_ver ${model_ver} --freeze_bn --freeze_lc --ens_weight 0.2972 0.0317 0.0108 0.0334 0.1812 0.0433 0.3481 0.0543
#sider
python finetune.py --averaging_aux ${datasets[@]/${target}} --dataset ${target} --epochs ${epochs} --model_ver ${model_ver} --freeze_bn --freeze_gnn --ens_weight 0.0061 0.2160 0.0847 0.2243 0.0125 0.1414 0.2218 0.0932
#clintox
python finetune.py --averaging_aux ${datasets[@]/${target}} --dataset ${target} --epochs ${epochs} --model_ver ${model_ver} --freeze_bn --freeze_lc --ens_weight 0.0064 0.1782 0.3595 0.0431 0.2378 0.0009 0.1277 0.0464
#hiv
python finetune.py --averaging_aux ${datasets[@]/${target}} --dataset ${target} --epochs ${epochs} --model_ver ${model_ver} --freeze_bn --freeze_lc --ens_weight 0.2399 0.0885 0.2300 0.0241 0.0385 0.1345 0.1285 0.1160
