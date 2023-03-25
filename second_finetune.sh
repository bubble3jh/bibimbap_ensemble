#!/bin/bash
FILE="done"
count=0
epochs=500
datasets=("tox21" "hiv" "muv" "bace" "bbbp" "toxcast" "sider" "clintox")
# read target_dataset
target_dataset=muv
model_ver=fr_fr
python3 notify.py --msg "$target_dataset $model_ver second finetune started!"
for target in $target_dataset
do
    for aux in ${datasets[@]/${target}}
    do
        # done 파일을 탐색하는 dir입니다
        dir=/disk/bubble3jh/DomainBed/pretrain-gnns/chem/runs/finetune_cls_runseed0/0215exp/target_${target}_aux_${aux}_${model_ver}_500
        if [ ! -d $dir ]; then #dir이 없으면 생성
            mkdir $dir
        fi
        if  [ ! -e $FILE ]; then
            # pwd -P
            # ls
            # echo $dataset
            # count=$(($count+1))
            python3 finetune.py --input_model_file /disk/bubble3jh/DomainBed/pretrain-gnns/chem/runs/finetune_cls_runseed0/freeze_bn/${aux}_500/best_model.pth --replace_classifier /disk/bubble3jh/DomainBed/pretrain-gnns/chem/runs/finetune_cls_runseed0/lp/${target}_lp_500/model.pth --dataset ${target} --filename target_${target}_aux_${aux}_${model_ver} --epochs 500 --freeze_bn
        fi
    done
done
python3 notify.py --msg "$target_dataset $model_ver second finetune completed!"
# echo $count