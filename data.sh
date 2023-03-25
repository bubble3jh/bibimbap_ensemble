#!/bin/bash
FILE="done"
freeze_dir=""
epochs=300
datasets=("tox21" "hiv" "muv" "bace" "bbbp" "toxcast" "sider" "clintox")

target="muv"
fbn="--freeze_bn"
seed=8
for model_ver in "fr_fr" "fr_nfr" 
do
    for freezing in "--freeze_lc" "--freeze_gnn"
    do
        # done 파일을 탐색하는 dir입니다
        if [ "$freezing" = "--freeze_lc --freeze_gnn" ]; then
            freeze_dir="_flc_fgnn"
        fi
        if [ "$freezing" = "--freeze_lc" ]; then
            freeze_dir=_flc
        fi
        if [ "$freezing" = "--freeze_gnn" ]; then
            freeze_dir=_fgnn
        fi
        if [ "$freezing" = "" ]; then
            freeze_dir=""
        fi
        if [ "$fbn" = "--freeze_bn" ]; then
            freeze_dir+="_fbn"
        fi
        echo $target $model_ver$freeze_dir
        dir=/disk/bubble3jh/DomainBed/pretrain-gnns/chem/runs/finetune_cls_runseed${seed}/0215exp/recycling_${target}_${model_ver}${freeze_dir}_${epochs}

        value=`cat $dir/done`
        echo "$value"
        
    done
    
done

# echo $count