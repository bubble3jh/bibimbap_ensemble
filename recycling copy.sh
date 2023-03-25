#!/bin/bash
FILE="done"
freeze_dir=""
count=0
epochs=300
datasets=("tox21" "hiv" "muv" "bace" "bbbp" "toxcast" "sider" "clintox")
echo "target datset : "
# read target
echo "bn freeze : "
# read fbn
target="sider"
fbn=""
python3 notify.py --msg "$target $fbn recycling started!"
for model_ver in "fr_fr" "fr_nfr" "nfr_fr" "nfr_nfr"
do
    
    for freezing in "--freeze_lc --freeze_gnn" "--freeze_lc" "--freeze_gnn" ""
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
        dir=/disk/bubble3jh/DomainBed/pretrain-gnns/chem/runs/finetune_cls_runseed${seed}/0215exp/recycling_${target}_${model_ver}${freeze_dir}_${epochs}
        echo $dir
        if [ ! -d $dir ]; then #dir이 없으면 생성
            mkdir $dir
        fi
        if  [ ! -e $FILE ]; then
            # pwd -P
            # ls
            # echo $dataset
            # count=$(($count+1))
            python3 finetune.py --averaging_target ${target} --averaging_aux ${datasets[@]/${target}} --dataset ${target} --epochs ${epochs} --model_ver ${model_ver} ${freezing} ${fbn}
            
        fi
    done
    
done
python3 notify.py --msg "$target $fbn recycling completed!"
# echo $count