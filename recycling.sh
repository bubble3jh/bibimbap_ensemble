#!/bin/bash
FILE="done"
freeze_dir=""
epochs=300
datasets=("tox21" "hiv" "muv" "bace" "bbbp" "toxcast" "sider" "clintox")
target="muv"
seed=2
echo "target datset : $target "
# read target
echo "bn freeze : yes"
# read fbn
fbn="--freeze_bn"
python3 notify.py --msg "$target $fbn seed $seed recycling started!"
for model_ver in "fr_fr" "fr_nfr" 
do
    for freezing in "--freeze_lc" "--freeze_gnn"
    do
        if [ "$freezing" = "--freeze_lc" ]; then
            freeze_dir=_flc
        fi
        if [ "$freezing" = "--freeze_gnn" ]; then
            freeze_dir=_fgnn
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
            python3 finetune.py --averaging_target ${target} --averaging_aux ${datasets[@]/${target}} --dataset ${target} --epochs ${epochs} --runseed ${seed} --model_ver ${model_ver} ${freezing} ${fbn}
            
        fi
    done
    
done
python3 notify.py --msg "$target $fbn seed $seed recycling completed!"
