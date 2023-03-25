#!/bin/bash
FILE="done"
count=0
epochs=500
for dataset in "tox21" "hiv" "muv" "bace" "bbbp" "toxcast" "sider" "clintox"
do
    # done 파일을 탐색하는 dir입니다
    dir=/disk/bubble3jh/DomainBed/pretrain-gnns/chem/runs/finetune_cls_runseed0/${dataset}_${epochs}
    if [ ! -d $dir ]; then #dir이 없으면 생성
        mkdir $dir
    fi
    cd $dir
    if  [ ! -e $FILE ]; then
        ##count sweep을 실행하지않고 실행되지 않은 tuning 후보가 얼마나 남았는지 count합니다
        # pwd -P
        # ls
        echo $dataset
        count=$(($count+1))
        ##sbatch sweep을 실행합니다. root dir로 돌아가서 sbatch를 위한 파일에 접근해서 hyp param에 해당하는 라인을 수정한 뒤에 이를 sbatch 시킵니다.
        # python3 finetune.py --input_model_file "/disk/bubble3jh/DomainBed/pretrain-gnns/chem/model_gin/contextpred.pth" --dataset ${dataset} --filename ${dataset} --epochs ${epochs}
    fi
done
echo $count