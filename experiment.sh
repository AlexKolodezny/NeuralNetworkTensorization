#!/bin/bash

while getopts "s:g:d:f:e:" opt
do
    case "$opt" in
        s ) seed=$OPTARG ;;
        g ) gain=$OPTARG ;;
        d ) device=$OPTARG ;;
        f ) full_train=$OPTARG ;;
        e ) edge_train=$OPTARG ;;
    esac
done

python3 ResNet-Greedy-TN-TRL-linked.py --seed $seed --device $device --full_train $full_train --edge_train $edge_train --slice_gain $gain | tee ./logs/greedy_tn_trl_ringed.$seed.log
COUNTER=0
grep -P 'core_ranks' ./logs/greedy_tn_trl_ringed.$seed.log | sed -n "0~4p" | while read line
do
    python3 ResNet-TRL-Ringed.py $line --seed $seed --device $device | tee ./logs/resnet_trl_ringed.$seed.$COUNTER.log
    COUNTER=$((COUNTER+1))
done | tee ./logs/experiment.$seed.log