#!/bin/bash

while getopts "s:g:d:f:e:t:" opt
do
    case "$opt" in
        s ) seed=$OPTARG ;;
        g ) gain=$OPTARG ;;
        d ) device=$OPTARG ;;
        f ) full_train=$OPTARG ;;
        e ) edge_train=$OPTARG ;;
        t ) tensorization=$OPTARG ;;
    esac
done

python3 ResNet-Greedy-TN-TRL-full-from-pretrained.py --filename resnet-full --tensorization $tensorization --seed $seed --device $device --full_train $full_train --edge_train $edge_train --slice_gain $gain | tee ./logs/greedy_tn_trl_full_from_pretrained.$seed.log
COUNTER=0
grep -P '^--ranks' ./logs/greedy_tn_trl_full_from_pretrained.$seed.log | sed -n "0~4p" | sed "s/--ranks '\(.*\)'/\1/g" | while read line
do
    python3 ResNet-TRL-Tensorized-Full.py --tensorization $tensorization --ranks "$line" --seed $seed --device $device | tee ./logs/resnet_trl_tensorized_full_from_pretrained.$seed.$COUNTER.log
    COUNTER=$((COUNTER+1))
done