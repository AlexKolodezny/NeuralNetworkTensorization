#!/bin/bash

while getopts "s:g:d:f:e:w:i:" opt
do
    case "$opt" in
        s ) seed=$OPTARG ;;
        g ) gain=$OPTARG ;;
        d ) device=$OPTARG ;;
        f ) full_train=$OPTARG ;;
        e ) edge_train=$OPTARG ;;
        w ) warmup=$OPTARG ;;
        i ) iterations=$OPTARG ;;
    esac
done

python3 ResNet-Greedy-TN-Conv.py --seed $seed --device $device --full_train $full_train --edge_train $edge_train --slice_gain $gain --warmup $warmup --iterations=$iterations | tee ./logs/greedy_tn_conv.$seed.log
COUNTER=0
grep -P '^--ranks' ./logs/greedy_tn_conv.$seed.log | sed -n "0~4p" | while read line
do
    python3 ResNet-Tensorized.py $line --seed $seed --device $device | tee ./logs/resnet_tensorized.$seed.$COUNTER.log
    COUNTER=$((COUNTER+1))
done