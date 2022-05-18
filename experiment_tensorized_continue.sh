#!/bin/bash

while getopts "s:g:d:f:e:w:i:S:" opt
do
    case "$opt" in
        s ) seed=$OPTARG ;;
        g ) gain=$OPTARG ;;
        d ) device=$OPTARG ;;
        f ) full_train=$OPTARG ;;
        e ) edge_train=$OPTARG ;;
        w ) warmup=$OPTARG ;;
        i ) iterations=$OPTARG ;;
        S ) start=$OPTARG ;;
    esac
done

COUNTER=0
grep -P '^--ranks' ./logs/greedy_tn_conv.$seed.log | sed -n "0~4p" | while read line
do
    if ((COUNTER >= start)); then
        python3 ResNet-Tensorized.py $line --seed $seed --device $device | tee ./logs/resnet_tensorized.$seed.$COUNTER.log
    fi
    COUNTER=$((COUNTER+1))
done