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

COUNTER=0
grep -P '^--ranks' ./logs/greedy_tn_trl_tensorized_full.$seed.log | sed -n "0~4p" | sed "s/--ranks '\(.*\)'/\1/g" | while read line
do
    python3 ResNet-TRL-Tensorized-Full.py --tensorization $tensorization --ranks "$line" --seed $seed --device $device | tee ./logs/resnet_trl_tensorized_full.$seed.$COUNTER.log
    COUNTER=$((COUNTER+1))
done