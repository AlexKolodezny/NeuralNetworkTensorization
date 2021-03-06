#!/bin/bash

while getopts "s:g:d:f:e:t:i:" opt
do
    case "$opt" in
        s ) seed=$OPTARG ;;
        d ) device=$OPTARG ;;
        t ) tensorization=$OPTARG ;;
        i ) iterations=$OPTARG ;;
    esac
done

python3 ResNet-Random-TRL-tensorized-full.py --tensorization $tensorization --seed $seed --device $device --iterations $iterations | tee ./logs/random_trl_tensorized_full.$seed.log
COUNTER=0
grep -P '^--ranks' ./logs/random_trl_tensorized_full.$seed.log | sed -n "0~4p" | sed "s/--ranks '\(.*\)'/\1/g" | while read line
do
    python3 ResNet-train.py --classifier full --tensorization $tensorization --ranks "$line" --seed $seed --device $device | tee ./logs/resnet_random_trl_tensorized_full.$seed.$COUNTER.log
    COUNTER=$((COUNTER+1))
done