#!/bin/bash

while getopts "s:g:d:f:e:t:c:p:n:i:S:" opt
do
    case "$opt" in
        s ) seed=$OPTARG ;;
        g ) gain=$OPTARG ;;
        d ) device=$OPTARG ;;
        f ) full_train=$OPTARG ;;
        e ) edge_train=$OPTARG ;;
        t ) tensorization=$OPTARG ;;
        c ) classifier=$OPTARG ;;
        p ) pretrained_filename="--pretrained_filename $OPTARG" ;;
        n ) network=$OPTARG ;;
        i ) iterations=$OPTARG ;;
        S ) start=$OPTARG ;;
    esac
done

COUNTER=0
grep -P '^--ranks' ./logs/greedy_tn.$seed.log | sed -n "0~4p" | sed "s/--ranks '\(.*\)'/\1/g" | while read line
do
    if ((COUNTER >= start)); then
        python3 ResNet-train.py --network $network --classifier $classifier --tensorization $tensorization --ranks "$line" --seed $seed --device $device | tee ./logs/resnet.$seed.$COUNTER.log
    fi
    COUNTER=$((COUNTER+1))
done