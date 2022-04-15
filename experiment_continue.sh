#!/bin/bash

while getopts "s:d:" opt
do
    case "$opt" in
        s ) seed=$OPTARG ;;
        d ) device=$OPTARG ;;
    esac
done

COUNTER=0
grep -P 'core_ranks' ./logs/greedy_tn_trl_ringed.$seed.log | sed -n "0~4p" | while read line
do
    python3 ResNet-TRL-Ringed.py $line --seed $seed --device $device | tee ./logs/resnet_trl_ringed.$seed.$COUNTER.log
    COUNTER=$((COUNTER+1))
done | tee ./logs/experiment.$seed.log