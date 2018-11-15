#!/usr/bin/env bash

backbone="resnet50"
bz=8

for ((FOLD=1; FOLD<6; FOLD++)); do
    python train.py --backbone ${backbone} --fold ${FOLD} --bz ${bz}
done