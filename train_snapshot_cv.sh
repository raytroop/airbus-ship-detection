#!/usr/bin/env bash

backbone="resnet50"
bz=8
nb_epochs=40
nb_snapshots=2
init_lr=0.003

for ((FOLD=1; FOLD<6; FOLD++)); do
    python train_snapshot.py --backbone ${backbone} --fold ${FOLD} --bz ${bz} --nb_epochs ${nb_epochs} --nb_snapshots ${nb_snapshots} --init_lr ${init_lr}
done