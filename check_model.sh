#!/usr/bin/env bash

backbone="resnet50"

for ((FOLD=5; FOLD<6; FOLD++)); do
    echo "==========================================================="
    echo "Fold ${FOLD}"
    python utils/check_simple.py --backbone ${backbone} --fold ${FOLD}
done