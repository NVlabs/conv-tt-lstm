#!/bin/bash

# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under a NVIDIA Open Source Non-commercial license

cd "$(dirname "$0")"
cd ..

# Pytorch standard implementation
python3 model_train.py --dataset KTH --use-sigmoid --img-channels 3 --img-height 120 --img-width 120 --kernel-size 5 --model convlstm --batch-size 8 --learning-rate 1e-3 --valid-samples 448 --num-epochs 500 --ssr-decay-ratio 4e-3
