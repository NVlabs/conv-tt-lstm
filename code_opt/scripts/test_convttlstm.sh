#!/bin/bash

# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under a NVIDIA Open Source Non-commercial license

cd "$(dirname "$0")"
cd ..

python3 -m torch.distributed.launch --nproc_per_node=1 model_test.py --dataset KTH --use-sigmoid --img-channels 3 --img-height 120 --img-width 120 --kernel-size 5 --model convttlstm --model-order 3 --model-steps 3 --model-rank 8 --batch-size 1