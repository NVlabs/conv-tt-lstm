# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under a NVIDIA Open Source Non-commercial license

PATH=/data

python3 -m torch.distributed.launch --nproc_per_node=1 test.py --dataset MNIST --future-frames 30 --batch-size 8 --model-path models/convttlstm-mnist.pt --data-path ${PATH}/moving-mnist/moving-mnist-test.npz

python3 -m torch.distributed.launch --nproc_per_node=1 test.py --dataset KTH --img-height 120 --img-width 120 --img-height-u 128 --img-width-u 128 --future-frames 40 --batch-size 8 --model-path models/convttlstm-kth.pt --data-path ${PATH}/kth/test
