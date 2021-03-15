# Convolutional Tensor-Train LSTM (Conv-TT-LSTM)

## Intro
PyTorch implementations of the paper, '***Convolutional Tensor-Train LSTM for Spatio-Temporal Learning***', NeurIPS 2020. [[project page](https://sites.google.com/nvidia.com/conv-tt-lstm)]

* This implemntation is highly optimized for NVIDIA GPUs. 
* The details of optimization tricks are presented at ECCV 2020 tutorial, 'Mixed Precision Training for Convolutional Tensor-Train LSTM' [[slides]](https://nvlabs.github.io/eccv2020-mixed-precision-tutorial/files/wonmin_byeon-mixed-precision-training-for-convolutional-tensor-train-lstm.pdf) [[video]](https://www.youtube.com/watch?v=1XuD-ozHTLY&feature=youtu.be)

## License 
Copyright (c) 2020 NVIDIA Corporation. All rights reserved. This work is licensed under a NVIDIA Open Source Non-commercial license.

## Requirements
- Python > 3.6
- Pytorch > 1.2

## Install with Conda

#### Configure the Conda 

1) Create a new conda environment
```
conda create -n convttlstm python=3.6
```

2) Install all dependencies using pip
```shell
pip install -U pip setuptools
pip install -r requirements.txt

```

#### Install apex
Following the instructions in https://github.com/NVIDIA/apex
```shell
git clone https://github.com/NVIDIA/apex
cd apex

```

If your CUDA version is greater than 10.1, install apex with CUDA and C++ extensions.
```shell
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

If your CUDA version is lower than 10.1, install apex without CUDA and C++ extensions.
```shell
pip install -v --no-cache-dir ./

```
or alternatively
```shell
python setup.py install

```
In this case, the fused kernels are not supported.


## Install with Docker
1) Install Docker on your machine

2) Download the image
```shell
docker pull nvcr.io/nvidia/pytorch:20.06-py3

```
3) Run the container image in interactive mode
```shell
docker run -it nvcr.io/nvidia/pytorch:20.06-py3
```
4) Install other dependencies using pip
```shell
pip scikit-image==0.17.2 tensorboardX 
```
The correct version of apex is already installed in the downloaded image. The fused kernels are supported.

## Add the module for Perceptual Similarity 
1) In the code/ directory, 
    git clone https://github.com/richzhang/PerceptualSimilarity.git -b 1.0
    
2) Add an empty \_\_init\_\_.py file in code/PerceptualSimilarity/

3) For all files in code/PerceptualSimilarity/models, 
    remove 'from Ipython import embed' 
    
4) For dist\_model.py and networks\_basic.py in code/PerceptualSimilarity/models/, 
    change 'import models as util' to 'from PerceptualSimilarity import models as util'
    
5) In code/PerceptualSimilarity/models/\_\_init\_\_.py, 
    change 'from models import dist\_model' to 'from PerceptualSimilarity.models import dist\_model'


## Dataset
- MNIST
- KTH

## Training the model
1) Using our scripts.
```shell
bash code_opt/scripts/train_convttlstm.sh
```

2) Creating a script on your own.
```shell
python3 -m torch.distributed.launch --nproc_per_node=[num_gpus] model_train.py [options]

```
options
```shell
python3 -m torch.distributed.launch \
--nproc_per_node=8 model_train.py \ # number of GPUs 
--dataset KTH \ # MNIST or KTH
--batch-size 8 \ # batch size 
--use-sigmoid \ # if using sigmoid output: false for MNIST, true for other datasets
--img-height 120 \ # the image height of video frame: 64 for MNIST and 120 for KTH
--img-width 120 \ # the image width of video frame: 64 for MNIST and 120 for KTH
--img-channals 3 \ # the image channel for video frame: 1 for MNIST and 3 for KTH
--kernel-size 5 \ # the kernel size of the convolutional operations 
--model convttlstm \ # 'convlstm' or 'convttlstm'
--model-order 3 \ # order of the convolutional tensor-train LSTMs
--model-steps 3 \ # steps of the convolutional tensor-train LSTMs
--model-rank 8 \ # tensor rank of the convolutional tensor-train LSTMs
--learning-rate 1e-4 \ # initial learning rate of the Adam optimizer
--gradient-clipping \ # use gradient clipping in training
--clipping-threshold 1 \ # threshold value for gradient clipping
--use-amp \ # (optional) use automatic mixed precision
--use-fused \ # (optional) use fused Adam optimizer
--use-checkpointing \ # (optional) use checkpointing to reduce memory
```
If apex is installed without CUDA and C++ extensions, remove the "--use-fused" flag.

## Testing the model
* In order to run the test code, you must have trained the model and saved at least one checkpoint as a '.pt'  file

1) Using our script: 'checkpoint.pt' is automatically loaded. 
```shell
bash code_opt/scripts/test_convttlstm.sh
```

2) Creating a script on your own.
```shell
python3 -m torch.distributed.launch --nproc_per_node=[num_gpus] model_test.py [options]

```
options
```shell
python3 -m torch.distributed.launch \
--nproc_per_node=8 model_test.py \ # number of GPUs 
--dataset KTH \ # MNIST or KTH
--data-path \ # path to the dataset folder
--test-data-file \ # Name of the file for test set
--checkpoint checkpoint.pt \ # name for the checkpoint
--batch-size 8 \ # batch size 
--use-sigmoid \ # if using sigmoid output: false for MNIST, true for other datasets
--img-height 120 \ # the image height of video frame: 64 for MNIST and 120 for KTH
--img-width 120 \ # the image width of video frame: 64 for MNIST and 120 for KTH
--img-channals 3 \ # the image channel for video frame: 1 for MNIST and 3 for KTH
--kernel-size 5 \ # the kernel size of the convolutional operations 
--model convttlstm \ # 'convlstm' or 'convttlstm'
--model-order 3 \ # order of the Convolutional Tensor-Train LSTMs
--model-steps 3 \ # steps of the Convolutional Tensor-Train LSTMs
--model-rank 8 \ # tensor rank of the Convolutional Tensor-Train LSTMs
--future-frames 20 \ # number of predicted frames
```
## Contacts
This code was written by [Wonmin Byeon](https://github.com/wonmin-byeon) \(wbyeon@nvidia.com\) and [Jiahao Su](https://github.com/jiahaosu) \(jiahaosu@terpmail.umd.edu\).
