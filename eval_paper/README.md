# Convolutional Tensor-Train LSTM (Conv-TT-LSTM)

## Intro
PyTorch implementations of the paper, '***Convolutional Tensor-Train LSTM for Spatio-Temporal Learning***', NeurIPS 2020. [[project page](https://sites.google.com/nvidia.com/conv-tt-lstm)]

* This code is for reproducing the results of [the paper (Table 4)](https://arxiv.org/pdf/2002.09131.pdf)

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

## Add the module for Perceptual Similary 
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

## Testing the model
1) Download and save the weight files in 'models/'
MNIST: [convttlstm-mnist.pt](https://drive.google.com/file/d/1MnK1ftUJgB0H4QOS-k4CVkwtBUjHtSO-/view?usp=sharing)
KTH: [convttlstm-kth.pt](https://drive.google.com/file/d/1gVicaRn6gJqIR-r89NxnIzVXAaIRomLc/view?usp=sharing)

2) Use the script 'test.sh' to evaluate the models. 
    Change the 'PATH' to your data path. 

```shell
python3 -m torch.distributed.launch --nproc_per_node=[num_gpus] test.py [options]

```
options
```shell
python3 -m torch.distributed.launch \
--nproc_per_node=1 model_test.py \ # number of GPUs 
--dataset MNIST \ # MNIST or KTH
--data-path \ # path to the dataset folder
--model-path models/convttlstm-mnist.pt \ # name for the checkpoint (MNIST: convttlstm-mnist.pt, KTH: convttlstm-kth.pt)
--batch-size 8 \ # batch size 
--future-frames 30 \ # number of predicted frames
--img-height 64 \ # the image height of video frame: 64 for MNIST and 120 for KTH
--img-width 64 \ # the image width of video frame: 64 for MNIST and 120 for KTH
--img-height-u 64 \ # the image height of upsampled frame: 64 for MNIST and 128 for KTH
--img-width-u 64 \ # the image width of upsampled frame: 64 for MNIST and 128 for KTH
```

## Contacts
This code was written by [Wonmin Byeon](https://github.com/wonmin-byeon) \(wbyeon@nvidia.com\) and [Jiahao Su](https://github.com/jiahaosu) \(jiahaosu@terpmail.umd.edu\).
