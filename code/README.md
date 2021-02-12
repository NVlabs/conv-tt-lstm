# Convolutional Tensor-Train LSTM (Conv-TT-LSTM)

## Intro
PyTorch implementations of the paper, '***Convolutional Tensor-Train LSTM for Spatio-Temporal Learning***', NeurIPS 2020. [[project page](https://sites.google.com/nvidia.com/conv-tt-lstm)]

* Original implementation to reproduce the numbers in the paper.
* We also provide a highly optimized implementation for NVIDIA GPU under the folder ../code_opt/, check it out! 

## License 
Copyright (c) 2020 NVIDIA Corporation. All rights reserved. This work is licensed under a NVIDIA Open Source Non-commercial license.

## Requirements
- Python > 3.0
- Pytorch 1.0

## Install 
pip3 install -r requirements.txt

## Dataset
- MNIST
- KTH

## Training the model
1) Using our scripts.
```shell
bash code/scripts/train_convttlstm.sh
```

2) Creating a script on your own.
```shell
python3 model_train.py [options]

```
options
```shell
python3 model_train.py \ 
--dataset MNIST \ # MNIST or KTH
--batch-size 8 \ # batch size 
--use-sigmoid \ # if using sigmoid output: true for MNIST, false for other datasets
--img-height 64 \ # the image height of video frame: 64 for MNIST and 120 for KTH
--img-width 64 \ # the image width of video frame: 64 for MNIST and 120 for KTH
--kernel-size 5 \ # the kernel size of the convolutional operations 
--model convttlstm \ # 'convlstm' or 'convttlstm'
--model-order 3 \ # order of the convolutional tensor-train LSTMs
--model-steps 3 \ # steps of the convolutional tensor-train LSTMs
--model-rank 8 \ # tensor rank of the convolutional tensor-train LSTMs
--learning-rate 1e-4 \ # initial learning rate of the Adam optimizer
--gradient-clipping \ # use gradient clipping in training
--clipping-threshold 3 \ # threshold value for gradient clipping
```

## Add the module for Perceptual Similarity
1) In the code/ directory, 
    git clone https://github.com/richzhang/PerceptualSimilarity.git -b 1.0
    
2) Add an empty \_\_init\_\_.py file in code/PerceptualSimilarity/

3) For all files in code/PerceptualSimilarity/models, remove 'from Ipython import embed' 
    
4) For dist\_model.py and networks\_basic.py in code/PerceptualSimilarity/models/, change 'import models as util' to 'from PerceptualSimilarity import models as util'

5) In code/PerceptualSimilarity/models/\_\_init\_\_.py, change 'from models import dist\_model' to 'from PerceptualSimilarity.models import dist\_model'


## Testing the model
* In order to run the test code, you must have trained the model and saved at least one checkpoint as a '.pt'  file

1) Using our script: 'checkpoint.pt' is automatically loaded.
```shell
bash code/scripts/test_convttlstm.sh
```

2) Creating a script on your own.
```shell
python3 model_test.py [options]

```
options
```shell
python3 model_test.py \ 
--dataset MNIST \ # MNIST or KTH
--data-path \ # path to the dataset folder
--test-data-file \ # Name of the file for test set
--checkpoint \ # name for the checkpoint
--batch-size 8 \ # batch size 
--use-sigmoid \ # if using sigmoid output: false for MNIST, true for other datasets
--img-height 64 \ # the image height of video frame: 64 for MNIST and 120 for KTH
--img-width 64 \ # the image width of video frame: 64 for MNIST and 120 for KTH
--kernel-size 5 \ # the kernel size of the convolutional operations 
--model convttlstm \ # 'convlstm' or 'convttlstm'
--model-order 3 \ # order of the Convolutional Tensor-Train LSTMs
--model-steps 3 \ # steps of the Convolutional Tensor-Train LSTMs
--model-rank 8 \ # tensor rank of the Convolutional Tensor-Train LSTMs
--future-frames 20 \ # number of predicted frames
```

## Contacts
This code was written by [Wonmin Byeon](https://github.com/wonmin-byeon) \(wbyeon@nvidia.com\) and [Jiahao Su](https://github.com/jiahaosu) \(jiahaosu@terpmail.umd.edu\).
