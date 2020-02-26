# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under a NVIDIA Open Source Non-commercial license

import os, glob
import numpy as np
import skimage
import random

from torch.utils.data.dataset import Dataset


# Dataloder for all Moving-MNIST datasets (binary and colored)
class MNIST_Dataset(Dataset):

    def __init__(self, params):
        # parameters of the dataset 
        path = params['path']
        assert os.path.exists(path), "The file does not exist."

        self.num_frames  = params['num_frames']
        self.num_samples = params.get('num_samples', None)

        self.random_crop = params.get('random_crop', False) 

        self.img_height   = params.get('height',  64)
        self.img_width    = params.get('width',   64)
        self.img_channels = params.get('channels', 3)

        self.data = np.float32(np.load(path)["data"] / 255.0)
        self.data_samples = self.data.shape[0]
        self.data_frames  = self.data.shape[1]

    def __getitem__(self, index):
        start = random.randint(0, self.data_frames - 
            self.num_frames) if self.random_crop else 0

        data  = self.data[index, start : start + self.num_frames]

        return data 

    def __len__(self):
        return len(self.data_samples) if self.num_samples is None \
            else min(self.data_samples,  self.num_samples)


# Dataloader for KTH dataset (TODO: test)
class KTH_Dataset(Dataset):

    def __init__(self, params):
        # parameters of the dataset
        path = params['path']
        assert os.path.exists(path), "The dataset folder does not exist."

        unique_mode = params.get('unique_mode', True)

        self.num_samples = params.get('num_samples', None)
        self.num_frames  = params['num_frames']

        self.img_height   = params.get('height', 120)
        self.img_width    = params.get('width',  120)
        self.img_channels = params.get('channels', 3)
        
        # parse the files in the data folder
        self.files = glob.glob(os.path.join(path, '*.npz*'))
        self.clips = []
        for i in range(len(self.files)):
            data = np.load(self.files[i])["data"]
            data_frames = data.shape[0] 

            self.clips += [(i, t) for t in range(data_frames - self.num_frames)] if not unique_mode \
                else [(i, t * self.num_frames) for t in range(data_frames // self.num_frames)]

        self.data_samples = len(self.clips)

    def __getitem__(self, index):
        (file_index, start_frame) = self.clips[index]

        # 4th order: num_frames(0) x img_height(1) x img_width(2) x img_channel(3)
        data = np.load(self.files[file_index])["data"]

        # place holder for data processing
        _, img_height, img_width, _ = data.shape
        if img_height == self.img_height and img_width == self.img_width:
            clip = data[start_frame : start_frame + self.num_frames]
        else: # resizing the input is needed
            clip = np.stack([resize(data[t], (self.img_height, self.img_width)) 
                for t in range(self.num_frames)], axis = 0)

        data = np.float32(clip)
        if self.img_channels == 1:
            data = np.mean(data, axis = -1, keepdims = True)

        return data

    def __len__(self):
        return self.data_samples if self.num_samples is None \
            else min(self.data_samples, self.num_samples)
