# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under a NVIDIA Open Source Non-commercial license.

# system modules
import os, argparse

# basic pytorch modules
import torch
import torch.nn as nn
import torch.nn.functional as F

# computer vision/image processing modules
import skimage

# math/probability modules
import random
import numpy as np

# custom utilities
from dataloader import MNIST_Dataset, KTH_Dataset 
from convlstmnet import ConvLSTMNet

# perceptive quality
import PerceptualSimilarity.models as PSmodels

def main(args):
    ## Model preparation (Conv-LSTM or Conv-TT-LSTM)

    # whether to use GPU (or CPU) 
    use_cuda  = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # whether to use multi-GPU (or single-GPU)
    multi_gpu = use_cuda and args.multi_gpu and torch.cuda.device_count() > 1
    num_gpus = (torch.cuda.device_count() if multi_gpu else 1) if use_cuda else 0

    # construct the model with the specified hyper-parameters
    model = ConvLSTMNet(
        # input to the model
        input_channels = args.img_channels, 
        # architecture of the model
        layers_per_block = (3, 3, 3, 3), 
        hidden_channels = (32, 48, 48, 32), 
        skip_stride = 2,
        # parameters of convolutional tensor-train layers
        cell = args.model, cell_params = {"order": args.model_order,
        "steps": args.model_steps, "rank": args.model_rank},
        # parameters of convolutional operations
        kernel_size = args.kernel_size, bias = True,
        # output function and output format
        output_sigmoid = args.use_sigmoid)

    # move the model to the device (CPU, GPU, multi-GPU) 
    model.to(device)
    if multi_gpu: 
        model = nn.DataParallel(model)

    # load the model parameters from checkpoint
    model.load_state_dict(torch.load(args.checkpoint))


    ## Dataset Preparation (Moving-MNIST, KTH)
    Dataset = {"MNIST": MNIST_Dataset, "KTH": KTH_Dataset}[args.dataset]

    DATA_DIR = os.path.join("../../datasets", 
        {"MNIST": "moving-mnist", "KTH": "kth"}[args.dataset])

    # number of total frames
    total_frames = args.input_frames + args.future_frames

    # dataloaer for test set
    test_data_path = os.path.join(DATA_DIR, args.test_data_file)

    test_data = Dataset({"path": test_data_path, "unique_mode": True,
        "num_frames": total_frames, "num_samples": args.test_samples,
        "height": args.img_height, "width": args.img_width, "channels": args.img_channels})

    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size = args.batch_size, 
        shuffle = False, num_workers = 5 * max(num_gpus, 1), drop_last = True)

    test_size = len(test_data_loader) * args.batch_size


    ## Main script for test phase
    model.eval()
    
    MSE  = np.zeros(args.future_frames, dtype = np.float32)
    PSNR = np.zeros(args.future_frames, dtype = np.float32)
    SSIM = np.zeros(args.future_frames, dtype = np.float32)
    PIPS = np.zeros(args.future_frames, dtype = np.float32)

    PSmodel = PSmodels.PerceptualLoss(model = 'net-lin', 
        net = 'alex', use_gpu = use_cuda, gpu_ids = [0])

    with torch.no_grad():
        
        for frames in test_data_loader:

            # 5-th order: batch_size x total_frames x channels x height x width 
            frames = frames.permute(0, 1, 4, 2, 3).to(device)

            inputs = frames[:,  :args.input_frames]
            origin = frames[:, -args.future_frames:]

            pred = model(inputs, 
                input_frames  =  args.input_frames, 
                future_frames = args.future_frames, 
                output_frames = args.future_frames, 
                teacher_forcing = False)

            # clamp the output to [0, 1]
            pred = torch.clamp(pred, min = 0, max = 1)

            # accumlate the statistics per frame
            for t in range(-args.future_frames, 0):
                origin_, pred_ = origin[:, t], pred[:, t]
                if args.img_channels == 1:
                    origin_ = origin_.repeat([1, 3, 1, 1])
                    pred_   =   pred_.repeat([1, 3, 1, 1])

                dist = PSmodel(origin_, pred_)
                PIPS[t] += torch.sum(dist).item() / test_size

            origin = origin.permute(0, 1, 3, 4, 2).cpu().numpy()
            pred   =   pred.permute(0, 1, 3, 4, 2).cpu().numpy()
            for t in range(-args.future_frames, 0):
                for i in range(args.batch_size):
                    origin_, pred_ = origin[i, t], pred[i, t]
                    if args.img_channels == 1:
                        origin_ = np.squeeze(origin_, axis = -1)
                        pred_   = np.squeeze(pred_,   axis = -1)

                    MSE[t]  += skimage.measure.compare_mse( origin_, pred_) / test_size
                    PSNR[t] += skimage.measure.compare_psnr(origin_, pred_) / test_size
                    SSIM[t] += skimage.measure.compare_ssim(origin_, pred_, 
                                    multichannel = (args.img_channels > 1)) / test_size


    print("MSE: {} (x1e-3); PSNR: {}, SSIM: {}, LPIPS: {}".format(
        1e3 * np.mean(MSE), np.mean(PSNR), np.mean(SSIM), np.mean(PIPS)))

    print("MSE:",  MSE)
    print("PSNR:", PSNR)
    print("SSIM:", SSIM)
    print("PIPS:", PIPS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Conv-TT-LSTM Test")

    ## Data format (batch_size x time_steps x height x width x channels)

    # batch size and the logging period 
    parser.add_argument('--batch-size',  default = 16, type = int,
        help = 'The batch size in training phase.')

    # frame split
    parser.add_argument('--input-frames',  default = 10, type = int,
        help = 'The number of input frames to the model.')
    parser.add_argument('--future-frames', default = 10, type = int,
        help = 'The number of predicted frames of the model.')
    
    # frame format
    parser.add_argument('--img-height',  default = 64, type = int, 
        help = 'The image height of each video frame.')
    parser.add_argument('--img-width',   default = 64, type = int, 
        help = 'The image width of each video frame.')
    parser.add_argument('--img-channels', default = 1, type = int, 
        help = 'The number of channels in each video frame.')

    ## Devices (CPU, single-GPU or multi-GPU)

    # whether to use GPU for testing
    parser.add_argument('--use-cuda', dest = 'use_cuda', action = 'store_true',
        help = 'Use GPU for testing.')
    parser.add_argument('--no-cuda',  dest = 'use_cuda', action = 'store_false', 
        help = "Use CPU for testing.")
    parser.set_defaults(use_cuda = True)

    # whether to use multi-GPU for testing (given GPU is used)
    parser.add_argument('--multi-gpu',  dest = 'multi_gpu', action = 'store_true',
        help = 'Use multiple GPUs for testing.')
    parser.add_argument('--single-gpu', dest = 'multi_gpu', action = 'store_false',
        help = 'Use single GPU for testing.')
    parser.set_defaults(multi_gpu = True)

    ## Models (Conv-LSTM or Conv-TT-LSTM)

    # model name (with time stamp as suffix)
    parser.add_argument('--checkpoint', default = "checkpoint.pt", type = str,
        help = 'The name for the checkpoint.')

    # model type and size (depth and width) 
    parser.add_argument('--model', default = 'convlstm', type = str,
        help = 'The model is either \"convlstm\"" or \"convttlstm\".')

    parser.add_argument('--use-sigmoid', dest = 'use_sigmoid', action = 'store_true',
        help = 'Use sigmoid function at the output of the model.')
    parser.add_argument('--no-sigmoid',  dest = 'use_sigmoid', action = 'store_false',
        help = 'Use output from the last layer as the final output.')
    parser.set_defaults(use_sigmoid = False)

    # parameters of the convolutional tensor-train layers
    parser.add_argument('--model-order', default = 3, type = int, 
        help = 'The order of the convolutional tensor-train LSTMs.')
    parser.add_argument('--model-steps', default = 3, type = int, 
        help = 'The steps of the convolutional tensor-train LSTMs')
    parser.add_argument('--model-rank',  default = 8, type = int, 
        help = 'The tensor rank of the convolutional tensor-train LSTMs.')
    
    # parameters of the convolutional operations
    parser.add_argument('--kernel-size', default = 3, type = int, 
        help = "The kernel size of the convolutional operations.")

    ## Dataset (Input)
    parser.add_argument('--dataset', default = "MNIST", type = str, 
        help = 'The dataset name. (Options: MNIST, KTH, KITTI)')
    parser.add_argument('--data-path', default = 'default', type = str,
        help = 'The path to the dataset folder.')

    parser.add_argument('--test-data-file', default = 'moving-mnist-test-f40-new1.npz', type = str, 
        help = 'Name of the folder/file for test set.')
    parser.add_argument('--test-samples', default = 5000, type = int, 
        help = 'Number of unique samples in test set.')

    main(parser.parse_args())
