# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under a NVIDIA Open Source Non-commercial license

# system modules
import os, argparse

# basic pytorch modules
import torch
import torch.nn as nn
import torch.nn.functional as F

# computer vision/image processing modules 
import torchvision
import skimage.metrics

# math/probability modules
import random
import numpy as np

# custom utilities
from convlstmnet import ConvLSTMNet 
from dataloader import KTH_Dataset, MNIST_Dataset

from torch.nn.parallel import DistributedDataParallel as DDP 
from gpu_affinity import set_affinity

# perceptive quality
import PerceptualSimilarity.models as PSmodels


def main(args):
    ## Distributed computing

    # utility for synchronization
    def reduce_tensor(tensor, reduce_sum = False):
        rt = tensor.clone()
        torch.distributed.all_reduce(rt, op = torch.distributed.ReduceOp.SUM)
        return rt if reduce_sum else (rt / world_size)

    # enable distributed computing
    if args.distributed:
        set_affinity(args.local_rank)
        num_devices = torch.cuda.device_count()
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend = 'nccl', init_method = 'env://')
        node_rank = args.node_rank
        global_rank = node_rank * num_devices + args.local_rank
        world_size  = torch.distributed.get_world_size() #os.environ['WORLD_SIZE']
    else:
        global_rank, num_devices, world_size = 0, 1, 1
        

    ## Data format: batch(0) x steps(1) x height(2) x width(3) x channels(4) 

    # batch_size (0)
    total_batch_size  = args.batch_size
    assert total_batch_size % world_size == 0, \
        'The batch_size is not divisible by world_size.'
    batch_size = total_batch_size // world_size

    # steps (1)
    total_frames = args.future_frames + args.input_frames

    # frame format (2, 3)
    img_resize = (args.img_height != args.img_height_u) or (args.img_width != args.img_width_u)


    ## Model preparation (Conv-LSTM or Conv-TT-LSTM)

    # size of the neural network model (depth and width)
    layers_per_block = (3, 3, 3, 3)
    hidden_channels  = (32, 48, 48, 32)
    skip_stride = 2

    # construct the model with the specified hyper-parameters
    model = ConvLSTMNet(
        # architecture of the model
        layers_per_block = layers_per_block, hidden_channels = hidden_channels, 
        input_channels = 1, skip_stride = skip_stride,
        cell_params = {"steps": 3, "order": 3, "ranks": 8},
        # parameters of convolutional operation
        kernel_size = 5, bias = True).cuda()

    if args.distributed:
        model = DDP(model, device_ids = [args.local_rank])

    PSmodel = PSmodels.PerceptualLoss(model = 'net-lin', 
        net = 'alex', use_gpu = True, gpu_ids = [args.local_rank])

    ## Dataset Preparation (KTH, UCF, tinyUCF)
    assert args.dataset in ["MNIST", "KTH"], \
        "The dataset is not currently supported."

    Dataset = {"KTH": KTH_Dataset, "MNIST": MNIST_Dataset}[args.dataset]
               
    # path to the dataset folder
    DATA_DIR = args.data_path

    assert os.path.exists(DATA_DIR), \
        "The dataset folder does not exist. "+DATA_DIR

    assert os.path.exists(DATA_DIR), \
        "The test dataset does not exist. "+DATA_DIR

    test_dataset = Dataset({"path": DATA_DIR, 
        "unique_mode": True, "num_frames": total_frames, "num_samples": args.test_samples,
        "height": args.img_height, "width": args.img_width, "channels": 1, 'training': False})

    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset, num_replicas = world_size, rank = global_rank, shuffle = False)
    test_loader  = torch.utils.data.DataLoader(
        test_dataset, batch_size = batch_size, drop_last = True, 
        num_workers = num_devices * 4, pin_memory = True, sampler = test_sampler)

    test_samples = len(test_loader) * total_batch_size

    MODEL_FILE = args.model_path

    assert os.path.exists(MODEL_FILE), \
        "The specified model is not found in the folder."

    checkpoint = torch.load(MODEL_FILE)
    eval_epoch = checkpoint.get("epoch", 0)
    model.load_state_dict(checkpoint["model_state_dict"])

    ## Main script for test phase 
    MSE_  = torch.zeros((args.future_frames), dtype = torch.float32).cuda()
    PSNR_ = torch.zeros((args.future_frames), dtype = torch.float32).cuda()
    SSIM_ = torch.zeros((args.future_frames), dtype = torch.float32).cuda()
    PIPS_ = torch.zeros((args.future_frames), dtype = torch.float32).cuda()

    with torch.no_grad():
        model.eval()
        
        samples = 0
        for it, frames in enumerate(test_loader):
            samples += total_batch_size

            frames = torch.mean(frames, dim = -1, keepdim = True)

            if img_resize:
                frames_ = frames.cpu().numpy()
                frames = np.zeros((batch_size, total_frames, 
                    args.img_height_u, args.img_width_u, 1), dtype = np.float32)

                for b in range(batch_size):
                    for t in range(total_frames):
                        frames[b, t] = skimage.transform.resize(
                            frames_[b, t], (args.img_height_u, args.img_width_u))

                frames = torch.from_numpy(frames)

            # 5-th order: batch_size x total_frames x channels x height x width 
            frames = frames.permute(0, 1, 4, 2, 3).cuda()
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

                origin_ = origin_.repeat([1, 3, 1, 1])
                pred_   =   pred_.repeat([1, 3, 1, 1])

                dist = PSmodel(origin_, pred_)
                PIPS_[t] += torch.sum(dist).item()

            origin = origin.permute(0, 1, 3, 4, 2).cpu().numpy()
            pred   =   pred.permute(0, 1, 3, 4, 2).cpu().numpy()

            for t in range(-args.future_frames, 0):
                for i in range(batch_size):
                    origin_, pred_ = origin[i, t], pred[i, t]

                    origin_ = np.squeeze(origin_, axis = -1)
                    pred_   = np.squeeze(pred_,   axis = -1)

                    MSE_[t]  += skimage.metrics.mean_squared_error(origin_, pred_)
                    PSNR_[t] += skimage.metrics.peak_signal_noise_ratio(origin_, pred_)
                    SSIM_[t] += skimage.metrics.structural_similarity(origin_, pred_)

            if args.distributed:
                MSE  = reduce_tensor(MSE_,  reduce_sum = True) / samples
                PSNR = reduce_tensor(PSNR_, reduce_sum = True) / samples
                SSIM = reduce_tensor(SSIM_, reduce_sum = True) / samples
                PIPS = reduce_tensor(PIPS_, reduce_sum = True) / samples
            else:
                MSE  = MSE_  / samples
                PSNR = PSNR_ / samples
                SSIM = SSIM_ / samples
                PIPS = PIPS_ / samples

            if ((it + 1) % 50 == 0 or it + 1 == len(test_loader)) and args.local_rank == 0:
                print((it + 1) * total_batch_size, '/', test_samples,
                    ": MSE:  ", torch.mean(MSE ).cpu().item() * 1e3,
                    "; PSNR: ", torch.mean(PSNR).cpu().item(), 
                    "; SSIM: ", torch.mean(SSIM).cpu().item(), 
                    ";LPIPS: ", torch.mean(PIPS).cpu().item())

        if args.distributed:
            MSE  = reduce_tensor(MSE_,  reduce_sum = True) / test_samples
            PSNR = reduce_tensor(PSNR_, reduce_sum = True) / test_samples
            SSIM = reduce_tensor(SSIM_, reduce_sum = True) / test_samples
            PIPS = reduce_tensor(PIPS_, reduce_sum = True) / test_samples
        else:
            MSE  = MSE_  / test_samples
            PSNR = PSNR_ / test_samples
            SSIM = SSIM_ / test_samples
            PIPS = PIPS_ / test_samples

        MSE_AVG  = torch.mean(MSE ).cpu().item()
        PSNR_AVG = torch.mean(PSNR).cpu().item()
        SSIM_AVG = torch.mean(SSIM).cpu().item()
        PIPS_AVG = torch.mean(PIPS).cpu().item()

        if args.local_rank == 0:
            print("Epoch \t{} \tMSE: \t{} (x1e-3) \tPSNR: \t{} \tSSIM: \t{} \tLPIPS: \t{}".format(
                eval_epoch, 1e3 * MSE_AVG, PSNR_AVG, SSIM_AVG, PIPS_AVG))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Conv-TT-LSTM Test")

    # whether to use distributed computing
    parser.add_argument('--use-distributed', dest = "distributed", 
        action = 'store_true',  help = 'Use distributed computing in training.')
    parser.add_argument( '--no-distributed', dest = "distributed", 
        action = 'store_false', help = 'Use single process (GPU) in training.')
    parser.set_defaults(distributed = True)

    # arguments for distributed computing 
    parser.add_argument('--local_rank', default = 0, type = int)
    parser.add_argument( '--node_rank', default = 0, type = int)

    ## Data format (batch_size x time_steps x height x width x channels)
    # batch size (0) 
    parser.add_argument('--batch-size', default = 16, type = int,
        help = 'The total batch size in each test iteration.')
    parser.add_argument('--log-iterations', default = 10, type = int,
        help = 'Log the test video every log_iterations.')

    # frame split (1)
    parser.add_argument('--input-frames',  default = 10, type = int,
        help = 'The number of input frames to the model.')
    parser.add_argument('--future-frames', default = 10, type = int,
        help = 'The number of predicted frames of the model.')
    
    # frame format (2, 3, 4)
    parser.add_argument('--img-channels', default =  1, type = int, 
        help = 'The number of channels in each video frame.')

    parser.add_argument('--img-height',   default = 64, type = int, 
        help = 'The image height of each video frame.')
    parser.add_argument('--img-width',    default = 64, type = int, 
        help = 'The image width  of each video frame.')

    parser.add_argument('--img-height-u', default = 64, type = int, 
        help = 'The image height of each upsampled frame.')
    parser.add_argument('--img-width-u',  default = 64, type = int, 
        help = 'The image width  of each upsampled frame.')

    ## Models (Conv-LSTM or Conv-TT-LSTM)

    # model name (with time stamp as suffix)
    parser.add_argument('--model-path',  default = "models/test_0000.pt", type = str,
        help = 'The model name is used to create the folder names.')

    ## Dataset (Input)
    parser.add_argument('--dataset', default = "KTH", type = str, 
        help = 'The dataset name. (Options: KTH, MNIST)')
    parser.add_argument('--data-path', default = 'default', type = str,
        help = 'The path to the dataset folder.')

    parser.add_argument('--test-samples', default = 5000, type = int, 
        help = 'Number of samples in test dataset.')

    main(parser.parse_args())
