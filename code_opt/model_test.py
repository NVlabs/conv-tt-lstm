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
from utils.convlstmnet import ConvLSTMNet 
from dataloader import KTH_Dataset, MNIST_Dataset

from utils.gpu_affinity import set_affinity

# perceptive quality
import PerceptualSimilarity.models as PSmodels


def main(args):
    ## Distributed computing

    # utility for synchronization
    def reduce_tensor(tensor):
        rt = tensor.clone()
        torch.distributed.all_reduce(rt, op = torch.distributed.ReduceOp.SUM)
        return rt

    # enable distributed computing
    if args.distributed:
        set_affinity(args.local_rank)
        num_devices = torch.cuda.device_count()
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend = 'nccl', init_method = 'env://')

        world_size  = torch.distributed.get_world_size() #os.environ['WORLD_SIZE']
        print('num_devices', num_devices, 
              'local_rank', args.local_rank, 
              'world_size', world_size)
    else: # if not args.distributed:
        num_devices, world_size = 1, 1

    ## Model preparation (Conv-LSTM or Conv-TT-LSTM)

    # construct the model with the specified hyper-parameters
    model = ConvLSTMNet(
        input_channels = args.img_channels, 
        output_sigmoid = args.use_sigmoid,
        # model architecture
        layers_per_block = (3, 3, 3, 3), 
        hidden_channels  = (32, 48, 48, 32), 
        skip_stride = 2,
        # convolutional tensor-train layers
        cell = args.model,
        cell_params = {
            "order": args.model_order, 
            "steps": args.model_steps, 
            "ranks": args.model_ranks},
        # convolutional parameters
        kernel_size = args.kernel_size).cuda()

    if args.distributed:
        if args.use_apex: # use DDP from apex.parallel
            from apex.parallel import DistributedDataParallel as DDP
            model = DDP(model, delay_allreduce = True)
        else: # use DDP from nn.parallel
            from torch.nn.parallel import DistributedDataParallel as DDP
            model = DDP(model, device_ids = [args.local_rank])

    PSmodel = PSmodels.PerceptualLoss(model = 'net-lin', 
        net = 'alex', use_gpu = True, gpu_ids = [args.local_rank])

    ## Dataset Preparation (KTH, UCF, tinyUCF)
    Dataset = {"KTH": KTH_Dataset, "MNIST": MNIST_Dataset}[args.dataset]

    DATA_DIR = os.path.join("../data", 
        {"MNIST": "mnist", "KTH": "kth"}[args.dataset])

    # batch size for each process
    total_batch_size  = args.batch_size
    assert total_batch_size % world_size == 0, \
        'The batch_size is not divisible by world_size.'
    batch_size = total_batch_size // world_size

    total_frames = args.input_frames + args.future_frames

    # dataloaer for the valiation dataset 
    test_data_path = os.path.join(DATA_DIR, args.test_data_file)
    assert os.path.exists(test_data_path), \
        "The test dataset does not exist. "+test_data_path

    test_dataset = Dataset({"path": test_data_path, 
        "unique_mode": True, "num_frames": total_frames, "num_samples": args.test_samples,
        "height": args.img_height, "width": args.img_width, "channels": args.img_channels, 'training': False})

    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset, num_replicas = world_size, rank = args.local_rank, shuffle = False)
    test_loader  = torch.utils.data.DataLoader(
        test_dataset, batch_size = batch_size, drop_last = True, 
        num_workers = num_devices * 4, pin_memory = True, sampler = test_sampler)

    test_samples = len(test_loader) * total_batch_size
    print(test_samples)

    ## Main script for test phase 
    MSE_  = torch.zeros((args.future_frames), dtype = torch.float32).cuda()
    PSNR_ = torch.zeros((args.future_frames), dtype = torch.float32).cuda()
    SSIM_ = torch.zeros((args.future_frames), dtype = torch.float32).cuda()
    PIPS_ = torch.zeros((args.future_frames), dtype = torch.float32).cuda()

    with torch.no_grad():
        model.eval()

        for it, frames in enumerate(test_loader):

            frames = frames.permute(0, 1, 4, 2, 3).cuda()
            inputs = frames[:,  :args.input_frames]
            origin = frames[:, -args.future_frames:]

            pred = model(inputs, 
                input_frames  =  args.input_frames, 
                future_frames = args.future_frames, 
                output_frames = args.future_frames, 
                teacher_forcing = False)

            # accumlate the statistics per frame
            for t in range(-args.future_frames, 0):
                origin_, pred_ = origin[:, t], pred[:, t]

                if args.img_channels == 1:
                    origin_ = origin_.repeat([1, 3, 1, 1])
                    pred_   =   pred_.repeat([1, 3, 1, 1])

                dist = PSmodel(origin_, pred_)
                PIPS_[t] += torch.sum(dist).item()

            origin = origin.permute(0, 1, 3, 4, 2).cpu().numpy()
            pred   =   pred.permute(0, 1, 3, 4, 2).cpu().numpy()

            for t in range(-args.future_frames, 0):
                for i in range(batch_size):
                    origin_, pred_ = origin[i, t], pred[i, t]

                    if args.img_channels == 1:
                        origin_ = np.squeeze(origin_, axis = -1)
                        pred_   = np.squeeze(pred_,   axis = -1)

                    MSE_[t]  += skimage.metrics.mean_squared_error(origin_, pred_)
                    PSNR_[t] += skimage.metrics.peak_signal_noise_ratio(origin_, pred_)
                    SSIM_[t] += skimage.metrics.structural_similarity(origin_, pred_, multichannel = args.img_channels > 1)

        if args.distributed:
            MSE  = reduce_tensor( MSE_) / test_samples
            PSNR = reduce_tensor(PSNR_) / test_samples
            SSIM = reduce_tensor(SSIM_) / test_samples
            PIPS = reduce_tensor(PIPS_) / test_samples
        else: # if not args.distributed:
            MSE  = MSE_  / test_samples
            PSNR = PSNR_ / test_samples
            SSIM = SSIM_ / test_samples
            PIPS = PIPS_ / test_samples

    if args.local_rank == 0:
        print("MSE: {} (x1e-3)\nPSNR: {}\nSSIM: {}\nLPIPS: {}".format(
            1e3 * torch.mean(MSE).cpu().item(), torch.mean(PSNR).cpu().item(), 
            torch.mean(SSIM).cpu().item(), torch.mean(PIPS).cpu().item()))

    print( "MSE:",  MSE.cpu().numpy())
    print("PSNR:", PSNR.cpu().numpy())
    print("SSIM:", SSIM.cpu().numpy())
    print("PIPS:", PIPS.cpu().numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Conv-TT-LSTM Test")

    ## Devices (Single GPU / Distributed computing)

    # whether to use distributed computing
    parser.add_argument('--use-distributed', dest = "distributed", 
        action = 'store_true',  help = 'Use distributed computing in testing.')
    parser.add_argument( '--no-distributed', dest = "distributed", 
        action = 'store_false', help = 'Use single process (GPU) in testing.')
    parser.set_defaults(distributed = True)

    parser.add_argument('--use-apex', dest = 'use_apex', 
        action = 'store_true',  help = 'Use apex.parallel.')
    parser.add_argument( '--no-apex', dest = 'use_apex', 
        action = 'store_false', help = 'Use torch.nn.distributed.')
    parser.set_defaults(use_apex = False)

    # arguments for distributed computing 
    parser.add_argument('--local_rank', default = 0, type = int)

    ## Data format (batch_size x time_steps x height x width x channels)

    # batch size (0) 
    parser.add_argument('--batch-size', default = 16, type = int,
        help = 'The total batch size in each test iteration.')

    # frame split (1)
    parser.add_argument('--input-frames',  default = 10, type = int,
        help = 'The number of input frames to the model.')
    parser.add_argument('--future-frames', default = 10, type = int,
        help = 'The number of predicted frames of the model.')

    # frame format (2, 3, 4)
    parser.add_argument('--img-channels', default =  3, type = int, 
        help = 'The number of channels in each video frame.')

    parser.add_argument('--img-height',   default = 120, type = int, 
        help = 'The image height of each video frame.')
    parser.add_argument('--img-width',    default = 120, type = int, 
        help = 'The image width  of each video frame.')

    ## Models (Conv-LSTM or Conv-TT-LSTM)

    # model type
    parser.add_argument('--model', default = 'convlstm', type = str,
        help = 'The model is either \"convlstm\"" or \"convttlstm\".')
    parser.add_argument('--checkpoint', default = "checkpoint.pt", type = str,
        help = 'The name for the checkpoint.')

    # output transformation
    parser.add_argument('--use-sigmoid', dest = 'use_sigmoid',
        action = 'store_true',  help = 'Use sigmoid function at the output of the model.')
    parser.add_argument('--no-sigmoid',  dest = 'use_sigmoid', 
        action = 'store_false', help = 'Use output from the last layer as the final output.')
    parser.set_defaults(use_sigmoid = False)

    # parameters of the convolutional tensor-train layers
    parser.add_argument('--model-order', default = 3, type = int, 
        help = 'The order of the convolutional tensor-train LSTMs.')
    parser.add_argument('--model-steps', default = 3, type = int, 
        help = 'The steps of the convolutional tensor-train LSTMs')
    parser.add_argument('--model-ranks',  default = 8, type = int, 
        help = 'The tensor rank of the convolutional tensor-train LSTMs.')
    
    # parameters of the convolutional operations
    parser.add_argument('--kernel-size', default = 5, type = int,
        help = "The kernel size of the convolutional operations.")

    ## Dataset (Input)
    parser.add_argument('--dataset', default = "KTH", type = str,
        help = 'The dataset name. (Options: KTH, MNIST)')

    parser.add_argument('--test-data-file', default = 'test', type = str, 
        help = 'Name of the folder/file for test set.')
    parser.add_argument('--test-samples', default = 5000, type = int, 
        help = 'Number of samples in test dataset.')

    main(parser.parse_args())
