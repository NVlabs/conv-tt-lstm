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
from tensorboardX import SummaryWriter
from utils.convlstmnet import ConvLSTMNet 
from dataloader import KTH_Dataset, MNIST_Dataset

from utils.gpu_affinity import set_affinity
from apex import amp

torch.backends.cudnn.benchmark = True

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

        world_size  = torch.distributed.get_world_size() #os.environ['WORLD_SIZE']
        print('num_devices', num_devices, 
              'local_rank', args.local_rank, 
              'world_size', world_size)
    else:
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

    ## Dataset Preparation (KTH, MNIST)
    Dataset = {"KTH": KTH_Dataset, "MNIST": MNIST_Dataset}[args.dataset]

    DATA_DIR = os.path.join("../data", 
        {"MNIST": "mnist", "KTH": "kth"}[args.dataset])

    # batch size for each process
    total_batch_size  = args.batch_size
    assert total_batch_size % world_size == 0, \
        'The batch_size is not divisible by world_size.'
    batch_size = total_batch_size // world_size

    # dataloader for the training dataset
    train_data_path = os.path.join(DATA_DIR, args.train_data_file)

    train_dataset = Dataset({"path": train_data_path, "unique_mode": False,
        "num_frames": args.input_frames + args.future_frames, "num_samples": args.train_samples, 
        "height": args.img_height, "width": args.img_width, "channels": args.img_channels, 'training': True})

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas = world_size, rank = args.local_rank, shuffle = True)

    train_loader  = torch.utils.data.DataLoader(
        train_dataset, batch_size = batch_size, drop_last = True,
        num_workers = num_devices * 4, pin_memory = True, sampler = train_sampler)

    train_samples = len(train_loader) * total_batch_size

    # dataloaer for the valiation dataset 
    valid_data_path = os.path.join(DATA_DIR, args.valid_data_file)

    valid_dataset = Dataset({"path": valid_data_path, "unique_mode": True,
        "num_frames": args.input_frames + args.future_frames, "num_samples": args.valid_samples,
        "height": args.img_height, "width": args.img_width, "channels": args.img_channels, 'training': False})

    valid_sampler = torch.utils.data.distributed.DistributedSampler(
        valid_dataset, num_replicas = world_size, rank = args.local_rank, shuffle = False)
    valid_loader  = torch.utils.data.DataLoader(
        valid_dataset, batch_size = batch_size, drop_last = True, 
        num_workers = num_devices * 4, pin_memory = True, sampler = valid_sampler)

    valid_samples = len(valid_loader) * total_batch_size

    # tensorboardX for logging learning curve
    if args.local_rank == 0:
        tensorboard = SummaryWriter()

    ## Main script for training and validation

    # loss function for training
    loss_func = lambda outputs, targets: \
        F.l1_loss(outputs, targets) + F.mse_loss(outputs, targets)

    # intialize the scheduled sampling ratio
    scheduled_sampling_ratio = 1
    ssr_decay_start = args.ssr_decay_start
    ssr_decay_mode  = False

    # initialize the learning rate
    learning_rate  = args.learning_rate
    lr_decay_start = args.num_epochs
    lr_decay_mode  = False

    # best model in validation loss
    min_epoch, min_loss = 0, float("inf")

    ## Main script for training and validation
    if args.use_fused:
        from apex.optimizers import FusedAdam
        optimizer = FusedAdam(model.parameters(), lr = learning_rate)
    else: # if not args.use_fused:
        optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    if args.use_amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level = "O1")

    if args.distributed:
        if args.use_apex: # use DDP from apex.parallel
            from apex.parallel import DistributedDataParallel as DDP
            model = DDP(model, delay_allreduce = True)
        else: # use DDP from nn.parallel
            from torch.nn.parallel import DistributedDataParallel as DDP
            model = DDP(model, device_ids = [args.local_rank])

    for epoch in range(args.num_epochs):

        ## Phase 1: Learning on the training set
        model.train()

        samples = 0
        for frames in train_loader:
            samples += total_batch_size

            frames = frames.permute(0, 1, 4, 2, 3).cuda()
            inputs = frames[:, :-1]
            origin = frames[:, -args.output_frames:]

            pred = model(inputs, 
                input_frames  =  args.input_frames, 
                future_frames = args.future_frames, 
                output_frames = args.output_frames, 
                teacher_forcing = True,
                scheduled_sampling_ratio = scheduled_sampling_ratio, 
                checkpointing = args.use_checkpointing)

            loss = loss_func(pred, origin)

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data)
            else: # if not args.distributed:
                reduced_loss = loss.data

            optimizer.zero_grad()
            
            if args.use_amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if args.gradient_clipping:
                    grad_norm = nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer), args.clipping_threshold)
            else: # if not args.use_amp:
                loss.backward()
                if args.gradient_clipping:
                    grad_norm = nn.utils.clip_grad_norm_(
                        model.parameters(), args.clipping_threshold)

            optimizer.step()

            if args.local_rank == 0:
                print('Epoch: {}/{}, Training: {}/{}, Loss: {}'.format(
                    epoch + 1, args.num_epochs, samples, train_samples, reduced_loss.item()))

        ## Phase 2: Evaluation on the validation set
        model.eval()

        with torch.no_grad():
            samples, LOSS = 0., 0.
            for it, frames in enumerate(valid_loader):
                samples += total_batch_size

                frames = frames.permute(0, 1, 4, 2, 3).cuda()
                inputs = frames[:,  :args.input_frames]
                origin = frames[:, -args.output_frames:]

                pred = model(inputs, 
                    input_frames  =  args.input_frames, 
                    future_frames = args.future_frames, 
                    output_frames = args.output_frames, 
                    teacher_forcing = False, 
                    checkpointing   = False)

                loss = loss_func(pred, origin)

                if args.distributed:
                    reduced_loss = reduce_tensor(loss.data)
                else: # if not args.distributed:
                    reduced_loss = loss.data

                LOSS += reduced_loss.item() * total_batch_size

            LOSS /= valid_samples

            if args.local_rank == 0:
                tensorboard.add_scalar("LOSS", LOSS, epoch + 1)

            if LOSS < min_loss:
                min_epoch, min_loss = epoch + 1, LOSS

        ## Phase 3: learning rate and scheduling sampling ratio adjustment
        if not ssr_decay_mode and epoch > ssr_decay_start \
            and epoch > min_epoch + args.decay_log_epochs:
            ssr_decay_mode = True
            lr_decay_start = epoch + args.lr_decay_start 

        if not  lr_decay_mode and epoch >  lr_decay_start \
            and epoch > min_epoch + args.decay_log_epochs:
            lr_decay_mode = True

        if ssr_decay_mode and (epoch + 1) % args.ssr_decay_epoch == 0:
            scheduled_sampling_ratio = max(
                scheduled_sampling_ratio - args.ssr_decay_ratio, 0)

        if  lr_decay_mode and (epoch + 1) % args.lr_decay_epoch == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= args.lr_decay_rate

    if args.local_rank == 0:
        torch.save(model.state_dict(), "checkpoint.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Conv-TT-LSTM Training")

    ## Devices (Single GPU / Distributed computing)

    # whether to use distributed computing
    parser.add_argument('--use-distributed', dest = "distributed", 
        action = 'store_true',  help = 'Use distributed computing in training.')
    parser.add_argument('--no-distributed',  dest = "distributed", 
        action = 'store_false', help = 'Use single process (GPU) in training.')
    parser.set_defaults(distributed = True)

    parser.add_argument('--use-apex', dest = 'use_apex', 
        action = 'store_true',  help = 'Use apex.parallel for distributed computing.')
    parser.add_argument( '--no-apex', dest = 'use_apex', 
        action = 'store_false', help = 'Use torch.nn.distributed for distributed computing.')
    parser.set_defaults(use_apex = False)

    parser.add_argument('--use-amp', dest = 'use_amp', 
        action = 'store_true',  help = 'Use automatic mixed precision in training.')
    parser.add_argument( '--no-amp', dest = 'use_amp', 
        action = 'store_false', help =  'No automatic mixed precision in training.')
    parser.set_defaults(use_amp = False)

    parser.add_argument('--use-fused', dest = 'use_fused', 
        action = 'store_true',  help = 'Use fused kernels in training.')
    parser.add_argument( '--no-fused', dest = 'use_fused',
        action = 'store_false', help =  'No fused kernels in training.')
    parser.set_defaults(use_fused = False)

    parser.add_argument('--use-checkpointing', dest = 'use_checkpointing', 
        action = 'store_true',  help = 'Use checkpointing to reduce memory utilization.')
    parser.add_argument( '--no-checkpointing', dest = 'use_checkpointing', 
        action = 'store_false', help = 'No checkpointing (faster training).')
    parser.set_defaults(use_checkpointing = False)

    parser.add_argument('--local_rank', default = 0, type = int)

    ## Data format (batch x steps x height x width x channels)

    # batch size (0) 
    parser.add_argument('--batch-size', default = 16, type = int,
        help = 'The total batch size in each training iteration.')

    # frame split (1)
    parser.add_argument('--input-frames',  default = 10, type = int,
        help = 'The number of input frames to the model.')
    parser.add_argument('--future-frames', default = 10, type = int,
        help = 'The number of predicted frames of the model.')
    parser.add_argument('--output-frames', default = 19, type = int,
        help = 'The number of output frames of the model.')

    # frame format (2, 3, 4)
    parser.add_argument('--img-height',  default = 120, type = int, 
        help = 'The image height of each video frame.')
    parser.add_argument('--img-width',   default = 120, type = int, 
        help = 'The image width of each video frame.')
    parser.add_argument('--img-channels', default =  3, type = int, 
        help = 'The number of channels in each video frame.')

    ## Models (Conv-LSTM or Conv-TT-LSTM)

    # model type and size (depth and width) 
    parser.add_argument('--model', default = 'convlstm', type = str,
        help = 'The model is either \"convlstm\", \"convttlstm\".')
    
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
    parser.add_argument('--model-ranks', default = 8, type = int, 
        help = 'The tensor rank of the convolutional tensor-train LSTMs.')

    # parameters of the convolutional operations
    parser.add_argument('--kernel-size', default = 5, type = int, 
        help = "The kernel size of the convolutional operations.")

    ## Dataset (Input to the training algorithm)
    parser.add_argument('--dataset', default = "KTH", type = str, 
        help = 'The dataset name. (Options: KTH, MNIST)')

    # training dataset
    parser.add_argument('--train-data-file', default = 'train', type = str,
        help = 'Name of the folder/file for training set.')
    parser.add_argument('--train-samples', default = 10000, type = int,
        help = 'Number of samples in each training epoch.')

    # validation dataset
    parser.add_argument('--valid-data-file', default = 'valid', type = str, 
        help = 'Name of the folder/file for validation set.')
    parser.add_argument('--valid-samples', default = 3000, type = int, 
        help = 'Number of unique samples in validation set.')

    ## Learning algorithm
    parser.add_argument('--num-epochs', default = 500, type = int, 
        help = 'Number of total epochs in training.')
    parser.add_argument('--decay-log-epochs', default = 20, type = int, 
        help = 'The window size to determine automatic scheduling.')

    # gradient clipping
    parser.add_argument('--gradient-clipping', dest = 'gradient_clipping', 
        action = 'store_true',  help = 'Use gradient clipping in training.')
    parser.add_argument(       '-no-clipping', dest = 'gradient_clipping', 
        action = 'store_false', help = 'No gradient clipping in training.')
    parser.set_defaults(gradient_clipping = False)

    parser.add_argument('--clipping-threshold', default = 1, type = float,
        help = 'The threshold value for gradient clipping.')

    # learning rate
    parser.add_argument('--learning-rate', default = 1e-3, type = float,
        help = 'Initial learning rate of the Adam optimizer.')
    parser.add_argument('--lr-decay-start', default = 20, type = int,
        help = 'The minimum epoch (after scheduled sampling) to start learning rate decay.')
    parser.add_argument('--lr-decay-epoch', default = 5, type = int,
        help = 'The learning rate is decayed every decay_epoch.')
    parser.add_argument('--lr-decay-rate', default = 0.98, type = float,
        help = 'The learning rate by decayed by decay_rate every epoch.')

    # scheduled sampling ratio
    parser.add_argument('--ssr-decay-start', default = 20, type = int,
        help = 'The minimum epoch to start scheduled sampling.')
    parser.add_argument('--ssr-decay-epoch', default =  1, type = int, 
        help = 'Decay the scheduled sampling every ssr_decay_epoch.')
    parser.add_argument('--ssr-decay-ratio', default = 2e-3, type = float,
        help = 'Decay the scheduled sampling by ssr_decay_ratio every time.')

    main(parser.parse_args())