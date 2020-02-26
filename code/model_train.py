# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under a NVIDIA Open Source Non-commercial license.

# system modules
import os, argparse

# basic pytorch modules
import torch
import torch.nn as nn
import torch.nn.functional as F

# custom utilities
from dataloader import MNIST_Dataset, KTH_Dataset
from convlstmnet import ConvLSTMNet


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


    ## Dataset Preparation (Moving-MNIST, KTH)
    Dataset = {"MNIST": MNIST_Dataset, "KTH": KTH_Dataset}[args.dataset]

    DATA_DIR = os.path.join("../../datasets", 
        {"MNIST": "moving-mnist", "KTH": "kth"}[args.dataset])

    # number of total frames
    total_frames = args.input_frames + args.future_frames

    # dataloaer for the training set
    train_data_path = os.path.join(DATA_DIR, args.train_data_file)

    train_data = Dataset({"path": train_data_path, "unique_mode": False,
        "num_samples": args.train_samples, "num_frames": total_frames,  
        "height": args.img_height, "width": args.img_width, "channels": args.img_channels})

    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size = args.batch_size,
        shuffle = True, num_workers = 5 * max(num_gpus, 1), drop_last = True)

    train_size = len(train_data_loader) * args.batch_size

    # dataloaer for the valiation set 
    valid_data_path = os.path.join(DATA_DIR, args.valid_data_file)

    valid_data = Dataset({"path": valid_data_path, "unique_mode": True,
        "num_samples": args.valid_samples, "num_frames": total_frames, 
        "height": args.img_height, "width": args.img_width, "channels": args.img_channels})

    valid_data_loader = torch.utils.data.DataLoader(valid_data, batch_size = args.batch_size, 
        shuffle = False, num_workers = 5 * max(num_gpus, 1), drop_last = True)

    valid_size = len(valid_data_loader) * args.batch_size


    ## Main script for training and validation

    # loss function for training
    loss_func = lambda pred, origin: (
        F.l1_loss( pred, origin, reduction = "mean") + 
        F.mse_loss(pred, origin, reduction = "mean"))

    # scheduling sampling
    ssr_decay_mode = False
    scheduled_sampling_ratio = 1

    # optimizer and learning rate
    lr_decay_mode = False
    optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate)

    # best model in validation loss
    min_epoch, min_loss = 0, float("inf")

    for epoch in range(0, args.num_epochs):

        ## Phase 1: Learning on the training set
        model.train()

        samples = 0
        for frames in train_data_loader:
            samples += args.batch_size

            frames = frames.permute(0, 1, 4, 2, 3).to(device)

            inputs = frames[:, :-1] 
            origin = frames[:, -args.output_frames:]

            pred = model(inputs, 
                input_frames  =  args.input_frames, 
                future_frames = args.future_frames, 
                output_frames = args.output_frames, 
                teacher_forcing = True, 
                scheduled_sampling_ratio = scheduled_sampling_ratio)

            loss = loss_func(pred, origin)
            loss.backward()
            
            if args.gradient_clipping: 
                nn.utils.clip_grad_norm_(
                    model.parameters(), args.clipping_threshold)

            optimizer.step()
            optimizer.zero_grad()

            print('Epoch: {}/{}, Training: {}/{}, Loss: {}'.format(
                epoch + 1, args.num_epochs, samples, train_size, loss.item()))

        # adjust the learning rate of the optimizer
        if lr_decay_mode and (epoch + 1) % args.lr_decay_epoch == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= args.lr_decay_rate

        # adjust the scheduled sampling ratio
        if ssr_decay_mode and (epoch + 1) % args.ssr_decay_epoch == 0:
            scheduled_sampling_ratio = max(scheduled_sampling_ratio - args.ssr_decay_ratio, 0) 


        ## Phase 2: Evaluation on the validation set
        model.eval()

        with torch.no_grad():
           
            samples, LOSS = 0, 0.
            for frames in valid_data_loader:
                samples += args.batch_size

                frames = frames.permute(0, 1, 4, 2, 3).to(device)

                inputs = frames[:,  :args.input_frames]
                origin = frames[:, -args.output_frames:]

                pred = model(inputs, 
                    input_frames  =  args.input_frames, 
                    future_frames = args.future_frames, 
                    output_frames = args.output_frames, 
                    teacher_forcing = False)

                LOSS += loss_func(pred, origin).item() * args.batch_size

        LOSS /= valid_size
        if LOSS < min_loss:
            min_epoch, min_loss = epoch + 1, LOSS

        ## Phase 3: learning rate and scheduling sampling ratio adjustment
        if not ssr_decay_mode and epoch > min_epoch + args.decay_log_epochs:
            min_epoch = epoch
            ssr_decay_mode = True

        if not  lr_decay_mode and epoch > min_epoch + args.decay_log_epochs:
           lr_decay_mode  = True

    torch.save(model.state_dict(), "checkpoint.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Conv-TT-LSTM Training")

    ## Data format (batch_size x time_steps x height x width x channels)

    # batch size and the logging period 
    parser.add_argument('--batch-size',  default =  16, type = int,
        help = 'The batch size in training phase.')

    # frame split
    parser.add_argument('--input-frames',  default = 10, type = int,
        help = 'The number of input frames to the model.')
    parser.add_argument('--future-frames', default = 10, type = int,
        help = 'The number of predicted frames of the model.')
    parser.add_argument('--output-frames', default = 19, type = int,
        help = 'The number of output frames of the model.')

    # frame format
    parser.add_argument('--img-height',  default = 64, type = int, 
        help = 'The image height of each video frame.')
    parser.add_argument('--img-width',   default = 64, type = int, 
        help = 'The image width  of each video frame.')
    parser.add_argument('--img-channels', default = 1, type = int, 
        help = 'The number of channels in each video frame.')

    ## Devices (CPU, single-GPU or multi-GPU)

    # whether to use GPU for training
    parser.add_argument('--use-cuda', dest = 'use_cuda', action = 'store_true',
        help = 'Use GPU for training.')
    parser.add_argument('--no-cuda',  dest = 'use_cuda', action = 'store_false', 
        help = "Do not use GPU for training.")
    parser.set_defaults(use_cuda = True)

    # whether to use multi-GPU for training
    parser.add_argument('--multi-gpu',  dest = 'multi_gpu', action = 'store_true',
        help = 'Use multiple GPUs for training.')
    parser.add_argument('--single-gpu', dest = 'multi_gpu', action = 'store_false',
        help = 'Do not use multiple GPU for training.')
    parser.set_defaults(multi_gpu = True)

    ## Models (Conv-LSTM or Conv-TT-LSTM)

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

    ## Dataset (Input to the training algorithm)
    parser.add_argument('--dataset', default = "MNIST", type = str, 
        help = 'The dataset name. (Options: MNIST, KTH)')

    # training set
    parser.add_argument('--train-data-file', default = 'moving-mnist-train-new1.npz', type = str,
        help = 'Name of the folder/file for training set.')
    parser.add_argument('--train-samples', default = 10000, type = int,
        help = 'Number of unique samples in training set.')

    # validation set
    parser.add_argument('--valid-data-file', default = 'moving-mnist-val-new1.npz', type = str, 
        help = 'Name of the folder/file for validation set.')
    parser.add_argument('--valid-samples', default = 3000, type = int, 
        help = 'Number of unique samples in validation set.')

    ## Learning algorithm
    parser.add_argument('--num-epochs', default = 400, type = int, 
        help = 'Number of total epochs in training.')

    parser.add_argument('--decay-log-epochs', default = 20, type = int, 
        help = 'The window size to determine automatic scheduling.')

    # gradient clipping
    parser.add_argument('--gradient-clipping', dest = 'gradient_clipping', 
        action = 'store_true',  help = 'Use gradient clipping in training.')
    parser.add_argument('--no-clipping', dest = 'gradient_clipping', 
        action = 'store_false', help =  'No gradient clipping in training.')
    parser.set_defaults(use_clipping = False)

    parser.add_argument('--clipping-threshold', default = 3, type = float,
        help = 'The threshold value for gradient clipping.')

    # learning rate scheduling
    parser.add_argument('--learning-rate', default = 1e-3, type = float,
        help = 'Initial learning rate of the Adam optimizer.')
    parser.add_argument('--lr-decay-epoch', default = 5, type = int, 
        help = 'The learning rate is decayed every lr_decay_epoch.')
    parser.add_argument('--lr-decay-rate', default = 0.98, type = float,
        help = 'The learning rate by decayed by decay_rate every lr_decay_epoch.')

    # scheduled sampling ratio
    parser.add_argument('--ssr-decay-epoch', default = 1, type = int, 
        help = 'Decay the scheduled sampling every ssr_decay_epoch.')
    parser.add_argument('--ssr-decay-ratio', default = 4e-3, type = float,
        help = 'Decay the scheduled sampling by ssr_decay_ratio every time.')

    main(parser.parse_args())
