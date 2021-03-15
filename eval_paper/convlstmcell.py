# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under a NVIDIA Open Source Non-commercial license

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.utils as utils

## Convolutional Tensor-Train LSTM Modules
class ConvTTLSTMCell(nn.Module):

    def __init__(self,
        # interface of the convolutional tensor-train LSTM module 
        input_channels, hidden_channels, 
        # convolutional tensor-train network
        version = "v4", concat_temporal = True,
        order = 3, steps = 3, ranks = 8, kernel_size = 5, bias = True
        ):
        """
        Initialization of convolutional tensor-train LSTM cell.

        Arguments:
        ----------
        (Hyper-parameters of the input/output channels/height/width)
        input_channels:  int
            Number of input channels of the input tensor.
        hidden_channels: int
            Number of hidden/output channels of the output tensor.
        Note: the number of hidden_channels is typically equal to the one of input_channels.

        hidden_height: int
            The height of hidden features in each channel.
        hidden_width: int
            The width  of hidden features in each channel.
        Note: the height/width of hidden features are equal to the ones for inputs. 

        (Hyper-parameters of the convolutional tensor-train format)
        version: str (options: "v1", "v2", "v3" or "v4")
            The version of convolutional tensor-train module.
            default: "v4" (the recent version)

        concat_temporal: bool
            In v3 and v4, the hidden states are concatenated before fed into convolutional tensor-train.
            The flag indicates whether the states are concatenated over the temporal axis.
            default: True (otherwise, they are concatenated over the channels axis.)

        order: int
            The order of convolutional tensor-train format (i.e. the number of core tensors).
            default: 3
        steps: int
            The total number of past steps used to compute the next step.
            default: 3
        ranks: int
            The ranks of convolutional tensor-train format (where all ranks are assumed to be the same).
            default: 8
        Notes: 
            For v1: ranks == hidden_channels;
            For v1 & v2: steps == order;
            For v4: order <= steps.

        (Hyper-parameters of the convolutional operations)
        kernel_size: int or (int, int)
            Size of the (squared) convolutional kernel.
            Note: If the size is a single scalar k, it will be mapped to (k, k)
            default: 5
        bias: bool
            Whether or not to add the bias in each convolutional operation.
            default: True
        """
        super(ConvTTLSTMCell, self).__init__()

        self.input_channels  = input_channels
        self.hidden_channels = hidden_channels

        ## (1) Convolutional tensor-train format
        self.version = version
        assert version in ["v1", "v2", "v3", "v4"], \
            "The specified version is not currently supported."

        # In v1, the ranks are equal to the number of hidden channels
        # if version == "v1": ranks = hidden_channels
        assert (not version == "v1") or (ranks == hidden_channels), \
            "For v1, the ranks are equal to the number of hidden_channels."

        # In v1 & v2, the order is equal to the number of steps
        # if version in ["v1", "v2"]: steps = order
        assert (not version in ["v1", "v2"]) or (steps == order), \
            "For v1 or v2, the number of steps is equal to the order."

        # In v4, the order is less than (or equal to) the number of steps
        assert (not version == "v4") or (steps >= order), \
            "For v4, the number of steps should be larger than (or equal to) the order."

        self.steps = steps
        self.order = order

        # In v3 & v4, whether the hidden states are concatenated temporally
        self.concat_temporal = concat_temporal if version in ["v3", "v4"] else False
        self.lags = 1 if version in ["v1", "v2"] else (
            steps if version == "v3" else (steps - order + 1))


        ## (2) Convolutional operations
        kernel_size = utils._pair(kernel_size)
        padding     = kernel_size[0] // 2, kernel_size[1] // 2

        # template for 2D convolutional operations
        Conv2d = lambda in_channels, out_channels: nn.Conv2d(
            in_channels = in_channels, out_channels = out_channels, 
            kernel_size = kernel_size, padding = padding, bias = bias)

        # template for 3D convolutional operations
        if version in ["v3", "v4"] and self.concat_temporal:
            Conv3d = lambda in_channels, out_channels: nn.Conv3d(
                in_channels = in_channels, out_channels = out_channels, bias = bias,
                kernel_size = kernel_size + (self.lags, ), padding = padding + (0,))

 
        ## (3) Construct and initialize the convolutional layers

        # convolutional layers in the tensor-train recurrent units
        self.layers = nn.ModuleList()
        for l in range(order):
            self.layers.append(Conv2d(
                in_channels  = ranks if l < order - 1 else ranks + input_channels, 
                out_channels = ranks if l < order - 1 else 4 * hidden_channels))

        self.init_conv_params(self.layers)

        # convolutional layers to the tensor-train recurrent units
        if version != "v1":
            self.layers_ = nn.ModuleList()
            for l in range(order):
                if version == "v2":
                    self.layers_.append(Conv2d(in_channels  = hidden_channels, 
                                               out_channels = ranks))
                else: # if version in ["v3", "v4"]:
                    self.layers_.append((Conv3d if concat_temporal else Conv2d)(
                        in_channels  = hidden_channels * (1 if self.concat_temporal else self.lags), 
                        out_channels = ranks))

            self.init_conv_params(self.layers_)

    def init_conv_params(self, layers):
        """
        Initialization of the parameters in the convolutional layers.

        Arguments:
        ----------
        layers: an nn.ModuleList object
            The list of convolutional layers that require initialization. 
        """

        # initialize the weights (and bias) explicitly 
        for layer in layers:
            torch.nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None: 
                torch.nn.init.zeros_(layer.bias)

    def init_states(self, inputs):
        """ 
        Initialization of the hidden/cell states of the convolutional tensor-train cell.

        Arguments:
        ----------
        inputs: 4-th order tensor of size [batch_size, input_channels, height, width]
            Input tensor to the convolutional tensor-train LSTM cell.
        """
        device = inputs.device # "cpu" or "cuda"
        batch_size, _, height, width = inputs.size()

        # initialize both hidden and cell states to all zeros
        self.hidden_states  = [torch.zeros(batch_size, self.hidden_channels, 
            height, width, device = device) for t in range(self.steps)]
        self.hidden_pointer = 0 # pointing to the position to be updated

        self.cell_states = torch.zeros(batch_size, 
            self.hidden_channels, height, width, device = device)

    def forward(self, inputs, first_step = False, checkpointing = False):
        """
        Computation of the convolutional tensor-train LSTM cell.
        
        Arguments:
        ----------
        inputs: a 4-th order tensor of size [batch_size, input_channels, height, width] 
            Input tensor to the convolutional-LSTM cell.

        first_step: bool
            Whether the tensor is the first step in the input sequence. 
            If so, both hidden and cell states are intialized to zeros tensors.
        
        Returns:
        --------
        hidden_states: another 4-th order tensor of size [batch_size, input_channels, height, width]
            Hidden states (and outputs) of the convolutional-LSTM cell.
        """

        if first_step: self.init_states(inputs) # intialize the states at the first step

        ## (1) Convolutional tensor-train module
        for l in range(self.order):
            input_pointer = self.hidden_pointer if l == 0 else (input_pointer + 1) % self.steps

            if self.version in ["v1", "v2"]:
                input_states = self.hidden_states[input_pointer]
                if self.version == "v2":
                    input_states = self.layers_[l](input_states)

            else: # if self.version in ["v3", "v4"]:
                input_states = self.hidden_states[input_pointer:] + self.hidden_states[:input_pointer]
                if self.version == "v4":
                    input_states = input_states[:self.lags]

                input_states = torch.stack(input_states, dim = -1) \
                    if self.concat_temporal else torch.cat(input_states, dim = 1)

                input_states = self.layers_[l](input_states)
                if self.concat_temporal: 
                    input_states = torch.squeeze(input_states, dim = -1)

            if l == 0:
                temp_states = input_states
            else: # if l > 0:
                temp_states = input_states + self.layers[l-1](temp_states)

                
        ## (2) Standard convolutional-LSTM module
        concat_conv = self.layers[-1](torch.cat([inputs, temp_states], dim = 1))

        cc_i, cc_f, cc_o, cc_g = torch.split(concat_conv, self.hidden_channels, dim = 1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        self.cell_states = f * self.cell_states + i * g
        outputs = o * torch.tanh(self.cell_states)
        
        self.hidden_states[self.hidden_pointer] = outputs
        self.hidden_pointer = (self.hidden_pointer + 1) % self.order
        
        return outputs
