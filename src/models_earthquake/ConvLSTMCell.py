import torch
import torch.nn as nn
import math
import torch.nn.init as init
class ConvLSTMCell(nn.Module):

    def __init__(self, in_channels, out_channels, 
    kernel_size, padding, activation, frame_size):

        super(ConvLSTMCell, self).__init__()  

        if activation == "tanh":
            self.activation = torch.tanh 
        elif activation == "relu":
            self.activation = torch.relu
        
        self.conv = nn.Conv2d(
            in_channels=in_channels + out_channels, 
            out_channels=4 * out_channels, 
            kernel_size=kernel_size, 
            padding=padding)           

        # Initialize weights for Hadamard Products
        self.W_ci = nn.Parameter(torch.Tensor(out_channels, *frame_size))
        self.W_co = nn.Parameter(torch.Tensor(out_channels, *frame_size))
        self.W_cf = nn.Parameter(torch.Tensor(out_channels, *frame_size))
        self.reset_parameters(out_channels)
        
    def reset_parameters(self, out_channels):
        for param in self.parameters():
            if len(param.shape) > 1:  # Initialize only the weight tensors, not bias
                init.xavier_uniform_(param)   
            else:
                nn.init.constant_(param, 0)
           

    def forward(self, X, H_prev, C_prev):

        conv_output = self.conv(torch.cat([X, H_prev], dim=1))

        i_conv, f_conv, C_conv, o_conv = torch.chunk(conv_output, chunks=4, dim=1)

        input_gate = torch.sigmoid(i_conv + self.W_ci * C_prev )
        forget_gate = torch.sigmoid(f_conv + self.W_cf * C_prev )

        C = forget_gate*C_prev + input_gate * self.activation(C_conv)

        output_gate = torch.sigmoid(o_conv + self.W_co * C )

        H = output_gate * self.activation(C)

        return H, C