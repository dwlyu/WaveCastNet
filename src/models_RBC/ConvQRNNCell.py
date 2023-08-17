import torch
import torch.nn as nn
import torch.nn.init as init
class QRNNCell(nn.Module):

    def __init__(self, in_channels, out_channels, 
    kernel_size, padding, activation, frame_size):

        super(QRNNCell, self).__init__()  

        if activation == "tanh":
            self.activation = torch.tanh 
        elif activation == "relu":
            self.activation = torch.relu   

        # Initialize weights for Hadamard Products
        self.W_ci = nn.Parameter(torch.Tensor(out_channels, *frame_size))
        self.W_co = nn.Parameter(torch.Tensor(out_channels, *frame_size))
        self.W_cf = nn.Parameter(torch.Tensor(out_channels, *frame_size))
        self.reset_parameters(out_channels)
        
    def reset_parameters(self, out_channels):
        for param in self.parameters():
            if len(param.shape) > 1:  # Initialize only the weight tensors, not bias
                init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')   
        

    def forward(self, X, H_prev, C_prev):
        
        i_conv, f_conv, C_conv, o_conv = torch.chunk(X, chunks=4, dim=1)

        input_gate = torch.sigmoid(i_conv + self.W_ci * C_prev )
        forget_gate = torch.sigmoid(f_conv + self.W_cf * C_prev )

        # Current Cell output
        C = forget_gate*C_prev + input_gate * self.activation(C_conv)

        output_gate = torch.sigmoid(o_conv + self.W_co * C )

        # Current Hidden State
        H = output_gate * self.activation(C)

        return H, C