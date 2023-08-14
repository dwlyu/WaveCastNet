import torch
import torch.nn as nn
from .ConvLEMCell import ConvLEMCell

#device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu') 

class ConvLEM(nn.Module): 
    
    def __init__(self, dt, in_channels, out_channels, 
    kernel_size, padding, activation, frame_size):
        
        super(ConvLEM, self).__init__()
        
        self.out_channels = out_channels
        self.convLEMCell = ConvLEMCell(dt, in_channels, out_channels, 
        kernel_size, padding, activation, frame_size)
    
    def forward(self, X):
        
        # X is a frame sequence (batch_size, num_channels, seq_len, height, width)
        input_device = X.device
        # Get the dimensions
        batch_size, _, seq_len, height, width = X.size()

        # Initialize output
        output = torch.zeros(batch_size, self.out_channels, seq_len, 
        height, width).to(input_device)
        
        # Initialize Hidden State
        Y = torch.zeros(batch_size, self.out_channels, 
        height, width).to(input_device)

        # Initialize Cell Input
        Z = torch.zeros(batch_size,self.out_channels, 
        height, width).to(input_device)

        # Unroll over time steps
        for time_step in range(seq_len):

            Y, Z = self.convLEMCell(X[:,:,time_step], Y, Z)

            output[:,:,time_step] = Y

        return output
