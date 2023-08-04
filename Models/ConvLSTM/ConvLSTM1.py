import torch
import torch.nn as nn
from ConvLSTMCell import ConvLSTMCell
from EncoderDecoder import encoderBlock

class ConvLSTM1(nn.Module): 
    
    def __init__(self, in_channels, out_channels, 
    kernel_size, padding, activation, frame_size, isencoder=False):
         
        super(ConvLSTM1, self).__init__()
        self.isencoder = isencoder
        self.out_channels = out_channels
        if self.isencoder:
            self.convLSTMCell = ConvLSTMCell(4*in_channels, out_channels, 
                                           kernel_size, padding, activation, frame_size)
        else:
            self.convLSTMCell = ConvLSTMCell(in_channels, out_channels, 
                                           kernel_size, padding, activation, frame_size) 
        self.encoder = encoderBlock(in_channels, num_layer=2)
    
    def forward(self, X):
        
        # X is a frame sequence (batch_size, num_channels, seq_len, height, width)
        input_device = X.device
        # Get the dimensions
        batch_size, _, seq_len, height, width = X.size()
        height = 32
        width = 32

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
            
            if self.isencoder:
                En_x = self.encoder(X[:,:,time_step])
                Y, Z = self.convLSTMCell(En_x, Y, Z)
            else:
                Y, Z = self.convLSTMCell(X[:,:,time_step], Y, Z)

            output[:,:,time_step] = Y

        return output