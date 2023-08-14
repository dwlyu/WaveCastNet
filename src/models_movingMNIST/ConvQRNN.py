import torch
import torch.nn as nn
from .ConvQRNNCell import QRNNCell

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ConvQRNN(nn.Module):

    def __init__(self, in_channels, out_channels, 
    kernel_size, padding, activation, frame_size):

        super(ConvQRNN, self).__init__()

        self.out_channels = out_channels

        # We will unroll this over time steps
        self.QRNNcell = QRNNCell(in_channels, out_channels, 
        kernel_size, padding, activation, frame_size)
        
        self.embedconv = nn.Conv3d(in_channels = in_channels, out_channels = 4 * out_channels,
            kernel_size=(2,3,3), padding=(0,0,0))
        
        self.padding =nn.ConstantPad3d((1,1,1,1,1,0),0)

    def forward(self, X):

        # X is a frame sequence (batch_size, num_channels, seq_len, height, width)

        # Get the dimensions
        input_device = X.device
        batch_size, _, seq_len, height, width = X.size()
        embedded_X = self.embedconv(self.padding(X))

        # Initialize output
        output = torch.zeros(batch_size, self.out_channels, seq_len, 
        height, width).to(input_device)
        
        # Initialize Hidden State
        H = torch.zeros(batch_size, self.out_channels, 
        height, width).to(input_device)

        # Initialize Cell Input
        C = torch.zeros(batch_size,self.out_channels, 
        height, width).to(input_device)

        # Unroll over time steps
        for time_step in range(seq_len):

            H, C = self.QRNNcell(embedded_X[:,:,time_step], H, C)

            output[:,:,time_step] = H

        return output
