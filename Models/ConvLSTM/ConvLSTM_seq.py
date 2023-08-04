import torch.nn as nn
import torch
import numpy as np
from ConvLSTM1 import ConvLSTM1
from ConvLSTMres import ConvLSTMres
from EncoderDecoder import decoderBlock
from EncoderDecoder import decoder_residual
        
class ShiftMean1(nn.Module):
    # data: [t,c,h,w]
    def __init__(self):
        super(ShiftMean1, self).__init__()
    def forward(self, x, mode, process):
        max = np.load(f'/scratch/dongwei/test/Climate_Forecasting/SST/'+f'{process}_max.npy')
        min = np.load(f'/scratch/dongwei/test/Climate_Forecasting/SST/'+f'{process}_min.npy')
        torch_mean = torch.from_numpy(max)
        torch_std = torch.from_numpy(min)
        # torch_mean = torch.from_numpy(mean).view(1,1,360,360)
        # torch_std = torch.from_numpy(std) .view(1,1,360,360)
        if mode == 'sub':
            return (x - torch_mean.to(x.device)) /torch_std.to(x.device)
        elif mode == 'add':
            return x * torch_std.to(x.device) + torch_mean.to(x.device)
        else:
            raise NotImplementedError

class Seq2Seqlstm(nn.Module):

    def __init__(self, num_channels, num_kernels, kernel_size, padding, 
    activation, frame_size, num_layers):

        super(Seq2Seqlstm, self).__init__()
        

        self.sequential = nn.Sequential()
        
        self.decoder = decoderBlock(num_channel= 4*num_channels, num_layer=2)
        self.shiftmean = ShiftMean1()
        # Add First layer (Different in_channels than the rest)
        self.sequential.add_module(
            "convlstm1", ConvLSTM1(
                in_channels=num_channels, out_channels=num_kernels,
                kernel_size=kernel_size, padding=padding, 
                activation=activation, frame_size=frame_size,isencoder=True)
        )

        self.sequential.add_module(
            "batchnorm1", nn.BatchNorm3d(num_features=num_kernels)
        ) 

        # Add rest of the layers
        for l in range(2, num_layers+1):

            self.sequential.add_module(
                f"convlem{l}", ConvLSTM1(
                    in_channels=num_kernels, out_channels=num_kernels,
                    kernel_size=kernel_size, padding=padding, 
                    activation=activation, frame_size=frame_size)
                )
                
            self.sequential.add_module(
                f"batchnorm{l}", nn.BatchNorm3d(num_features=num_kernels)
                ) 

        # Add Convolutional Layer to predict output frame
        self.conv = nn.Conv2d(
            in_channels=num_kernels, out_channels=4*num_channels,
            kernel_size=kernel_size, padding=padding)
        self.conv1 = nn.Conv2d(
            in_channels=num_channels, out_channels=num_channels,
            kernel_size=kernel_size, padding=padding)

    def forward(self, X):
        batch_size, _, seq_len, height, width = X.size()
        # X_norm = self.shiftmean(X, process=process, mode='sub')
        out_decoder = torch.zeros(batch_size, _, seq_len, height, width).to(X.device)
        # Forward propagation through all the layers
        output = self.sequential(X)
        # Return only the last output frame
        for l in range(seq_len):
            y = self.conv(output[:,:,l])
            out_decoder[:,:,l] =self.conv1(self.decoder(nn.LeakyReLU()(y)))
        # out = self.shiftmean(out_decoder, process=process, mode='add')
        return out_decoder