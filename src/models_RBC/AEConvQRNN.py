import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
from .ConvQRNNCell import QRNNCell
from EncoderDecoder import encoderBlock
from EncoderDecoder import decoderBlock

class AEConvQRNN(nn.Module):
    def __init__(self, num_channels, num_kernels, kernel_size, padding, 
    activation, frame_size):
        super(AEConvQRNN, self).__init__()
        
        self.out_channels = num_kernels
        self.encoder = encoderBlock(num_channels, num_layer=2)
        self.decoder = decoderBlock(num_channel= 4*num_channels, num_layer=2)
        self.encoder_1_convqrnn = QRNNCell(
                in_channels=4*num_channels, out_channels=num_kernels,
                kernel_size=kernel_size, padding=padding, 
                activation=activation, frame_size=frame_size)

        self.encoder_2_convqrnn = QRNNCell(
                in_channels=num_kernels, out_channels=num_kernels,
                kernel_size=kernel_size, padding=padding, 
                activation=activation, frame_size=frame_size)

        self.decoder_1_convqrnn = QRNNCell(
                in_channels=num_kernels, out_channels=num_kernels,
                kernel_size=kernel_size, padding=padding, 
                activation=activation, frame_size=frame_size)

        self.decoder_2_convqrnn = QRNNCell(
                in_channels=num_kernels, out_channels=num_kernels,
                kernel_size=kernel_size, padding=padding, 
                activation=activation, frame_size=frame_size)
        
        self.padding =nn.ConstantPad3d((1,1,1,1,1,0),0)
        self.embedconv1 = nn.Conv3d(in_channels=4*num_channels, out_channels=4*num_kernels,
            kernel_size=(2,3,3), padding=(0,0,0))
        
        self.embedconv2 = nn.Conv3d(in_channels=num_kernels, out_channels=4*num_kernels,
            kernel_size=(2,3,3), padding=(0,0,0))
        
        self.embedconv3 = nn.Conv3d(in_channels=num_kernels, out_channels=4*num_kernels,
            kernel_size=(2,3,3), padding=(0,0,0))

        self.conv = nn.Conv2d(
            in_channels=num_kernels, out_channels=4 * num_channels,
            kernel_size=kernel_size, padding=padding)
        
        self.conv1 = nn.Conv2d(
            in_channels=num_channels, out_channels=num_channels,
            kernel_size=kernel_size, padding=padding)
        
        self.reset_parameters(num_channels)
        
    def reset_parameters(self, out_channels):
        for param in self.parameters():
            if len(param.shape) > 1:  # Initialize only the weight tensors, not bias
                init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu') 


    def autoencoder(self, x, future_step, h_t, c_t, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4):

        batch_size, _, seq_len, height, width = x.size()
        outputs = torch.zeros(batch_size, _, future_step, 
        height, width).to(x.device)
        
        encoded_x = torch.zeros(batch_size, 4 * _, seq_len, 
        (int)(height/4), (int)(width/4)).to(x.device)
        
        H_t = torch.zeros(batch_size, self.out_channels, seq_len, 
        (int)(height/4), (int)(width/4)).to(x.device)
        
        H_t1 = torch.zeros(batch_size, self.out_channels, future_step, 
        (int)(height/4), (int)(width/4)).to(x.device)
        
        
        for t in range(seq_len):
            encoded_x[:,:,t] = self.encoder(x[:,:,t])
        
        embedded_x = self.embedconv1(self.padding(encoded_x))
        
        for t in range(seq_len):
            h_t, c_t = self.encoder_1_convqrnn(embedded_x[:,:,t],
                                              h_t,c_t)  # we could concat to provide skip conn here
            H_t[:,:,t] = h_t
        
        embedded_h = self.embedconv2(self.padding(H_t))
        
        for t in range(seq_len):
            h_t2, c_t2 = self.encoder_2_convqrnn(embedded_h[:,:,t],
                                              h_t,c_t)  # we could concat to provide skip conn here
            H_t[:,:,t] = h_t2
            
        h_t3 = h_t
        c_t3 = c_t
        h_t4 = h_t2
        c_t4 = c_t2
        
        for t in range(future_step):
            h_t3, c_t3 = self.decoder_1_convqrnn(embedded_x[:,:,t],
                                              h_t3,c_t3)  # we could concat to provide skip conn here
            H_t1[:,:,t] = h_t3
            
        embedded_h = self.embedconv3(self.padding(H_t1))

        for t in range(future_step):
            h_t4, c_t4 = self.decoder_2_convqrnn(embedded_h[:,:,t],
                                                h_t4, c_t4)
            output = self.conv(h_t4)
            outputs[:,:,t] = self.conv1(self.decoder(nn.LeakyReLU()(output)))
                
        return outputs
    
    def forward(self, x, future_seq, hidden_state=None):
        
        input_device = x.device
        batch_size,_,seq_len, height, width = x.size()
        height = (int)(height/4)
        width = (int)(width/4)
        
        # Initialize Hidden State
        h_t = torch.zeros(batch_size, self.out_channels, 
        height, width).to(input_device)
        c_t = torch.zeros(batch_size,self.out_channels, 
        height, width).to(input_device)
        
        h_t2 = torch.zeros(batch_size, self.out_channels, 
        height, width).to(input_device)
        c_t2 = torch.zeros(batch_size,self.out_channels, 
        height, width).to(input_device)
        
        h_t3 = torch.zeros(batch_size, self.out_channels, 
        height, width).to(input_device)
        c_t3 = torch.zeros(batch_size,self.out_channels, 
        height, width).to(input_device)
        
        h_t4 = torch.zeros(batch_size, self.out_channels, 
        height, width).to(input_device)
        c_t4 = torch.zeros(batch_size,self.out_channels, 
        height, width).to(input_device)
        
        outputs = self.autoencoder(x, future_seq, h_t, c_t, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4)

        return outputs