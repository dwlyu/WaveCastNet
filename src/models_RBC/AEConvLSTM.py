import torch
import torch.nn as nn
import numpy as np
from ConvLSTMCell import ConvLSTMCell
from EncoderDecoder import encoderBlock
from EncoderDecoder import decoderBlock


class AEConvLSTM(nn.Module):
    def __init__(self, num_channels, num_kernels, kernel_size, padding, 
    activation, frame_size):
        super(AEConvLSTM, self).__init__()

        self.out_channels = num_kernels
        self.encoder = encoderBlock(num_channels, num_layer=2)
        self.decoder = decoderBlock(num_channel= 4*num_channels, num_layer=2)
        self.encoder_1_convlstm = ConvLSTMCell(
                in_channels=4*num_channels, out_channels=num_kernels,
                kernel_size=kernel_size, padding=padding, 
                activation=activation, frame_size=frame_size)

        self.encoder_2_convlstm = ConvLSTMCell(
                in_channels=num_kernels, out_channels=num_kernels,
                kernel_size=kernel_size, padding=padding, 
                activation=activation, frame_size=frame_size)

        self.decoder_1_convlstm = ConvLSTMCell(
                in_channels=num_kernels, out_channels=num_kernels,
                kernel_size=kernel_size, padding=padding, 
                activation=activation, frame_size=frame_size)

        self.decoder_2_convlstm = ConvLSTMCell(
                in_channels=num_kernels, out_channels=num_kernels,
                kernel_size=kernel_size, padding=padding, 
                activation=activation, frame_size=frame_size)

        self.conv = nn.Conv2d(
            in_channels=num_kernels, out_channels=4 * num_channels,
            kernel_size=kernel_size, padding=padding)
        
        self.conv1 = nn.Conv2d(
            in_channels=num_channels, out_channels=num_channels,
            kernel_size=kernel_size, padding=padding)


    def autoencoder(self, x, future_step, h_t, c_t, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4):

        batch_size, _, seq_len, height, width = x.size()
        outputs = torch.zeros(batch_size, _, future_step, 
        height, width).to(x.device)
        for t in range(seq_len):
            h_t, c_t = self.encoder_1_convlstm(self.encoder(x[:,:,t]),
                                              h_t,c_t)  # we could concat to provide skip conn here
            h_t2, c_t2 = self.encoder_2_convlstm(h_t,
                                                h_t2, c_t2)
        h_t3 = h_t
        c_t3 = c_t
        h_t4 = h_t2
        c_t4 = c_t2
        encoder_vector = torch.rand_like(h_t)
        for t in range(future_step):
            h_t3, c_t3 = self.decoder_1_convlstm(encoder_vector,
                                                 h_t3, c_t3)  # we could concat to provide skip conn here
            h_t4, c_t4 = self.decoder_2_convlstm(h_t3,
                                                h_t4, c_t4)
            output = self.conv(h_t4)
            encoder_vector = h_t4
            outputs[:,:,t] = self.conv1(self.decoder(nn.LeakyReLU()(output)))
                
        return outputs
        
    def forward(self, x, future_seq, hidden_state=None):
        input_device = x.device
        batch_size,_,seq_len, height, width = x.size()
        height = (int)(height/4)
        width = (int)(width/4)

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