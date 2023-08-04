import torch
import torch.nn as nn
import numpy as np
from ConvLEMCell import ConvLEMCell
from EncoderDecoder import encoderBlock
from EncoderDecoder import encoder_residual
from EncoderDecoder import decoderBlock
from EncoderDecoder import decoder_residual

class ShiftMean(nn.Module):
    # data: [t,c,h,w]
    def __init__(self):
        super(ShiftMean, self).__init__()
    def forward(self, x, mode, process):
        mean = np.load(f'/scratch/dongwei/test/Climate_Forecasting/'+f'{process}_mean.npy')
        std = np.load(f'/scratch/dongwei/test/Climate_Forecasting/'+f'{process}_std.npy')
        torch_mean = torch.from_numpy(mean).view(1,1,360,360)
        torch_std = torch.from_numpy(std) .view(1,1,360,360)
        if mode == 'sub':
            return (x - torch_mean.to(x.device)) /torch_std.to(x.device)
        elif mode == 'add':
            return x * torch_std.to(x.device) + torch_mean.to(x.device)
        else:
            raise NotImplementedError

class AEConvLEM(nn.Module):
    def __init__(self, dt, num_channels, num_kernels, kernel_size, padding, 
    activation, frame_size):
        super(AEConvLEM, self).__init__()

        """ ARCHITECTURE 

        # Encoder (ConvLSTM)
        # Encoder Vector (final hidden state of encoder)
        # Decoder (ConvLSTM) - takes Encoder Vector as input
        # Decoder (3D CNN) - produces regression predictions for our model

        """
        self.out_channels = num_kernels
        self.encoder = encoderBlock(num_channels, num_layer=2)
        self.decoder = decoderBlock(num_channel= 4*num_channels, num_layer=2)
        self.shiftmean = ShiftMean()
        self.encoder_1_convlem = ConvLEMCell(dt = dt,
                in_channels=4*num_channels, out_channels=num_kernels,
                kernel_size=kernel_size, padding=padding, 
                activation=activation, frame_size=frame_size)

        self.encoder_2_convlem = ConvLEMCell(dt = dt,
                in_channels=num_kernels, out_channels=num_kernels,
                kernel_size=kernel_size, padding=padding, 
                activation=activation, frame_size=frame_size)

        self.decoder_1_convlem = ConvLEMCell(dt = dt,
                in_channels=num_kernels, out_channels=num_kernels,
                kernel_size=kernel_size, padding=padding, 
                activation=activation, frame_size=frame_size)

        self.decoder_2_convlem = ConvLEMCell(dt = dt,
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
            h_t, c_t = self.encoder_1_convlem(self.encoder(x[:,:,t]),
                                              h_t,c_t)  # we could concat to provide skip conn here
            h_t2, c_t2 = self.encoder_2_convlem(h_t,
                                                h_t2, c_t2)
        h_t3 = h_t
        c_t3 = c_t
        h_t4 = h_t2
        c_t4 = c_t2
        encoder_vector = torch.rand_like(h_t)
        for t in range(future_step):
            h_t3, c_t3 = self.decoder_1_convlem(encoder_vector,
                                                 h_t3, c_t3)  # we could concat to provide skip conn here
            h_t4, c_t4 = self.decoder_2_convlem(h_t3,
                                                h_t4, c_t4)
            output = self.conv(h_t4)
            encoder_vector = h_t4
            outputs[:,:,t] = self.conv1(self.decoder(nn.LeakyReLU()(output)))
                
        return outputs
        # if process == 'train':
        #     input_len = seq_len - future_step-1
        #     for t in range(input_len):
        #         h_t, c_t = self.encoder_1_convlem(self.encoder(x[:,:,t]),
        #                                           h_t,c_t)
        #         h_t2, c_t2 = self.encoder_2_convlem(h_t,
        #                                             h_t2,c_t2)
        #     h_t3 = h_t
        #     c_t3 = c_t
        #     h_t4 = h_t2
        #     c_t4 = c_t2
        #     # encoder_vector = self.decoder(nn.LeakyReLU()(self.conv(h_t2)))
        #     for t in range(future_step):
        #         h_t3, c_t3 = self.decoder_1_convlem(self.encoder(x[:,:,input_len+t-1]),
        #                                          h_t3, c_t3)
        #         # if t ==0 :
        #         #     h_t3, c_t3 = self.decoder_1_convlem(self.encoder(encoder_vector),
        #         #                                  h_t3, c_t3)
        #         # else:
        #         #     h_t3, c_t3 = self.decoder_1_convlem(self.encoder(x[:,:,input_len+t-1]),
        #         #                                  h_t3, c_t3)
        #         h_t4, c_t4 = self.decoder_2_convlem(h_t3,
        #                                         h_t4, c_t4)
        #         output = self.conv(h_t4)
        #         outputs[:,:,t] = self.decoder(nn.LeakyReLU()(output))
        #     return outputs
        # else:
        #     for t in range(seq_len):
        #         h_t, c_t = self.encoder_1_convlem(self.encoder(x[:,:,t]),
        #                                       h_t,c_t)  # we could concat to provide skip conn here
        #         h_t2, c_t2 = self.encoder_2_convlem(h_t,
        #                                         h_t2, c_t2)
        #     h_t3 = h_t
        #     c_t3 = c_t
        #     h_t4 = h_t2
        #     c_t4 = c_t2
        #     # encoder_vector = self.decoder(nn.LeakyReLU()(self.conv(h_t2)))
        #     for t in range(future_step):
        #         if t==0:
        #             h_t3, c_t3 = self.decoder_1_convlem(self.encoder(x[:,:,seq_len-1]),
        #                                          h_t3, c_t3) 
        #         else:
        #             h_t3, c_t3 = self.decoder_1_convlem(self.encoder(encoder_vector),
        #                                          h_t3, c_t3)  # we could concat to provide skip conn here
        #         h_t4, c_t4 = self.decoder_2_convlem(h_t3,
        #                                         h_t4, c_t4)
        #         output = self.conv(h_t4)
        #         encoder_vector = self.decoder(nn.LeakyReLU()(output))
        #         outputs[:,:,t] = self.decoder(nn.LeakyReLU()(output))
                
        #     return outputs       
        # # encoder
        # for t in range(seq_len):
        #     h_t, c_t = self.encoder_1_convlem(self.encoder(x[:,:,t]),
        #                                       h_t,c_t)  # we could concat to provide skip conn here
        #     h_t2, c_t2 = self.encoder_2_convlem(h_t,
        #                                         h_t2, c_t2)  # we could concat to provide skip conn here

        # # encoder_vector
        # # encoder_vector = h_t2.clone()

        # # decoder
        # for t in range(future_step):
        #     h_t3, c_t3 = self.decoder_1_convlem(encoder_vector,
        #                                          h_t3, c_t3)  # we could concat to provide skip conn here
        #     h_t4, c_t4 = self.decoder_2_convlem(h_t3,
        #                                         h_t4, c_t4)  # we could concat to provide skip conn here
        #     encoder_vector = h_t4.clone()
        #     output = self.conv(h_t4)
        #     outputs[:,:,t] = self.decoder(nn.LeakyReLU()(output))

        # return outputs

    def forward(self, x, future_seq, hidden_state=None):

        """
        Parameters
        ----------
        input_tensor:
            5-D Tensor of shape (b, t, c, h, w)        #   batch, time, channel, height, width
        """
            
        # find size of different input dimensions
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

        # initialize hidden states
        # h_t, c_t = self.encoder_1_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        # h_t2, c_t2 = self.encoder_2_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        # h_t3, c_t3 = self.decoder_1_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        # h_t4, c_t4 = self.decoder_2_convlstm.init_hidden(batch_size=b, image_size=(h, w))

        # autoencoder forward
        # x_norm = self.shiftmean(x, mode='sub',process=process)
        outputs = self.autoencoder(x, future_seq, h_t, c_t, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4)

        return outputs