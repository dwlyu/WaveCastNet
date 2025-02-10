import torch
import torch.nn as nn
import numpy as np
from .ConvLEMCell import ConvLEMCell_1
from .EncoderDecoder import decoder2_48
from .EncodingSparse import Encoder1d

# Seq2Seq(ConvLEM) for sparse sampling scenarios

class AEConvLEM_sparse(nn.Module):
    def __init__(self, dt, num_channels, num_kernels, kernel_size, padding, 
    activation, frame_size, mask_mode=False, mask_ratio=0, return_dt=False):
        super(AEConvLEM_sparse, self).__init__()

        """ ARCHITECTURE 

        # Encoder (ConvLEM)
        # Encoder Vector (final hidden state of encoder)
        # Decoder (ConvLEM) - takes Encoder Vector as input
        # Decoder (CNN) - produces regression predictions for our model

        """
        self.out_channels = num_kernels
        self.returndt = return_dt #for producing dt and dt_bar histogram visualization 
        
        self.encoder = Encoder1d(mask_mode, mask_ratio) # Attaining and encoding sparse input
        self.decoder = decoder2_48(num_channel = num_channels)
        
        self.encoder_1_convlem = ConvLEMCell_1(dt = dt,
                in_channels=num_kernels, out_channels=num_kernels,
                kernel_size=kernel_size, padding=padding, 
                activation=activation, frame_size=frame_size,return_dt=return_dt)

        self.encoder_2_convlem = ConvLEMCell_1(dt = dt,
                in_channels=num_kernels, out_channels=num_kernels,
                kernel_size=kernel_size, padding=padding, 
                activation=activation, frame_size=frame_size,return_dt=return_dt)

        self.decoder_1_convlem = ConvLEMCell_1(dt = dt,
                in_channels=num_kernels, out_channels=num_kernels,
                kernel_size=kernel_size, padding=padding, 
                activation=activation, frame_size=frame_size,return_dt=return_dt)

        self.decoder_2_convlem = ConvLEMCell_1(dt = dt,
                in_channels=num_kernels, out_channels=num_kernels,
                kernel_size=kernel_size, padding=padding, 
                activation=activation, frame_size=frame_size,return_dt=return_dt)
        
        self.conv1 = nn.Conv2d(
            in_channels=3 * num_channels, out_channels= num_channels,
            kernel_size=kernel_size, padding=padding)
        
        self.mask_mode = mask_mode
        


    def autoencoder(self, x, future_step, h_t, c_t, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4):

        batch_size, _, seq_len, height, width = x.size()
        outputs = torch.zeros(batch_size,  _, future_step, 
        height, width).to(x.device)
        
        dt_all = []
        dt_bar_all = []
        if self.mask_mode:
            mask_sample = len(np.load('/global/homes/d/dwlyu/earthquake/filtered_coord.npy'))
            space_only_mask = torch.rand(batch_size, mask_sample).to(x.device)
        for t in range(seq_len):

            step_input = x[:,:,t]

            if self.mask_mode:
                    encode = self.encoder(step_input,space_only_mask = space_only_mask)
            else:
                    encode = self.encoder(step_input)

            if self.returndt:
                h_t, c_t, dt, dt_bar = self.encoder_1_convlem(encode,
                                              h_t,c_t)  # we could concat to provide skip conn here
                h_t2, c_t2, dt2, dt_bar_2 = self.encoder_2_convlem(h_t,
                                                h_t2, c_t2)
                dt_all.append(dt)
                # dt_all.append(dt2)
                dt_bar_all.append(dt_bar)
                # dt_bar_all.append(dt_bar_2)
            else:
                h_t, c_t = self.encoder_1_convlem(encode,
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
            output = h_t4
            encoder_vector = h_t4
            outputs[:,:,t] = self.conv1(self.decoder(output))
                
        if self.returndt:
            if len(dt_all)>0:
                return outputs, torch.cat(dt_all,dim=0),torch.cat(dt_bar_all,dim=0)
            else:
                return outputs, [] , []
        else:
            return outputs
    def forward(self, x, future_seq, hidden_state=None):

        """
        Parameters
        ----------
        input_tensor:
            5-D Tensor of shape (b, c, t, h, w)        # batch, channel, time, height, width
        """
            
        # find size of different input dimensions
        input_device = x.device
        batch_size,_,seq_len, height, width = x.size()
        height = (int)(height/8) # spatial scaling factor for encoded latent space is 8
        width = (int)(width/8)
        
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

