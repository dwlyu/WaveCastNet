import torch
import torch.nn as nn
import torch.nn.init as init
from .EncoderDecoder import encoder2
from .EncoderDecoder import decoder3

class ConvGRU_Cell(nn.Module):
    def __init__(self, in_channels, out_channels, 
    kernel_size, padding, activation, dt, frame_size, alpha=1, beta=1):
        super(ConvGRU_Cell, self).__init__()
        
        if activation == "tanh":
            self.activation = torch.tanh 
        elif activation == "relu":
            self.activation = torch.relu
            
        self.ninp = in_channels
        self.nhid = out_channels
        self.alpha = alpha
        self.beta = beta
        self.dt = dt
        
        self.convx = nn.Conv2d(
            in_channels=in_channels, 
            out_channels= 3 * out_channels, 
            kernel_size=kernel_size, 
            padding=padding) 
        
        self.convh = nn.Conv2d(
            in_channels=out_channels, 
            out_channels= 2 * out_channels, 
            kernel_size=kernel_size, 
            padding=padding)
        
        self.conv_hreset = nn.Conv2d(
            in_channels=out_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            padding=padding)
        
        self.W_1 = nn.Parameter(torch.Tensor(out_channels, *frame_size))
        self.W_2 = nn.Parameter(torch.Tensor(out_channels, *frame_size))
        
        self.reset_parameters(out_channels)

    def reset_parameters(self, out_channels):
            for param in self.parameters():
                if len(param.shape) > 1:  # Initialize only the weight tensors, not bias
                    init.xavier_uniform_(param)   
                else:
                    nn.init.constant_(param, 0)

    def forward(self, x, h):
        in_ux, in_gx, in_ax = torch.chunk(self.convx(x), chunks=3, dim=1)
        in_gh, in_ah = torch.chunk(self.convh(h), chunks=2, dim=1)
        G = torch.sigmoid(in_gx + in_gh + self.W_1 * h)
        A = torch.sigmoid(in_ax + in_ah + self.W_2 * h)
        in_uh = self.conv_hreset(A * h)
        U = self.activation(in_ux + in_uh) 
        h = (1-G) * h + G * U
        return h

class AEConvGRU(nn.Module):
    def __init__(self, dt, num_channels, num_kernels, kernel_size, padding, 
    activation, frame_size):
        super(AEConvGRU, self).__init__()
        
        #Seq2Seq(ConvGRU) for Ablation study on (H/4, W/4)
        
        """ ARCHITECTURE 

        # Encoder (ConvGRU)
        # Encoder Vector (final hidden state of encoder)
        # Decoder (ConvGRU) - takes Encoder Vector as input
        # Decoder (CNN) - produces regression predictions for our model

        """
        self.out_channels = num_kernels
        self.encoder = encoder2(num_channel = num_channels)
        self.decoder = decoder3(num_channel = num_channels)
        
        self.encoder_1_gru = ConvGRU_Cell(dt = dt,
                in_channels=144, out_channels=num_kernels,
                kernel_size=kernel_size, padding=padding, 
                activation=activation, frame_size=frame_size)

        self.encoder_2_gru = ConvGRU_Cell(dt = dt,
                in_channels=num_kernels, out_channels=num_kernels,
                kernel_size=kernel_size, padding=padding, 
                activation=activation, frame_size=frame_size)

        self.decoder_1_gru = ConvGRU_Cell(dt = dt,
                in_channels=num_kernels, out_channels=num_kernels,
                kernel_size=kernel_size, padding=padding, 
                activation=activation, frame_size=frame_size)

        self.decoder_2_gru = ConvGRU_Cell(dt = dt,
                in_channels=num_kernels, out_channels=num_kernels,
                kernel_size=kernel_size, padding=padding, 
                activation=activation, frame_size=frame_size)
        
        self.conv1 = nn.Conv2d(
            in_channels=3 * num_channels, out_channels= num_channels,
            kernel_size=kernel_size, padding=padding)


    def autoencoder(self, x, future_step, h_t, h_t2, h_t3, h_t4):

        batch_size, _, seq_len, height, width = x.size()
        outputs = torch.zeros(batch_size, _, future_step, 
        height, width).to(x.device)
        for t in range(seq_len):
            h_t = self.encoder_1_gru(self.encoder(x[:,:,t]),
                                              h_t)
            h_t2 = self.encoder_2_gru(h_t,h_t2)
        h_t3 = h_t
        h_t4 = h_t2
        encoder_vector = torch.rand_like(h_t)
        for t in range(future_step):
            h_t3= self.decoder_1_gru(encoder_vector,
                                        h_t3)  # we could concat to provide skip conn here
            h_t4= self.decoder_2_gru(h_t3,
                                        h_t4)
            output = h_t4
            encoder_vector = h_t4
            outputs[:,:,t] = self.conv1(self.decoder(output))
                
        return outputs
    def forward(self, x, future_seq, hidden_state=None):

        """
        Parameters
        ----------
        input_tensor:
            5-D Tensor of shape (b, c, t, h, w)        # batch, channel, time, height, width
        """
        input_device = x.device
        batch_size,_,seq_len, height, width = x.size()
        height = (int)(height/4)
        width = (int)(width/4)
        
        # Initialize Hidden State
        h_t = torch.zeros(batch_size, self.out_channels,
        height, width).to(input_device)
        
        h_t2 = torch.zeros(batch_size, self.out_channels, 
        height, width).to(input_device)
        
        h_t3 = torch.zeros(batch_size, self.out_channels, 
        height, width).to(input_device)
        
        h_t4 = torch.zeros(batch_size, self.out_channels,
        height, width).to(input_device)
        
        outputs = self.autoencoder(x, future_seq, h_t, h_t2, h_t3, h_t4)

        return outputs