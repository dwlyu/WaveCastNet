import torch
import torch.nn as nn
import torch.nn.init as init
import math
class tauGRU_Cell(nn.Module):
    def __init__(self, in_channels, out_channels, 
    kernel_size, padding, activation, dt, frame_size, alpha=1, beta=1):
        super(tauGRU_Cell, self).__init__()
        
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
            out_channels= 4 * out_channels, 
            kernel_size=kernel_size, 
            padding=padding) 
        
        self.convh = nn.Conv2d(
            in_channels=out_channels, 
            out_channels= 3 * out_channels, 
            kernel_size=kernel_size, 
            padding=padding)
        
        self.conv_hdelay = nn.Conv2d(
            in_channels=out_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            padding=padding)
        
        self.W_decay1 = nn.Parameter(torch.Tensor(out_channels, *frame_size))
        self.W_decay2 = nn.Parameter(torch.Tensor(out_channels, *frame_size))
        
    #     self.reset_parameters()

    # def reset_parameters(self):
    #     std = 1.0 / math.sqrt(self.nhid)
    #     for param in self.parameters():
    #         if len(param.shape) > 1:  # Initialize only the weight tensors, not bias
    #             init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')

    def forward(self, x, h, h_delay):
        # v1.3
        in_ux, in_zx, in_gx, in_ax = torch.chunk(self.convx(x), chunks=4, dim=1)
        in_uh, in_gh, in_ah = torch.chunk(self.convh(h), chunks=3, dim=1)
        in_zhdelay = self.conv_hdelay(h_delay)
        G = self.dt * torch.sigmoid(in_gx + in_gh + self.W_decay1 * h_delay)
        A = torch.sigmoid(in_ax + in_ah + self.W_decay2 * h_delay)
        U = self.activation(in_ux + in_uh) 
        Z = self.activation(in_zx + in_zhdelay)
        h = (1-G) * h + G * (self.beta * U + self.alpha * A * Z)      
        return h
    
class tauGRU(nn.Module): 
    
    def __init__(self, dt, tau, in_channels, out_channels, 
    kernel_size, padding, activation, frame_size):
        
        super(tauGRU, self).__init__()
        
        self.out_channels = out_channels
        self.tau = tau
        self.tauGRUcell = tauGRU_Cell(in_channels, out_channels, 
                                      kernel_size, padding, activation, dt, frame_size,)

    def forward(self, X):
        
        # X is a frame sequence (batch_size, num_channels, seq_len, height, width)
        input_device = X.device
        # Get the dimensions
        batch_size, _, seq_len, height, width = X.size()
        h_history = []

        # Initialize output
        output = torch.zeros(batch_size, self.out_channels, seq_len, 
        height, width).to(input_device)
        
        # Initialize Hidden State
        H = torch.zeros(batch_size, self.out_channels, 
        height, width).to(input_device)
        h_history.append(H)
        input_zero = torch.zeros_like(H)


        # Unroll over time steps
        for time_step in range(seq_len):
            if time_step < self.tau:
                
                H = self.tauGRUcell(X[:,:,time_step], H, input_zero)
                output[:,:,time_step] = H
                h_history.append(H)
                
            else:
                
                delta = (int)(time_step - self.tau)
                H= self.tauGRUcell(X[:,:,time_step], H, h_history[delta])
                output[:,:,time_step] = H
                h_history.append(H)
                
        return output

class Seq2Seq(nn.Module):

    def __init__(self, dt, tau, num_channels, num_kernels, kernel_size, padding, 
    activation, frame_size, num_layers):

        super(Seq2Seq, self).__init__()

        self.sequential = nn.Sequential()

        # Add First layer (Different in_channels than the rest)
        self.sequential.add_module(
            "tauGRU1", tauGRU(
                dt = dt, tau=tau,
                in_channels=num_channels, out_channels=num_kernels,
                kernel_size=kernel_size, padding=padding, 
                activation=activation, frame_size=frame_size)
        )

        self.sequential.add_module(
            "batchnorm1", nn.BatchNorm3d(num_features=num_kernels)
        ) 

        # Add rest of the layers
        for l in range(2, num_layers+1):

            self.sequential.add_module(
                f"tauGRU{l}", tauGRU(
                    dt = dt,  tau=tau,
                    in_channels=num_kernels, out_channels=num_kernels,
                    kernel_size=kernel_size, padding=padding, 
                    activation=activation, frame_size=frame_size)
                )
                
            self.sequential.add_module(
                f"batchnorm{l}", nn.BatchNorm3d(num_features=num_kernels)
                ) 

        # Add Convolutional Layer to predict output frame
        self.conv = nn.Conv2d(
            in_channels=num_kernels, out_channels=num_channels,
            kernel_size=kernel_size, padding=padding)

    def forward(self, X):

        # Forward propagation through all the layers
        output = self.sequential(X)

        # Return only the last output frame
        output = self.conv(output[:,:,-1])
        
        return nn.Sigmoid()(output)