import torch
import torch.nn as nn
import torch.nn.init as init

class ConvLEMCell(nn.Module):
    def __init__(self, dt, in_channels, out_channels, 
    kernel_size, padding, activation, frame_size):
        super(ConvLEMCell, self).__init__()
        if activation == "tanh":
            self.activation = torch.tanh 
        elif activation == "relu":
            self.activation = torch.relu
        
        self.convx = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=4 * out_channels, 
            kernel_size=kernel_size, 
            padding=padding) 
        
        self.convy = nn.Conv2d(
            in_channels=out_channels, 
            out_channels=3 * out_channels, 
            kernel_size=kernel_size, 
            padding=padding) 
        
        self.convz = nn.Conv2d(
            in_channels=out_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            padding=padding) 
        
        self.dt = dt
        self.W_z1 = nn.Parameter(torch.Tensor(out_channels, *frame_size))
        self.W_z2 = nn.Parameter(torch.Tensor(out_channels, *frame_size))
        self.dropout = nn.Dropout(p=0.4)
        self.reset_parameters(out_channels)
        
    def reset_parameters(self, out_channels):
        for param in self.parameters():
            if len(param.shape) > 1:  # Initialize only the weight tensors, not bias
                init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')   

    def forward(self, x, y, z, isdrop=False):
        transformed_inp = self.convx(x)
        transformed_hid = self.convy(y)
        i_dt1, i_dt2, i_z, i_y = torch.chunk(transformed_inp, chunks=4, dim=1)
        h_dt1, h_dt2, h_y = torch.chunk(transformed_hid, chunks=3, dim=1)

        ms_dt = self.dt * torch.sigmoid(i_dt2 + h_dt2 + self.W_z2 * z)

        z = (1.-ms_dt) * z + ms_dt * self.activation(i_y + h_y)
        
    
        transformed_z = self.convz(z)
       
        if isdrop:
            ms_dt_bar = self.dropout(self.dt * torch.sigmoid(i_dt1 + h_dt1 + self.W_z1 * z))
        else:
            ms_dt_bar = self.dt * torch.sigmoid(i_dt1 + h_dt1 + self.W_z1 * z)
        # print(ms_dt_bar)
        
        y = (1.-ms_dt_bar)* y + ms_dt_bar * self.activation(transformed_z+i_z)
        
        return y, z