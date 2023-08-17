import torch
import torch.nn as nn
import torch.nn.init as init

class encoder_layer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(encoder_layer, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size =(4,4), stride =(2,2), padding = (1,1), bias = True),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU()
        )
    def forward(self, x):
        return self.layer(x)


class encoderBlock(nn.Module):
    def __init__(self, num_channel, num_layer):
        super(encoderBlock, self).__init__()
        self.model = nn.Sequential()
        for i in range(1,num_layer+1):
            self.model.add_module(f"encoder_layer{i}", encoder_layer(num_channel, 2 * num_channel))
            num_channel*=2
    def forward(self,x):
        out = self.model(x)
        return out


class decoder_layer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(decoder_layer, self).__init__()
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim, kernel_size = 4, stride = 2,padding = 1, bias = True),
            nn.LeakyReLU()
        )
    def forward(self, x):
        return self.layer(x)
    

class decoderBlock(nn.Module):
    def __init__(self, num_channel, num_layer):
        super(decoderBlock, self).__init__()
        self.model = nn.Sequential()
        for i in range(1,num_layer+1):
            self.model.add_module(f"decoder_layer{i}", decoder_layer(num_channel, (int)(num_channel/2)))
            num_channel = (int)(num_channel/2)
    def forward(self, x):
        out = self.model(x)
        return out
    
