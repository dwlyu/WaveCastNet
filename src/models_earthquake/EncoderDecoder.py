import torch
import torch.nn as nn
import torch.nn.init as init

class encoder_layer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(encoder_layer, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size =(4,4), stride =(2,2), padding = (1,1), bias = True),
            # init.kaiming_normal_(self.conv.weight, mode='fan_in', nonlinearity='leaky_relu'),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU()
        )
    def forward(self, x):
        return self.layer(x)


class encoderBlock(nn.Module):
    def __init__(self, num_channel, num_layer):
        super(encoderBlock, self).__init__()
        #https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html example 
        self.model = nn.Sequential()
        for i in range(1,num_layer+1):
            self.model.add_module(f"encoder_layer{i}", encoder_layer(num_channel, 2 * num_channel))
            num_channel*=2
    def forward(self,x):
        out = self.model(x)
        return out
    
class encoder1(nn.Module):
    def __init__(self, num_channel):
        super(encoder1, self).__init__()
        # Dense Encoder
        self.model = nn.Sequential()
        self.model.add_module(f"encoder_layer{1}", encoder_layer(num_channel, 12* num_channel))
        self.model.add_module(f"encoder_layer{2}", encoder_layer(12* num_channel, 24* num_channel))
        self.model.add_module(f"encoder_layer{3}", encoder_layer(24*num_channel, 48* num_channel))
    def forward(self,x):
        out = self.model(x)
        return out

class encoder2(nn.Module):
    def __init__(self, num_channel):
        super(encoder2, self).__init__()
        # Ablation Encoder
        self.model = nn.Sequential()
        self.model.add_module(f"encoder_layer{1}", encoder_layer(num_channel, 24* num_channel))
        self.model.add_module(f"encoder_layer{2}", encoder_layer(24*num_channel, 48* num_channel))
    def forward(self,x):
        out = self.model(x)
        return out
    
class encoder_sparser(nn.Module):
    def __init__(self, num_channel):
        super(encoder_sparser, self).__init__()
        # Sparse Encoder for reshaped input
        self.model = nn.Sequential(
            nn.Conv2d(num_channel, 24 *num_channel, kernel_size =(3,3), padding = (1,1), bias = True),
            nn.BatchNorm2d(24 * num_channel),
            nn.LeakyReLU()
            )
        self.model.add_module(f"encoder_layer{2}", encoder_layer(24*num_channel, 48* num_channel))
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
        #https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html example 
        self.model = nn.Sequential()
        for i in range(1,num_layer+1):
            self.model.add_module(f"decoder_layer{i}", decoder_layer(num_channel, (int)(num_channel/2)))
            num_channel = (int)(num_channel/2)
    def forward(self, x):
        out = self.model(x)
        return out
    
class decoder3(nn.Module):
    def __init__(self, num_channel):
        super(decoder3, self).__init__()
        # Ablation Decoder
        self.model = nn.Sequential()
        self.model.add_module(f"decoder_layer{2}", nn.PixelShuffle(4))
    def forward(self,x):
        out = self.model(x)
        return out

    
class decoder2_48(nn.Module):
    def __init__(self, num_channel):
        super(decoder2_48, self).__init__()
        # Decoder for both dense and sparse scenarios
        self.model = nn.Sequential()
        self.model.add_module(f"decoder_layer{1}", decoder_layer(48*num_channel, 48* num_channel))
        self.model.add_module(f"decoder_layer{2}", nn.PixelShuffle(4))
    def forward(self,x):
        out = self.model(x)
        return out
    
