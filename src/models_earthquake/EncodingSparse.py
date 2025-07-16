import torch
import math
from torch import nn
from torch.nn import functional as F
from EncoderDecoder import encoder_sparser
import numpy as np

class SelfAttention(nn.Module):
    def __init__(self, n_heads, d_embed, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x, causal_mask=False):
        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape
        interim_shape = (batch_size, sequence_length, self.n_heads, self.d_head)

        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        weight = q @ k.transpose(-1, -2)
        if causal_mask:
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)
        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)

        output = weight @ v
        output = output.transpose(1, 2)
        output = output.reshape(input_shape)
        output = self.out_proj(output)
        return output
    
class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(16, channels)
        self.attention = SelfAttention(1, channels)
    
    def forward(self, x):
        residue = x
        x = self.groupnorm(x)

        n, c, h, w = x.shape
        x = x.view((n, c, h * w))
        x = x.transpose(-1, -2)
        x = self.attention(x)
        x = x.transpose(-1, -2)
        x = x.view((n, c, h, w))

        x += residue
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(24, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(24, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.activation = nn.LeakyReLU()

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    
    def forward(self, x):
        residue = x

        x = self.groupnorm_1(x)
        x = self.activation(x)
        x = self.conv_1(x)

        x = self.groupnorm_2(x)
        x = self.activation(x)
        x = self.conv_2(x)

        return x + self.residual_layer(residue)

class Attention1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(3, channels)
        self.attention = SelfAttention(1, channels)
    
    def forward(self, x):
        residue = x
        x = self.groupnorm(x)
        n, c, l = x.shape
        x = x.transpose(-1, -2)
        x = self.attention(x)
        x = x.transpose(-1, -2)

        x += residue
        return x
    
class Encoder1d(nn.Module):
    def __init__(self, mask_mode=False, mask_ratio=0.3):
        super(Encoder1d,self).__init__()
        # self.Attention=Attention1D(3)
        self.mask_mode = mask_mode
        self.mask_ratio = mask_ratio
        self.input_len = len(np.load('/global/homes/d/dwlyu/earthquake/filtered_coord.npy'))
        self.sample_list = np.load('/global/homes/d/dwlyu/earthquake/filtered_coord.npy')
        self.FC1=nn.Sequential(nn.Linear(self.input_len, 1204),
                            nn.LeakyReLU(), 
                            nn.BatchNorm1d(3))
        
        self.FC2=nn.Sequential(nn.Linear(1204, 4816),
                               nn.LeakyReLU(), 
                               nn.BatchNorm1d(3))
        self.encoder = encoder_sparser(num_channel = 3)
        # self.Attention=Attention1D(3)
        
    def mask(self,input,space_only_mask = None):
        if self.mask_mode:
            mask = space_only_mask.unsqueeze(1).expand_as(input) < self.mask_ratio
            input = input.masked_fill(mask, 0)
            return input
        else:
            return input
            
    
    def irr_sample(self,input):
        inputlist = []
        for i in range(self.sample_list.shape[0]):
            inputlist.append(input[:,:,self.sample_list[i,0],self.sample_list[i,1]])
        final_input = torch.stack(inputlist,dim=2)
        return final_input

    def forward(self, x, space_only_mask = None, train=True):
        n,c,h,w = x.shape
        sampled_x = self.irr_sample(x)
        if train:
            if self.mask_mode:
                sampled_x = self.mask(sampled_x,space_only_mask)
            else:
                sampled_x = self.Attention(sampled_x)
        out = self.FC1(sampled_x)
        out = self.FC2(out)
        out = out.view((n,3,86,56))
        out = self.encoder(out)
        return out