# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright 2020 Ross Wightman
# Modified Model definition
# Updated: Incoperate Pyramid Structures

import torch
import torch.nn as nn
from functools import partial
import math
import warnings
import torch.nn.functional as F
import numpy as np

from vit_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from vit_utils import DropPath, to_2tuple, trunc_normal_,to_3tuple

from torch import einsum
from einops import rearrange, reduce, repeat

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'vit_base_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
}

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., with_qkv=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.with_qkv = with_qkv
        if self.with_qkv:
           self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
           self.proj = nn.Linear(dim, dim)
           self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        B, N, C = x.shape
        if self.with_qkv:
           qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
           q, k, v = qkv[0], qkv[1], qkv[2]
        else:
           qkv = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
           q, k, v  = qkv, qkv, qkv

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        if self.with_qkv:
           x = self.proj(x)
           x = self.proj_drop(x)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm, attention_type='divided_space_time'):
        super().__init__()
        self.attention_type = attention_type
        assert(attention_type in ['divided_space_time', 'space_only','joint_space_time'])

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
           dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        ## Temporal Attention Parameters
        if self.attention_type == 'divided_space_time':
            self.temporal_norm1 = norm_layer(dim)
            self.temporal_attn = Attention(
              dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
            self.temporal_fc = nn.Linear(dim, dim)

        ## drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)


    def forward(self, x, B, T, W):
        num_spatial_tokens = (x.size(1)) // T
        H = num_spatial_tokens // W

        if self.attention_type in ['space_only', 'joint_space_time']:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        elif self.attention_type == 'divided_space_time':
            
            ## Temporal
            xt = x[:,:]
            xt = rearrange(xt, 'b (h w t) m -> (b h w) t m',b=B,h=H,w=W,t=T)
            res_temporal = self.drop_path(self.temporal_attn(self.temporal_norm1(xt)))
            res_temporal = rearrange(res_temporal, '(b h w) t m -> b (h w t) m',b=B,h=H,w=W,t=T)
            res_temporal = self.temporal_fc(res_temporal)
            xt = x[:,:] + res_temporal

            ## Spatial
            xs = xt
            xs = rearrange(xs, 'b (h w t) m -> (b t) (h w) m',b=B,h=H,w=W,t=T)
            res_spatial = self.drop_path(self.attn(self.norm1(xs)))
            res_spatial = rearrange(res_spatial, '(b t) (h w) m -> b (h w t) m',b=B,h=H,w=W,t=T)
            res = res_spatial
            x = xt

            ## Mlp
            x = x  + res
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_3tuple(img_size)
        patch_size = to_3tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])*(img_size[2] // patch_size[2])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, T, H, W = x.shape
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if T % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - T % self.patch_size[0]))
        x = self.proj(x)
        T = x.size(2)
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        W = x.size(-1)
        x = x.flatten(2).transpose(1, 2)
        return x, T, W

class PatchMerging(nn.Module):
    """ Patch Merging Layer
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.linear = nn.Linear(4*dim, 2*dim, bias=False)
        self.norm = norm_layer(4*dim)

    def forward(self, x):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, D, H, W, C).
        """
        BT,C,H,W = x.shape
        pad_input = (x.shape[2] % 2 == 1) or (x.shape[3] % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, x.shape[3] % 2, 0, x.shape[2] % 2))
            H = x.shape[2]
            W = x.shape[3]
        x = torch.reshape(x, (BT,C,H//2,W//2, 2, 2))
        x = x.permute(0,2,3,1,4,5)
        x = torch.reshape(x,(BT,H//2,W//2,4*C))
        x = self.norm(x)
        x = self.linear(x)
        return x


class UpSample(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        '''Up-sampling operation'''
        # Linear layers without bias to increase channels of the data
        self.linear1 = nn.Linear(dim, dim*4, bias=False)
        # Linear layers without bias to mix the data up
        self.linear2 = nn.Linear(dim, dim//2, bias=False)
        # Normalization
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x):
        BT,C,H,W = x.shape
        x = x.permute(0,2,3,1) # x of shape (N,H,W,C)
        # Call the linear functions to increase channels of the data
        x = self.linear1(x)
        # Reshape x to facilitate upsampling.
        x = torch.reshape(x, (BT,H,W, 2, 2, x.shape[-1]//4))
        # Change the order of x
        x = x.permute(0,1,3,2,4,5)
        # Reshape to get Tensor with a resolution of (8, 360, 182)
        x = torch.reshape(x, (BT, 2*H, 2*W, x.shape[-1]))
        # Call the layer normalization
        x = self.norm(x)
        # Mixup normalized tensors
        x = self.linear2(x)
        return x

class VisionTransformer(nn.Module):
    """ Vision Transformere
    """
    def __init__(self, img_size=(60,80,56), patch_size=8, in_chans=3, embed_dim=192, depth=16,
                 num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm, num_frames=60, attention_type='divided_space_time', dropout=0.,
                 is_embed=False,is_position=False):
        super().__init__()
        self.attention_type = attention_type
        self.depth = depth
        self.is_embed = is_embed
        self.is_position = is_position
        self.num_frame = num_frames
        self.dropout = nn.Dropout(dropout)
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        if is_embed:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = (img_size[1]//patch_size[1]) * (img_size[2]//patch_size[2])
        ## Positional Embeddings
        if is_position:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            if self.attention_type != 'space_only':
                self.time_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
                self.time_drop = nn.Dropout(p=drop_rate)
        
        self.pos_drop = nn.Dropout(p=drop_rate)
                
        ## Attention Blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, attention_type=self.attention_type)
            for i in range(self.depth)])
        self.norm = norm_layer(embed_dim)
        if is_position:
            trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

        ## initialization of temporal attention weights
        if self.attention_type == 'divided_space_time':
            i = 0
            for m in self.blocks.modules():
                m_str = str(m)
                if 'Block' in m_str:
                    if i > 0:
                      nn.init.constant_(m.temporal_fc.weight, 0)
                      nn.init.constant_(m.temporal_fc.bias, 0)
                    i += 1

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'time_embed'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        if self.is_embed:
            B = x.shape[0]
            x, T, W = self.patch_embed(x)
            H = x.size(1) // W
        else:
            T  =  self.num_frame
            B = x.shape[0]//T
            W = x.shape[2]
            H = x.shape[1]
            x = rearrange(x, 'n h w c -> n (h w) c')

        # incorperating positional embeddings for embedding or reconstruction when necessary
        if self.is_position:
            ## resizing the positional embeddings in case they don't match the input at inference
            if x.size(1) != self.pos_embed.size(1):
                pos_embed = self.pos_embed
                H = self.pos_embed.size(1) // W
                other_pos_embed = pos_embed[0,:,:].unsqueeze(0).transpose(1, 2)
                P = int(other_pos_embed.size(2) ** 0.5)
                other_pos_embed = other_pos_embed.reshape(1, x.size(2), H, W)
                H = x.size(1) // W
                new_pos_embed = F.interpolate(other_pos_embed, size=(H, W), mode='nearest')
                new_pos_embed = new_pos_embed.flatten(2)
                new_pos_embed = new_pos_embed.transpose(1, 2)
                x = x + new_pos_embed
            else:
                x = x + self.pos_embed
            x = self.pos_drop(x)


            ## Time Embeddings
            if self.attention_type != 'space_only':
                x = x[:,0:]
                x = rearrange(x, '(b t) n m -> (b n) t m',b=B,t=T)
                ## Resizing time embeddings in case they don't match
                if T != self.time_embed.size(1):
                    time_embed = self.time_embed.transpose(1, 2)
                    new_time_embed = F.interpolate(time_embed, size=(T), mode='nearest')
                    new_time_embed = new_time_embed.transpose(1, 2)
                    x = x + new_time_embed
                else:
                    x = x + self.time_embed
                x = self.time_drop(x)
                x = rearrange(x, '(b n) t m -> b (n t) m',b=B,t=T)
        
        else:
            x = self.pos_drop(x)
            x = rearrange(x, '(b t) n m -> b (n t) m',b=B,t=T)
        ## Attention blocks
        for blk in self.blocks:
            x = blk(x, B, T, W)

        ### Predictions for space-only baseline
        if self.attention_type == 'space_only':
            x = rearrange(x, '(b t) n m -> b t n m',b=B,t=T)
            x = torch.mean(x, 1) # averaging predictions for every frame
        # x = self.norm(x)
        x = rearrange(x, 'b (h w t) m -> (b t) m h w', h=H,w=W,t=T)
        return x
    
    def forward(self, x):
        x = self.forward_features(x)
        return x


class Timesformer_eq(nn.Module):
    def __init__(self,
                 img_size=(60,80,56),
                 patch_size=(1,8,8),
                 in_chans=3,
                 embed_dim=192,
                 depths=[2,6,6,2],
                 num_heads=[4,12,12,4],
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 num_frames=60,
                 attention_type='divided_space_time'):
        super().__init__()

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.num_frames = num_frames

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        
        self.layer1 = VisionTransformer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, depth=depths[0],
                 num_heads=num_heads[0], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                 drop_path_rate=drop_path_rate, norm_layer=nn.LayerNorm, num_frames=num_frames, attention_type=attention_type,
                 is_embed=True,is_position=True)
        
        self.layer2 = VisionTransformer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim= 2 * embed_dim, depth=depths[1],
                 num_heads=num_heads[1], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                 drop_path_rate=drop_path_rate, norm_layer=nn.LayerNorm, num_frames=num_frames, attention_type=attention_type,
                 is_embed=False,is_position=False)
        
        self.layer3 = VisionTransformer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim= 2 * embed_dim, depth=depths[2],
                 num_heads=num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                 drop_path_rate=drop_path_rate, norm_layer=nn.LayerNorm, num_frames=num_frames, attention_type=attention_type,
                 is_embed=False,is_position=False)
        
        self.layer4 = VisionTransformer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim= embed_dim, depth=depths[3],
                 num_heads=num_heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                 drop_path_rate=drop_path_rate, norm_layer=nn.LayerNorm, num_frames=num_frames, attention_type=attention_type,
                 is_embed=False,is_position=True)
        
        self.merge = PatchMerging(dim = int(embed_dim))
        self.upsample = UpSample(dim = 2 * int(embed_dim))
        self.num_features = embed_dim

        # add a norm layer for each output
        self.norm = norm_layer(self.num_features)
        self.reverse = nn.ConvTranspose3d(2 * embed_dim, in_chans, kernel_size=patch_size, stride=patch_size)     
        self.init_weights()

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        """
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)         
            self.apply(_init_weights)

    def forward(self, x):
        """Forward function."""
        b,C,origin_d, origin_h, origin_w = x.shape
        x = self.layer1(x.contiguous())
        
        skip = x.clone() # skip connection
        
        pad_input = (skip.shape[2] % 2 == 1) or (skip.shape[3] % 2 == 1)
        if pad_input:
            skip = F.pad(skip, (0, skip.shape[3] % 2, 0, skip.shape[2] % 2))
        
        x = self.merge(x) # patch merging
        
        x = self.layer2(x.contiguous())
        x = rearrange(x, '(bt) m h w -> (bt) h w m')
        x = self.layer3(x.contiguous())
        
        x = self.upsample(x) # upsampling
        x = self.layer4(x.contiguous())
        
        x = torch.concat((x,skip),dim=1) # residual connection
        x = rearrange(x, '(b t) m h w -> b m t h w',t=self.num_frames)
        
        x = self.reverse(x)
        return x[:,:,:origin_d,:origin_h,:origin_w]

# @MODEL_REGISTRY.register()
# class vit_base_patch16_224(nn.Module):
#     def __init__(self, cfg, **kwargs):
#         super(vit_base_patch16_224, self).__init__()
#         self.pretrained=True
#         patch_size = 16
#         self.model = VisionTransformer(img_size=cfg.DATA.TRAIN_CROP_SIZE, num_classes=cfg.MODEL.NUM_CLASSES, patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, num_frames=cfg.DATA.NUM_FRAMES, attention_type=cfg.TIMESFORMER.ATTENTION_TYPE, **kwargs)

#         self.attention_type = cfg.TIMESFORMER.ATTENTION_TYPE
#         self.model.default_cfg = default_cfgs['vit_base_patch16_224']
#         self.num_patches = (cfg.DATA.TRAIN_CROP_SIZE // patch_size) * (cfg.DATA.TRAIN_CROP_SIZE // patch_size)
#         pretrained_model=cfg.TIMESFORMER.PRETRAINED_MODEL
#         if self.pretrained:
#             load_pretrained(self.model, num_classes=self.model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter, img_size=cfg.DATA.TRAIN_CROP_SIZE, num_patches=self.num_patches, attention_type=self.attention_type, pretrained_model=pretrained_model)

#     def forward(self, x):
#         x = self.model(x)
#         return x

# @MODEL_REGISTRY.register()
# class TimeSformer(nn.Module):
#     def __init__(self, img_size=224, patch_size=16, num_classes=400, num_frames=8, attention_type='divided_space_time',  pretrained_model='', **kwargs):
#         super(TimeSformer, self).__init__()
#         self.pretrained=True
#         self.model = VisionTransformer(img_size=img_size, num_classes=num_classes, patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, num_frames=num_frames, attention_type=attention_type, **kwargs)

#         self.attention_type = attention_type
#         self.model.default_cfg = default_cfgs['vit_base_patch'+str(patch_size)+'_224']
#         self.num_patches = (img_size // patch_size) * (img_size // patch_size)
#         if self.pretrained:
#             load_pretrained(self.model, num_classes=self.model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter, img_size=img_size, num_frames=num_frames, num_patches=self.num_patches, attention_type=self.attention_type, pretrained_model=pretrained_model)
#     def forward(self, x):
#         x = self.model(x)
#         return x