# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""
Projected discriminator architecture from
"StyleGAN-T: Unlocking the Power of GANs for Fast Large-Scale Text-to-Image Synthesis".
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.spectral_norm import SpectralNorm
from torchvision.transforms import RandomCrop, Normalize
import timm
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import sys 
import ADD.th_utils.misc as misc
from ADD.models.shared import ResidualBlock, FullyConnectedLayer
from ADD.models.vit_utils import make_vit_backbone, forward_vit
from ADD.models.DiffAugment import DiffAugment
from ADD.utils.util_net import reload_model_

from .basic_ops import (
    linear,
    conv_nd,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)

class SpectralConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        SpectralNorm.apply(self, name='weight', n_power_iterations=1, dim=0, eps=1e-12)


class BatchNormLocal(nn.Module):
    def __init__(self, num_features: int, affine: bool = True, virtual_bs: int = 3, eps: float = 1e-5):
        super().__init__()
        self.virtual_bs = virtual_bs
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.size()

        # Reshape batch into groups.
        G = np.ceil(x.size(0)/self.virtual_bs).astype(int)
        x = x.view(G, -1, x.size(-2), x.size(-1))

        # Calculate stats.
        mean = x.mean([1, 3], keepdim=True)
        var = x.var([1, 3], keepdim=True, unbiased=False)
        x = (x - mean) / (torch.sqrt(var + self.eps))

        if self.affine:
            x = x * self.weight[None, :, None] + self.bias[None, :, None]

        return x.view(shape)


def make_block(channels: int, kernel_size: int) -> nn.Module:
    return nn.Sequential(
        SpectralConv1d(
            channels,
            channels,
            kernel_size = kernel_size,
            padding = kernel_size//2,
            padding_mode = 'circular',
        ),
        #BatchNormLocal(channels),
        nn.GroupNorm(4, channels),
        nn.LeakyReLU(0.2, True),
    )


class DiscHead(nn.Module):
    def __init__(self, channels: int, c_dim: int):
        super().__init__()
        self.channels = channels
        self.c_dim = c_dim
        

        self.main = nn.Sequential(
            make_block(channels, kernel_size=1),
            ResidualBlock(make_block(channels, kernel_size=9))
        )

        if self.c_dim > 0:
            self.cls = SpectralConv1d(channels, self.c_dim, kernel_size=1, padding=0)
        else:
            self.cls = SpectralConv1d(channels, 1, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.main(x)
        out = self.cls(h)
        return out


class DINO(torch.nn.Module):
    def __init__(self, hooks, hook_patch: bool = True):
        super().__init__()
        self.n_hooks = len(hooks) + int(hook_patch)

        self.model = make_vit_backbone(
            timm.create_model('vit_small_patch16_224_dino', pretrained=False),
            patch_size=[16,16], hooks=hooks, hook_patch=hook_patch,
        )
        reload_model_(self.model, torch.load('./preset/models/dino_deitsmall16_pretrain.pth'))
        self.model = self.model.eval().requires_grad_(False)


        self.img_resolution = self.model.model.patch_embed.img_size[0]
        self.embed_dim = self.model.model.embed_dim
        self.norm = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ''' input: x in [0, 1]; output: dict of activations '''
        x = F.interpolate(x, self.img_resolution, mode='area')
        x = self.norm(x)
        features = forward_vit(self.model, x)
        return features


class ProjectedDiscriminator(nn.Module):
    def __init__(self, c_dim: int, diffaug: bool = True, p_crop: float = 0.5):
        super().__init__()
        self.c_dim = c_dim
        self.diffaug = diffaug
        self.p_crop = p_crop

        self.dino = DINO()

        heads = []
        for i in range(self.dino.n_hooks):
            heads += [str(i), DiscHead(self.dino.embed_dim, c_dim)],
        self.heads = nn.ModuleDict(heads)

    def train(self, mode: bool = True):
        self.dino = self.dino.train(False)
        self.heads = self.heads.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # Apply augmentation (x in [-1, 1]).
        if self.diffaug:
            x = DiffAugment(x, policy='translation,cutout')

        # Transform to [0, 1].
        x = x.add(1).div(2)

        # Take crops with probablity p_crop if the image is larger.
        if x.size(-1) > self.dino.img_resolution and np.random.random() < self.p_crop:
            x = RandomCrop(self.dino.img_resolution)(x)

        # Forward pass through DINO ViT.
        features = self.dino(x)

        # Apply discriminator heads.
        logits = []
        for k, head in self.heads.items():
            features[k].requires_grad_(True)
            logits.append(head(features[k], c).view(x.size(0), -1))
        #logits = torch.cat(logits, dim=1)

        return logits, features
    
class Time_Aware_Discriminator(nn.Module):
    def __init__(self, c_dim: int, embed_dim: int):
        super().__init__()
        self.c_dim = c_dim
        self.embed_dim = embed_dim
        heads = []
        heads += [str(0), DiscHead(self.embed_dim, c_dim)],
        heads += [str(1), DiscHead(self.embed_dim*2, c_dim)],
        heads += [str(2), DiscHead(self.embed_dim*4, c_dim)],
        heads += [str(3), DiscHead(self.embed_dim*4, c_dim)],
        self.heads = nn.ModuleDict(heads)

        self.emb_layers = nn.ModuleList()
        self.emb_layers.append(nn.Sequential(nn.SiLU(),linear(self.embed_dim,2 * self.embed_dim),))
        self.emb_layers.append(nn.Sequential(nn.SiLU(),linear(self.embed_dim,4 * self.embed_dim),))
        self.emb_layers.append(nn.Sequential(nn.SiLU(),linear(self.embed_dim,8 * self.embed_dim),))
        self.emb_layers.append(nn.Sequential(nn.SiLU(),linear(self.embed_dim,8 * self.embed_dim),))


        self.out_layers = nn.ModuleList()
        self.out_layers.append(nn.Sequential(normalization(self.embed_dim),nn.SiLU(),nn.Dropout(p=0),zero_module(conv_nd(2, self.embed_dim, self.embed_dim, 3, padding=1)),)) 
        self.out_layers.append(nn.Sequential(normalization(self.embed_dim*2),nn.SiLU(),nn.Dropout(p=0),zero_module(conv_nd(2, self.embed_dim*2, self.embed_dim*2, 3, padding=1)),))
        self.out_layers.append(nn.Sequential(normalization(self.embed_dim*4),nn.SiLU(),nn.Dropout(p=0),zero_module(conv_nd(2, self.embed_dim*4, self.embed_dim*4, 3, padding=1)),))
        self.out_layers.append(nn.Sequential(normalization(self.embed_dim*4),nn.SiLU(),nn.Dropout(p=0),zero_module(conv_nd(2, self.embed_dim*4, self.embed_dim*4, 3, padding=1)),))
        


    def train(self, mode: bool = True):
        self.heads = self.heads.train(mode)
        self.emb_layers = self.emb_layers.train(mode)
        self.out_layers = self.out_layers.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def forward(self, x: torch.Tensor,t,detach=False) -> torch.Tensor:

        # Apply discriminator heads.
        logits = []
        emb = timestep_embedding(t,self.embed_dim)
        for k, head in self.heads.items():
            emb_out = self.emb_layers[int(k)](emb)
            while len(emb_out.shape) < len(x[int(k)].shape):
                emb_out = emb_out[..., None]
            out_norm, out_rest = self.out_layers[int(k)][0], self.out_layers[int(k)][1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            if detach:
                h = out_norm(x[int(k)].detach()) * (1 + scale) + shift
            else:
                h = out_norm(x[int(k)]) * (1 + scale) + shift
            h = out_rest(h)
            input = torch.flatten(h,start_dim=2)
            logits.append(head(input).view(x[int(k)].size(0), -1))

        return logits
