# Part of the implementation is borrowed and modified from stable-diffusion,
# publicly avaialbe at https://github.com/Stability-AI/stablediffusion
# under CreativeML Open RAIL-M license.
# and from The Alibaba Fundamental Vision Team Authors
# publicly avaialbe at https://github.com/modelscope/modelscope/tree/master/modelscope/models/multi_modal/video_synthesis
# under Apache 2.0 license
# -------------
# This project aims to implement NUWA-XL Diffusion over Diffusion laid out by Microsoft in https://arxiv.org/pdf/2303.12346.pdf
# ---
# The *unofficial implementation* conducted by the Deforum-art organization under the supervision of kabachuha
# Apache 2.0 license

import importlib
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import open_clip
from os import path as osp

__all__ = ['UNetSD']

try:
    import gc
    import torch

    def torch_gc():
        """Performs garbage collection for both Python and PyTorch CUDA tensors.

        This function collects Python garbage and clears the PyTorch CUDA cache
        and IPC (Inter-Process Communication) resources.
        """
        gc.collect()  # Collect Python garbage
        torch.cuda.empty_cache()  # Clear PyTorch CUDA cache
        torch.cuda.ipc_collect()  # Clear PyTorch CUDA IPC resources

except:

    def torch_gc():
        """Dummy function when torch is not available.

        This function does nothing and serves as a placeholder when torch is
        not available, allowing the rest of the code to run without errors.
        """
        pass

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def sinusoidal_embedding(timesteps, dim):
    # check input
    half = dim // 2
    timesteps = timesteps.float()
    # compute sinusoidal embedding
    sinusoid = torch.outer(
        timesteps, torch.pow(10000,
                             -torch.arange(half).to(timesteps).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    if dim % 2 != 0:
        x = torch.cat([x, torch.zeros_like(x[:, :1])], dim=1)
    return x

# feedforward
class GEGLU(nn.Module):

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def _i(tensor, t, x):
    r"""Index tensor using t and format the output according to x.
    """
    tensor = tensor.to(x.device)
    shape = (x.size(0), ) + (1, ) * (x.ndim - 1)
    return tensor[t].view(shape).to(x)

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        mask = torch.zeros(shape, device=device).float().uniform_(0, 1) < prob
        # aviod mask all, which will cause find_unused_parameters error
        if mask.all():
            mask[0] = False
        return mask

def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f'unsupported dimensions: {dims}')

def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f'unsupported dimensions: {dims}')
