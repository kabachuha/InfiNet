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

from ldm.modules.diffusionmodules.model import Decoder, Encoder
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution

from .attention import *
from .conv_blocks import *
from .utils import sinusoidal_embedding, prob_mask_like


class DoDBlock(nn.Module):
    """
    A downconvolution layer with masked video latents
    Gets the masked video latents (the first and the last frame) and makes a masked convolution
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: Always 3D, downsampling occurs in the inner-two dimensions.
    """

    def __init__(self,
                 channels,
                 dims=3,
                 depth=0,
                 out_channels=None,
                 padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.dims = dims
        stride = (1, 2**depth, 2**depth) # if depth is zero, the stride is 1

        # Convolution block, which should be initialized with zero weights and biases
        # (zero conv)
        self.conv_w = nn.Conv2d(
            self.channels,
            self.out_channels,
            3,
            stride=stride,
            padding=padding)
        
        self.conv_b = nn.Conv2d(
            self.channels,
            self.out_channels,
            3,
            stride=stride,
            padding=padding)
        
        # Conv for masking
        self.mask_conv_w = nn.Conv2d(
            1, # only black and white
            self.out_channels,
            3,
            stride=stride,
            padding=padding)
        
        self.mask_conv_b = nn.Conv2d(
            1, # only black and white
            self.out_channels,
            3,
            stride=stride,
            padding=padding)

    # h - hidden states, x_c - frame conditioning, x_m - masked video latents
    def forward(self, h, x_c=None, x_m=None):

        # When no frame conditioning is provided (top DoD iteration)
        # return the untouched hidden states
        if x_c is None or x_m is None:
            return h

        # Add image conditioning as linear operation

        # get weights and biases from frame conditioning
        # vid convolution (initialized with zero weights and biases at first)
        x_c_w = self.conv_w(x_c)
        x_c_b = self.conv_b(x_c)
        
        h = x_c_w * h + x_c_b + h # uses hadamard product

        # Use masked video latents to mask the convolution
        x_m_w = self.mask_conv_w(x_m)
        x_m_b = self.mask_conv_b(x_m)

        h = x_m_w * h + x_m_b + h # uses hadamard product
        
        return h
    
    def _init_weights(self):
        # Zero initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.constant_(m.weight, 0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

# Class to keep DiffusionOverDiffusion modules as a separate model
# with weights saveable as a detachable checkpoint
class InfiNet(nn.Module):
    def __init__(self,
                 in_dim=7,
                 dim=512,
                 y_dim=512,
                 context_dim=512,
                 out_dim=6,
                 dim_mult=[1, 2, 3, 4],
                 num_res_blocks=3,
                 dropout=0.1):
        super(InfiNet, self).__init__()
        self.in_dim = in_dim
        self.dim = dim
        self.y_dim = y_dim
        self.context_dim = context_dim
        self.out_dim = out_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.dropout = dropout

        self.input_blocks_injections = nn.ModuleList()
        self.output_blocks_injections = nn.ModuleList()
    
    def _init_weights(self):
        # Zero initialization
        for m in self.modules():
            if isinstance(m, DoDBlock):
                m._init_weights()
            if isinstance(m, nn.ModuleList):
                for l in m:
                    if isinstance(l, DoDBlock):
                        l._init_weights()

# TODO: There should be a better way to do hijack it, but I don't want to bother with it atm
class UNet_with_Infinet_SD(nn.Module):

    def __init__(self,
                 in_dim=7,
                 dim=512,
                 y_dim=512,
                 context_dim=512,
                 out_dim=6,
                 dim_mult=[1, 2, 3, 4],
                 num_heads=None,
                 head_dim=64,
                 num_res_blocks=3,
                 attn_scales=[1 / 2, 1 / 4, 1 / 8],
                 use_scale_shift_norm=True,
                 dropout=0.1,
                 temporal_attn_times=2,
                 temporal_attention=True,
                 use_checkpoint=False,
                 use_image_dataset=False,
                 use_fps_condition=False,
                 use_sim_mask=False,
                 use_infinet=True,):
        embed_dim = dim * 4
        num_heads = num_heads if num_heads else dim // 32
        super(UNet_with_Infinet_SD, self).__init__()
        self.in_dim = in_dim
        self.dim = dim
        self.y_dim = y_dim
        self.context_dim = context_dim
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.dim_mult = dim_mult
        self.num_heads = num_heads
        # parameters for spatial/temporal attention
        self.head_dim = head_dim
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.use_scale_shift_norm = use_scale_shift_norm
        self.temporal_attn_times = temporal_attn_times
        self.temporal_attention = temporal_attention
        self.use_checkpoint = use_checkpoint
        self.use_image_dataset = use_image_dataset
        self.use_fps_condition = use_fps_condition
        self.use_sim_mask = use_sim_mask
        use_linear_in_temporal = False
        self.use_infinet = use_infinet
        transformer_depth = 1
        disabled_sa = False
        # params
        enc_dims = [dim * u for u in [1] + dim_mult]
        dec_dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        shortcut_dims = []
        scale = 1.0

        # embeddings
        self.time_embed = nn.Sequential(
            nn.Linear(dim, embed_dim), nn.SiLU(),
            nn.Linear(embed_dim, embed_dim))

        if self.use_fps_condition:
            self.fps_embedding = nn.Sequential(
                nn.Linear(dim, embed_dim), nn.SiLU(),
                nn.Linear(embed_dim, embed_dim))
            nn.init.zeros_(self.fps_embedding[-1].weight)
            nn.init.zeros_(self.fps_embedding[-1].bias)

        if self.use_infinet:
            # Introduce InfiNet
            self.infinet = InfiNet(
                in_dim=in_dim,
                dim=dim,
                y_dim=y_dim,
                context_dim=context_dim,
                out_dim=out_dim,
                dim_mult=dim_mult,
                num_res_blocks=num_res_blocks,
                dropout=dropout)

        # encoder
        self.input_blocks = nn.ModuleList()
        init_block = nn.ModuleList([nn.Conv2d(self.in_dim, dim, 3, padding=1)])

        if temporal_attention:
            init_block.append(
                TemporalTransformer(
                    dim,
                    num_heads,
                    head_dim,
                    depth=transformer_depth,
                    context_dim=context_dim,
                    disable_self_attn=disabled_sa,
                    use_linear=use_linear_in_temporal,
                    multiply_zero=use_image_dataset))

        self.input_blocks.append(init_block)
        shortcut_dims.append(dim)
        for i, (in_dim,
                out_dim) in enumerate(zip(enc_dims[:-1], enc_dims[1:])):
            for j in range(num_res_blocks):
                # residual (+attention) blocks
                block = nn.ModuleList([
                    ResBlock(
                        in_dim,
                        embed_dim,
                        dropout,
                        out_channels=out_dim,
                        use_scale_shift_norm=False,
                        use_image_dataset=use_image_dataset,
                    )
                ])
                if self.use_infinet:
                    self.infinet.input_blocks_injections.append(DoDBlock(self.in_dim,
                        embed_dim,
                        out_channels=out_dim,
                        depth=j
                    ))
                    block.append(self.infinet.input_blocks_injections[-1])

                if scale in attn_scales:
                    block.append(
                        SpatialTransformer(
                            out_dim,
                            out_dim // head_dim,
                            head_dim,
                            depth=1,
                            context_dim=self.context_dim,
                            disable_self_attn=False,
                            use_linear=True))
                    if self.temporal_attention:
                        block.append(
                            TemporalTransformer(
                                out_dim,
                                out_dim // head_dim,
                                head_dim,
                                depth=transformer_depth,
                                context_dim=context_dim,
                                disable_self_attn=disabled_sa,
                                use_linear=use_linear_in_temporal,
                                multiply_zero=use_image_dataset))

                in_dim = out_dim
                self.input_blocks.append(block)
                shortcut_dims.append(out_dim)

                # downsample
                if i != len(dim_mult) - 1 and j == num_res_blocks - 1:
                    downsample = Downsample(
                        out_dim, True, dims=2, out_channels=out_dim)
                    shortcut_dims.append(out_dim)
                    scale /= 2.0
                    self.input_blocks.append(downsample)

        # middle
        self.middle_block = nn.ModuleList([
            ResBlock(
                out_dim,
                embed_dim,
                dropout,
                use_scale_shift_norm=False,
                use_image_dataset=use_image_dataset,
            ),
            SpatialTransformer(
                out_dim,
                out_dim // head_dim,
                head_dim,
                depth=1,
                context_dim=self.context_dim,
                disable_self_attn=False,
                use_linear=True)
        ])

        if self.temporal_attention:
            self.middle_block.append(
                TemporalTransformer(
                    out_dim,
                    out_dim // head_dim,
                    head_dim,
                    depth=transformer_depth,
                    context_dim=context_dim,
                    disable_self_attn=disabled_sa,
                    use_linear=use_linear_in_temporal,
                    multiply_zero=use_image_dataset,
                ))

        self.middle_block.append(
            ResBlock(
                out_dim,
                embed_dim,
                dropout,
                use_scale_shift_norm=False,
                use_image_dataset=use_image_dataset,
            ))

        # decoder
        self.output_blocks = nn.ModuleList()
        for i, (in_dim,
                out_dim) in enumerate(zip(dec_dims[:-1], dec_dims[1:])):
            for j in range(num_res_blocks + 1):
                # residual (+attention) blocks
                block = nn.ModuleList([
                    ResBlock(
                        in_dim + shortcut_dims.pop(),
                        embed_dim,
                        dropout,
                        out_dim,
                        use_scale_shift_norm=False,
                        use_image_dataset=use_image_dataset,
                    )
                ])

                if self.use_infinet:
                    self.infinet.output_blocks_injections.append(DoDBlock(self.in_dim,
                        embed_dim,
                        out_channels=out_dim,
                        depth=j
                    ))
                    block.append(self.infinet.output_blocks_injections[-1])

                if scale in attn_scales:
                    block.append(
                        SpatialTransformer(
                            out_dim,
                            out_dim // head_dim,
                            head_dim,
                            depth=1,
                            context_dim=1024,
                            disable_self_attn=False,
                            use_linear=True))

                    if self.temporal_attention:
                        block.append(
                            TemporalTransformer(
                                out_dim,
                                out_dim // head_dim,
                                head_dim,
                                depth=transformer_depth,
                                context_dim=context_dim,
                                disable_self_attn=disabled_sa,
                                use_linear=use_linear_in_temporal,
                                multiply_zero=use_image_dataset))
                in_dim = out_dim

                # upsample
                if i != len(dim_mult) - 1 and j == num_res_blocks:
                    upsample = Upsample(
                        out_dim, True, dims=2.0, out_channels=out_dim)
                    scale *= 2.0
                    block.append(upsample)
                self.output_blocks.append(block)

        # head
        self.out = nn.Sequential(
            nn.GroupNorm(32, out_dim), nn.SiLU(),
            nn.Conv2d(out_dim, self.out_dim, 3, padding=1))

        # zero out the last layer params
        nn.init.zeros_(self.out[-1].weight)

        # zero out the infinet
        if self.use_infinet:
            self.infinet._init_weights()

    def forward(
            self,
            x, # (B, T, C, H, W) - latent variables of the original video
            t, # timesteps
            y, # text conditioning
            fps=None,
            video_mask=None,
            focus_present_mask=None,
            prob_focus_present=0.,
            mask_last_frame_num=0,  # mask last frame num
            diffusion_depth=0, # Whether we are concerned with making the overall keyframes or filling in the gaps
    ):
        """
        prob_focus_present: probability at which a given batch sample will focus on the present
                            (0. is all off, 1. is completely arrested attention across time)
        """
        batch, device = x.shape[0], x.device
        self.batch = batch

        # image and video joint training, if mask_last_frame_num is set, prob_focus_present will be ignored
        if mask_last_frame_num > 0:
            focus_present_mask = None
            video_mask[-mask_last_frame_num:] = False
        else:
            focus_present_mask = default(
                focus_present_mask, lambda: prob_mask_like(
                    (batch, ), prob_focus_present, device=device))

        time_rel_pos_bias = None
        # embeddings
        if self.use_fps_condition and fps is not None:
            e = self.time_embed(sinusoidal_embedding(
                t, self.dim)) + self.fps_embedding(
                    sinusoidal_embedding(fps, self.dim))
        else:
            e = self.time_embed(sinusoidal_embedding(t, self.dim))
        context = y

        # repeat f times for spatial e and context
        f = x.shape[2]
        e = e.repeat_interleave(repeats=f, dim=0)
        context = context.repeat_interleave(repeats=f, dim=0)

        # If aiming for DiffusionOverDiffusion and have InfiNet enabled, keep the original video
        # + its mask of the first and last frames

        if self.use_infinet and diffusion_depth > 0:
            x_c = x.clone().detach()
            x_m = torch.zeros_like(x_c)
            x_m[:, :, 0, :, :] = torch.ones_like(x_c[:, :, 0, :, :])
            x_m[:, :, -1, :, :] = torch.ones_like(x_c[:, :, -1, :, :])
            x_c = rearrange(x_c, 'b c f h w -> (b f) c h w')
            x_m = rearrange(x_m, 'b c f h w -> (b f) c h w')
        else:
            x_c = None
            x_m = None

        # always in shape (b f) c h w, except for temporal layer
        x = rearrange(x, 'b c f h w -> (b f) c h w')

        # encoder
        xs = []
        for block in self.input_blocks:
            x = self._forward_single(block, x, e, context, time_rel_pos_bias,
                                     focus_present_mask, video_mask,
                                     x_c=x_c, x_m=x_m)
            xs.append(x)

        # middle
        for block in self.middle_block:
            x = self._forward_single(block, x, e, context, time_rel_pos_bias,
                                     focus_present_mask, video_mask)

        # decoder
        for block in self.output_blocks:
            x = torch.cat([x, xs.pop()], dim=1)
            x = self._forward_single(
                block,
                x,
                e,
                context,
                time_rel_pos_bias,
                focus_present_mask,
                video_mask,
                reference=xs[-1] if len(xs) > 0 else None,
                x_c=x_c, x_m=x_m)

        # head
        x = self.out(x)
        # reshape back to (b c f h w)
        x = rearrange(x, '(b f) c h w -> b c f h w', b=batch)
        return x

    def _forward_single(self,
                        module,
                        x,
                        e,
                        context,
                        time_rel_pos_bias,
                        focus_present_mask,
                        video_mask,
                        reference=None,
                        x_c=None, x_m=None):
        if isinstance(module, ResidualBlock):
            x = x.contiguous()
            x = module(x, e, reference)
        elif isinstance(module, ResBlock):
            x = x.contiguous()
            x = module(x, e, self.batch)
        elif isinstance(module, SpatialTransformer):
            x = module(x, context)
        elif isinstance(module, TemporalTransformer):
            x = rearrange(x, '(b f) c h w -> b c f h w', b=self.batch)
            x = module(x, context)
            x = rearrange(x, 'b c f h w -> (b f) c h w')
        elif isinstance(module, CrossAttention):
            x = module(x, context)
        elif isinstance(module, BasicTransformerBlock):
            x = module(x, context)
        elif isinstance(module, FeedForward):
            x = module(x, context)
        elif isinstance(module, Upsample):
            x = module(x)
        elif isinstance(module, Downsample):
            x = module(x)
        elif isinstance(module, DoDBlock):
            x = module(x, x_c, x_m)
        elif isinstance(module, Resample):
            x = module(x, reference)
        elif isinstance(module, nn.ModuleList):
            for block in module:
                x = self._forward_single(block, x, e, context,
                                         time_rel_pos_bias, focus_present_mask,
                                         video_mask, reference, x_c, x_m)
        else:
            x = module(x)
        return x
