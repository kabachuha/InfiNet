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
# TODO: add a proper license (openrails or apache) before releasing
import importlib
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import open_clip
from os import path as osp

DEFAULT_MODEL_REVISION = None
from ldm.util import instantiate_from_config
class Invoke(object):
    KEY = 'invoked_by'
    PRETRAINED = 'from_pretrained'
    PIPELINE = 'pipeline'
    TRAINER = 'trainer'
    LOCAL_TRAINER = 'local_trainer'
    PREPROCESSOR = 'preprocessor'

class FrozenOpenCLIPEmbedder(torch.nn.Module):
    """
    Uses the OpenCLIP transformer encoder for text
    """
    LAYERS = ['last', 'penultimate']

    def __init__(self,
                 arch='ViT-H-14',
                 version='open_clip_pytorch_model.bin',
                 device='cuda',
                 max_length=77,
                 freeze=True,
                 layer='last'):
        super().__init__()
        assert layer in self.LAYERS
        model, _, _ = open_clip.create_model_and_transforms(
            arch, device=torch.device('cpu'), pretrained=version)
        del model.visual
        self.model = model

        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == 'last':
            self.layer_idx = 0
        elif self.layer == 'penultimate':
            self.layer_idx = 1
        else:
            raise NotImplementedError()

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        tokens = open_clip.tokenize(text)
        z = self.encode_with_transformer(tokens.to(self.device))
        return z

    def encode_with_transformer(self, text):
        x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)
        return x

    def text_transformer_forward(self, x: torch.Tensor, attn_mask=None):
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - self.layer_idx:
                break
            x = r(x, attn_mask=attn_mask)
        return x

    def encode(self, text):
        return self(text)

    def from_pretrained(cls,
                        model_name_or_path: str,
                        revision: Optional[str] = DEFAULT_MODEL_REVISION,
                        cfg_dict = None,
                        device: str = None,
                        **kwargs):
        """Instantiate a model from local directory or remote model repo. Note
        that when loading from remote, the model revision can be specified.

        Args:
            model_name_or_path(str): A model dir or a model id to be loaded
            revision(str, `optional`): The revision used when the model_name_or_path is
                a model id of the remote hub. default `master`.
            cfg_dict(Config, `optional`): An optional model config. If provided, it will replace
                the config read out of the `model_name_or_path`
            device(str, `optional`): The device to load the model.
            **kwargs:
                task(str, `optional`): The `Tasks` enumeration value to replace the task value
                read out of config in the `model_name_or_path`. This is useful when the model to be loaded is not
                equal to the model saved.
                For example, load a `backbone` into a `text-classification` model.
                Other kwargs will be directly fed into the `model` key, to replace the default configs.
        Returns:
            A model instance.

        """
        prefetched = kwargs.get('model_prefetched')
        if prefetched is not None:
            kwargs.pop('model_prefetched')
        invoked_by = kwargs.get(Invoke.KEY)
        if invoked_by is not None:
            kwargs.pop(Invoke.KEY)
        else:
            invoked_by = Invoke.PRETRAINED

        if osp.exists(model_name_or_path):
            local_model_dir = model_name_or_path
        if cfg_dict is not None:
            cfg = cfg_dict
            """else:
            cfg = Config.from_file(
                osp.join(local_model_dir, ModelFile.CONFIGURATION))"""
        task_name = cfg.task
        if 'task' in kwargs:
            task_name = kwargs.pop('task')
        model_cfg = cfg.model
        if hasattr(model_cfg, 'model_type') and not hasattr(model_cfg, 'type'):
            model_cfg.type = model_cfg.model_type
        model_cfg.model_dir = local_model_dir

        print("plugins",cfg.safe_get('plugins'))


        # install and import remote repos before build
        #register_plugins_repo(cfg.safe_get('plugins'))
        #register_modelhub_repo(local_model_dir, cfg.get('allow_remote', False))

        for k, v in kwargs.items():
            model_cfg[k] = v
        if device is not None:
            model_cfg.device = device
        """if task_name is Tasks.backbone:
            model_cfg.init_backbone = True
            model = build_backbone(model_cfg)
        else:"""
        model = instantiate_from_config(model_cfg)
        #model = build_model(model_cfg, task_name=task_name)

        # dynamically add pipeline info to model for pipeline inference
        if hasattr(cfg, 'pipeline'):
            model.pipeline = cfg.pipeline

        if not hasattr(model, 'cfg'):
            model.cfg = cfg

        model_cfg.pop('model_dir', None)
        model.name = model_name_or_path
        model.model_dir = local_model_dir
        return model

