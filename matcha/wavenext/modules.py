"""ConvNeXt building blocks for the WaveNeXt vocoder.

Vendored (pure-torch subset) from https://github.com/wetdog/wavenext_pytorch
(a fork of gemelo-ai/vocos), file vocos/modules.py.
MIT License, Copyright (c) 2023 Charactr Inc. Trimmed to the ConvNeXt blocks
used by VocosBackbone; ResBlock1 / safe_log / symexp / spectral helpers removed.

Attribute names are kept identical to upstream so that the BSC-LT/wavenext-mel
state_dict keys (``backbone.convnext.{i}.{dwconv,norm,pwconv1,pwconv2,gamma}``)
load 1:1 without renaming.
"""

import torch
from torch import nn


class AdaLayerNorm(nn.Module):
    """Adaptive LayerNorm conditioned on an embedding id.

    Not used by the BSC-LT/wavenext-mel config (no adanorm), but kept so the
    module tree matches upstream exactly.
    """

    def __init__(self, num_embeddings, embedding_dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.dim = embedding_dim
        self.scale = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.shift = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        nn.init.ones_(self.scale.weight)
        nn.init.zeros_(self.shift.weight)

    def forward(self, x, cond_embedding_id):
        scale = self.scale(cond_embedding_id)
        shift = self.shift(cond_embedding_id)
        x = nn.functional.layer_norm(x, (self.dim,), eps=self.eps)
        x = x * scale + shift
        return x


class ConvNeXtBlock(nn.Module):
    """ConvNeXt block adapted for 1D audio, operating over channels-last features."""

    def __init__(self, dim, intermediate_dim, layer_scale_init_value, adanorm_num_embeddings=None):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.adanorm = adanorm_num_embeddings is not None
        if adanorm_num_embeddings:
            self.norm = AdaLayerNorm(adanorm_num_embeddings, dim, eps=1e-6)
        else:
            self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x, cond_embedding_id=None):
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        if self.adanorm:
            assert cond_embedding_id is not None
            x = self.norm(x, cond_embedding_id)
        else:
            x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)
        x = residual + x
        return x
