"""VocosBackbone and WaveNextHead for the WaveNeXt vocoder.

Vendored (pure-torch subset) from https://github.com/wetdog/wavenext_pytorch
(vocos/models.py, vocos/heads.py). MIT License, Copyright (c) 2023 Charactr Inc.

WaveNextHead replaces Vocos' ISTFTHead with two linear layers + reshape, so the
whole vocoder is standard ops only (Conv1d / LayerNorm / Linear / GELU / view /
clip) — no iSTFT. This is what makes the ONNX / mobile export clean and fast.

Attribute names match upstream so BSC-LT/wavenext-mel state_dict keys
(``backbone.*`` / ``head.*``) load 1:1.
"""

import torch
from torch import nn

from matcha.wavenext.modules import AdaLayerNorm, ConvNeXtBlock


class VocosBackbone(nn.Module):
    """ConvNeXt backbone: (B, input_channels, T) -> (B, T, dim)."""

    def __init__(
        self,
        input_channels,
        dim,
        intermediate_dim,
        num_layers,
        layer_scale_init_value=None,
        adanorm_num_embeddings=None,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.embed = nn.Conv1d(input_channels, dim, kernel_size=7, padding=3)
        self.adanorm = adanorm_num_embeddings is not None
        if adanorm_num_embeddings:
            self.norm = AdaLayerNorm(adanorm_num_embeddings, dim, eps=1e-6)
        else:
            self.norm = nn.LayerNorm(dim, eps=1e-6)
        layer_scale_init_value = layer_scale_init_value or 1 / num_layers
        self.convnext = nn.ModuleList(
            [
                ConvNeXtBlock(
                    dim=dim,
                    intermediate_dim=intermediate_dim,
                    layer_scale_init_value=layer_scale_init_value,
                    adanorm_num_embeddings=adanorm_num_embeddings,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x, bandwidth_id=None):
        x = self.embed(x)
        if self.adanorm:
            assert bandwidth_id is not None
            x = self.norm(x.transpose(1, 2), cond_embedding_id=bandwidth_id)
        else:
            x = self.norm(x.transpose(1, 2))
        x = x.transpose(1, 2)
        for conv_block in self.convnext:
            x = conv_block(x, cond_embedding_id=bandwidth_id)
        x = self.final_layer_norm(x.transpose(1, 2))
        return x


class WaveNextHead(nn.Module):
    """WaveNeXt head: (B, T, dim) -> (B, T * hop_length) waveform in [-1, 1].

    Two linear layers followed by a frame-order reshape. No iSTFT.
    """

    def __init__(self, dim, n_fft, hop_length, padding="same"):
        super().__init__()
        # padding kept for signature parity with ISTFTHead; unused here.
        self.linear_1 = nn.Linear(dim, n_fft + 2)
        self.linear_2 = nn.Linear(n_fft + 2, hop_length, bias=False)
        nn.init.trunc_normal_(self.linear_1.weight, std=0.02)
        nn.init.trunc_normal_(self.linear_2.weight, std=0.02)

    def forward(self, x):
        b = x.shape[0]
        x = self.linear_1(x)
        x = self.linear_2(x)  # (B, T, hop_length)
        audio = x.view(b, -1)  # frame-order concat -> (B, T * hop_length)
        audio = torch.clip(audio, min=-1.0, max=1.0)
        return audio
