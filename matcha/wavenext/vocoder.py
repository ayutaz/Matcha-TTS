"""WaveNeXt vocoder wrapper — a drop-in replacement for the HiFi-GAN Generator.

Input:  (B, 80, T) denormalized natural-log log-mel (the same tensor Matcha's
        ``to_waveform`` feeds HiFi-GAN; mel_mean/mel_std are already folded in by
        ``synthesise``). fmax=8000, hop=256, sr=22050, log(clamp(x, 1e-5)).
Output: (B, 1, T * hop_length) waveform in [-1, 1], 22050 Hz.

The trailing ``unsqueeze(1)`` restores HiFi-GAN's channel dim so that
``Denoiser(mode='zeros')`` (which indexes ``bias_spec[:, :, 0][:, :, None]``),
``to_waveform`` (clamp+squeeze) and ONNX ``MatchaWithVocoder`` (clamp+squeeze(1))
all work unchanged.

Weights: BSC-LT/wavenext-mel (Apache-2.0). state_dict keys ``backbone.*`` / ``head.*``.
"""

import torch
from torch import nn

from matcha.wavenext.models import VocosBackbone, WaveNextHead

# Transcribed from BSC-LT/wavenext-mel config.yaml: 22050 Hz / 80 mel / n_fft 1024 /
# hop 256 / fmax 8000, mel = Slaney norm + Slaney scale + log(clip, 1e-5), matching
# matcha/utils/audio.py::mel_spectrogram. ~13.7M params.
WAVENEXT_CONFIG = {
    "backbone": {"input_channels": 80, "dim": 512, "intermediate_dim": 1536, "num_layers": 8},
    "head": {"dim": 512, "n_fft": 1024, "hop_length": 256, "padding": "same"},
}


class WaveNeXtVocoder(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        config = config or WAVENEXT_CONFIG
        self.backbone = VocosBackbone(**config["backbone"])
        self.head = WaveNextHead(**config["head"])

    def forward(self, mel):
        x = self.backbone(mel)  # (B, T, dim)
        audio = self.head(x)  # (B, T * hop_length)
        return audio.unsqueeze(1)  # (B, 1, T * hop_length) — same rank as HiFi-GAN Generator
