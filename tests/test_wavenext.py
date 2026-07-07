"""Tests for the vendored WaveNeXt vocoder (matcha/wavenext/).

Network-free: exercises the ported architecture with random init. The key
regressions guarded here are the vocoder IF contract shared with HiFi-GAN:
  - forward(mel(B,80,T)) -> (B, 1, T*256) waveform in [-1, 1]
  - a zero mel (1, 80, 88) is accepted so Denoiser(mode="zeros") can init
  - Denoiser construction (needs a 3D vocoder output)
  - param count ~13.6-13.8M (catches transcription drift in the port)
"""

import torch

from matcha.hifigan.denoiser import Denoiser
from matcha.wavenext import WaveNeXtVocoder


def test_forward_shape_and_range():
    voc = WaveNeXtVocoder().eval()
    with torch.no_grad():
        out = voc(torch.randn(2, 80, 50))
    assert out.shape == (2, 1, 50 * 256)
    assert out.dtype == torch.float32
    assert out.abs().max() <= 1.0  # WaveNextHead clips to [-1, 1]


def test_zero_mel_accepted():
    """Denoiser(mode='zeros') calls vocoder(torch.zeros((1,80,88))) at init."""
    voc = WaveNeXtVocoder().eval()
    with torch.no_grad():
        out = voc(torch.zeros(1, 80, 88))
    assert out.shape == (1, 1, 88 * 256)
    assert torch.isfinite(out).all()


def test_denoiser_constructs():
    """3D (B,1,T) output contract: 2D output would IndexError in Denoiser."""
    voc = WaveNeXtVocoder().eval()
    denoiser = Denoiser(voc, mode="zeros")  # must not raise
    assert denoiser is not None


def test_param_count():
    n = sum(p.numel() for p in WaveNeXtVocoder().parameters())
    assert 13_600_000 <= n <= 13_800_000, f"unexpected param count {n} (port drift?)"


def test_state_dict_keys_prefix():
    """state_dict keys must be backbone.* / head.* to load BSC-LT/wavenext-mel 1:1."""
    keys = WaveNeXtVocoder().state_dict().keys()
    assert all(k.startswith(("backbone.", "head.")) for k in keys)
    # WaveNextHead.linear_2 has no bias (matches the checkpoint).
    assert "head.linear_2.bias" not in keys
    assert "head.linear_2.weight" in keys
