"""TDD tests for wavenext_train.features.MatchaMelFeatures (fmax=11025).

The vocoder is trained on the EXACT mel domain Matcha feeds it at inference:
matcha.utils.audio.mel_spectrogram at fmax=11025, center=False, raw natural-log
log-mel WITHOUT z-score normalization. These tests pin that identity.
"""

import torch

from matcha.utils.audio import mel_spectrogram
from wavenext_train.features import MatchaMelFeatures


def _ref(a):
    return mel_spectrogram(a, 1024, 80, 22050, 256, 1024, 0.0, 11025, center=False)


def test_matches_matcha_mel_spectrogram_exactly():
    torch.manual_seed(0)
    a = torch.rand(2, 16384) * 2 - 1
    out = MatchaMelFeatures(fmax=11025)(a)
    assert torch.equal(out, _ref(a))  # atol=0: core mel-domain identity


def test_output_shape_B_T_to_B_80_Tframes():
    out = MatchaMelFeatures(fmax=11025)(torch.zeros(3, 256 * 40))
    assert out.shape == (3, 80, 40)


def test_stft_is_fp32_under_autocast_and_bf16_input():
    feat = MatchaMelFeatures(fmax=11025)
    a = torch.rand(1, 256 * 32) * 2 - 1
    with torch.autocast(device_type="cuda", enabled=False):  # concept-check on CPU
        out = feat(a)
    assert out.dtype == torch.float32
    out_bf16 = feat(a.to(torch.bfloat16))
    assert out_bf16.dtype == torch.float32
    assert torch.isfinite(out_bf16).all()


def test_frames_times_hop_equals_input_length():
    t = 256 * 50
    out = MatchaMelFeatures(fmax=11025)(torch.zeros(1, t))
    assert out.shape[-1] * 256 == t  # center=False guarantee


def test_fmax_is_honored():
    a = torch.rand(1, 16384) * 2 - 1
    assert not torch.allclose(MatchaMelFeatures(fmax=11025)(a), MatchaMelFeatures(fmax=8000)(a))


def test_no_zscore_normalization():
    feat = MatchaMelFeatures(fmax=11025)
    assert not hasattr(feat, "mel_mean")
    assert not hasattr(feat, "mel_std")
    out = feat(torch.rand(1, 16384) * 2 - 1)
    assert out.min().item() < -1.0  # raw log-mel floor ~= -11.5; z-scored would center near 0
