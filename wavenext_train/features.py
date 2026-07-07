"""MatchaMelFeatures — WaveNeXt-training feature extractor (fmax=11025).

Interface mirrors wetdog/wavenext_pytorch ``vocos/feature_extractors.py``
``MelSpectrogramFeatures`` (MIT License, Copyright (c) 2023 Charactr Inc.): an
``nn.Module`` whose ``forward(audio, **kwargs) -> (B, n_mels, T')`` returns a log-mel.
Unlike upstream, the mel is produced by ``matcha.utils.audio.mel_spectrogram`` (NOT
``torchaudio.transforms.MelSpectrogram``) so the vocoder trains on the *exact* mel domain
Matcha feeds it at inference (``synthesise`` -> ``to_waveform`` denormalized log-mel):

  - ``fmax=11025`` (Nyquist for 22050 Hz), ``center=False``; the reflect pad of
    ``(n_fft-hop)/2 == 384`` per side is baked into ``mel_spectrogram`` and matches
    upstream's ``padding="same"``. Keeps head output length == input waveform length.
  - Compression is Matcha's ``log(clamp(x, 1e-5))`` on a Slaney-normed basis.
  - NO z-score normalization: returns the raw natural-log log-mel. ``mel_mean``/``mel_std``
    are irrelevant to the vocoder and deliberately absent.

D11: the STFT/mel is forced to FP32 (``audio.float()`` in a disabled-autocast region) so
bf16-mixed GAN training stays numerically stable and byte-identical to inference.
"""

import torch
from torch import nn

from matcha.utils.audio import mel_spectrogram


class MatchaMelFeatures(nn.Module):
    def __init__(
        self,
        sample_rate=22050,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mels=80,
        fmin=0.0,
        fmax=11025.0,
        center=False,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.center = center

    def forward(self, audio, **kwargs):
        if audio.dim() == 3:  # (B, 1, T) -> (B, T) (generator wrapper safety)
            audio = audio.squeeze(1)
        elif audio.dim() == 1:  # (T,) -> (1, T)
            audio = audio.unsqueeze(0)
        device_type = "cuda" if audio.is_cuda else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # D11: stft in FP32
            mel = mel_spectrogram(
                audio.float(),
                self.n_fft,
                self.n_mels,
                self.sample_rate,
                self.hop_length,
                self.win_length,
                self.fmin,
                self.fmax,
                center=self.center,
            )
        return mel  # (B, 80, T'), natural-log log-mel, no z-score normalization
