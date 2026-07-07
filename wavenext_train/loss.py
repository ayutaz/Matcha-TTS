"""GAN (hinge) / Feature-Matching / mel-L1 losses for WaveNeXt training.

DiscriminatorLoss / GeneratorLoss / FeatureMatchingLoss are ported verbatim (equation
level) from wetdog/wavenext_pytorch vocos/loss.py. MIT License, Copyright (c) 2023
Charactr Inc. Only MelSpecReconstructionLoss diverges: it reuses MatchaMelFeatures
(fmax=11025, matcha mel) instead of torchaudio's MelSpectrogram, so the recon loss lives
in the exact mel domain Matcha feeds the vocoder. matcha's mel_spectrogram already applies
log(clamp(x, 1e-5)), so upstream's extra safe_log is dropped (no double log).
"""

import torch
import torch.nn.functional as F
from torch import nn

from wavenext_train.features import MatchaMelFeatures


class MelSpecReconstructionLoss(nn.Module):
    def __init__(self, mel_features: MatchaMelFeatures = None, fmax: float = 11025):
        super().__init__()
        self.mel_features = mel_features or MatchaMelFeatures(fmax=fmax)

    def forward(self, y_hat, y):  # arg order matches upstream (y_hat, y)
        return F.l1_loss(self.mel_features(y), self.mel_features(y_hat))


class GeneratorLoss(nn.Module):
    """Hinge generator loss: sum of mean(relu(1 - dg))."""

    def forward(self, disc_outputs):
        loss = torch.zeros(1, device=disc_outputs[0].device, dtype=disc_outputs[0].dtype)
        gen_losses = []
        for dg in disc_outputs:
            l = torch.mean(torch.clamp(1 - dg, min=0))
            gen_losses.append(l)
            loss += l
        return loss, gen_losses


class DiscriminatorLoss(nn.Module):
    """Hinge discriminator loss: sum of mean(relu(1 - dr)) + mean(relu(1 + dg))."""

    def forward(self, disc_real_outputs, disc_generated_outputs):
        loss = torch.zeros(1, device=disc_real_outputs[0].device, dtype=disc_real_outputs[0].dtype)
        r_losses = []
        g_losses = []
        for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
            r_loss = torch.mean(torch.clamp(1 - dr, min=0))
            g_loss = torch.mean(torch.clamp(1 + dg, min=0))
            loss += r_loss + g_loss
            r_losses.append(r_loss)
            g_losses.append(g_loss)
        return loss, r_losses, g_losses


class FeatureMatchingLoss(nn.Module):
    """Sum of mean(|rl - gl|) over ragged feature maps."""

    def forward(self, fmap_r, fmap_g):
        loss = torch.zeros(1, device=fmap_r[0][0].device, dtype=fmap_r[0][0].dtype)
        for dr, dg in zip(fmap_r, fmap_g):
            for rl, gl in zip(dr, dg):
                loss += torch.mean(torch.abs(rl - gl))
        return loss
