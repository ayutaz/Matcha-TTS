"""TDD tests for wavenext_train.loss (hinge GAN + Feature Matching + matcha-mel L1)."""

import torch
import torch.nn.functional as F

from matcha.utils.audio import mel_spectrogram
from wavenext_train.features import MatchaMelFeatures
from wavenext_train.loss import (
    DiscriminatorLoss,
    FeatureMatchingLoss,
    GeneratorLoss,
    MelSpecReconstructionLoss,
)


def test_discriminator_loss_hinge_scalar_and_perdisc_lists():
    dr = [torch.randn(2, 5), torch.randn(2, 3)]
    dg = [torch.randn(2, 5), torch.randn(2, 3)]
    loss, r, g = DiscriminatorLoss()(dr, dg)
    assert loss.numel() == 1 and torch.isfinite(loss).all()
    assert len(r) == len(g) == 2
    manual = sum(torch.mean(F.relu(1 - a)) + torch.mean(F.relu(1 + b)) for a, b in zip(dr, dg))
    assert torch.allclose(loss.squeeze(), manual, atol=1e-6)
    # determinism: strong-real + strong-fake => ~0 hinge loss
    big, nb = torch.full((1, 1, 4), 10.0), torch.full((1, 1, 4), -10.0)
    assert torch.allclose(DiscriminatorLoss()([big], [nb])[0].squeeze(), torch.tensor(0.0), atol=1e-6)


def test_generator_loss_is_hinge_relu_not_neg_mean():
    dg = [torch.randn(2, 5), torch.randn(2, 3)]
    loss, _ = GeneratorLoss()(dg)
    manual = sum(torch.mean(F.relu(1 - x)) for x in dg)
    assert torch.allclose(loss.squeeze(), manual, atol=1e-6)
    neg_mean = sum(-x.mean() for x in dg)
    assert not torch.allclose(loss.squeeze(), neg_mean)  # regression: NOT -dg.mean()


def test_feature_matching_loss_l1_over_ragged_maps():
    fr = [[torch.randn(2, 4, 3), torch.randn(2, 8)], [torch.randn(2, 6)]]
    fg = [[torch.randn(2, 4, 3), torch.randn(2, 8)], [torch.randn(2, 6)]]
    loss = FeatureMatchingLoss()(fr, fg)
    manual = sum(torch.mean(torch.abs(a - b)) for dr, dg in zip(fr, fg) for a, b in zip(dr, dg))
    assert torch.allclose(loss.squeeze(), manual, atol=1e-6)
    assert FeatureMatchingLoss()(fr, fr).item() == 0.0  # identity -> 0


def test_mel_recon_uses_matcha_features_fmax11025_no_double_log():
    mloss = MelSpecReconstructionLoss(fmax=11025)
    assert isinstance(mloss.mel_features, MatchaMelFeatures)
    y = torch.rand(2, 16384) * 2 - 1
    assert mloss(y, y).item() == 0.0  # identity
    yh = torch.rand(2, 16384) * 2 - 1
    ref = F.l1_loss(mel_spectrogram(y, 1024, 80, 22050, 256, 1024, 0.0, 11025, center=False),
                    mel_spectrogram(yh, 1024, 80, 22050, 256, 1024, 0.0, 11025, center=False))
    assert torch.allclose(mloss(yh, y), ref, atol=0)  # arg order (y_hat, y)
    assert torch.isfinite(mloss(yh, y)).all() and torch.isfinite(mloss(y, yh)).all()
    assert torch.isfinite(mloss(yh.unsqueeze(1), y.unsqueeze(1))).all()  # (B,1,T) accepted


def test_all_losses_finite_and_backward_cpu():
    dr = [torch.randn(2, 5, requires_grad=True)]
    dg = [torch.randn(2, 5, requires_grad=True)]
    dloss, _, _ = DiscriminatorLoss()(dr, dg)
    dloss.backward()
    assert dr[0].grad is not None and torch.isfinite(dr[0].grad).all()
    assert dg[0].grad is not None
    a = [[torch.randn(2, 4, requires_grad=True)]]
    b = [[torch.randn(2, 4)]]
    FeatureMatchingLoss()(a, b).backward()
    assert a[0][0].grad is not None and torch.isfinite(a[0][0].grad).all()


def test_generator_and_mel_backward_cpu():
    leaf = torch.randn(2, 5, requires_grad=True)
    GeneratorLoss()([leaf])[0].backward()
    assert leaf.grad is not None and torch.isfinite(leaf.grad).all()
    # make yh a leaf (torch.rand()*2-1 with requires_grad would be non-leaf -> no .grad)
    yh = (torch.rand(1, 16384) * 2 - 1).requires_grad_(True)
    y = torch.rand(1, 16384) * 2 - 1
    MelSpecReconstructionLoss(fmax=11025)(yh, y).backward()
    assert yh.grad is not None and torch.isfinite(yh.grad).all()
