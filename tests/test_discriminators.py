"""TDD tests for wavenext_train.discriminators (MPD + MRD, byte-vendored from wetdog).

All CPU, tiny inputs. Pins the forward contract (4-tuple of ragged lists), per-sub
counts, DiscriminatorP-flattens-2D vs DiscriminatorR-keeps-4D, reflect-pad path, and
gradient flow.
"""

import torch

from wavenext_train.discriminators import MultiPeriodDiscriminator, MultiResolutionDiscriminator


def test_mpd_forward_returns_four_lists_per_period():
    mpd = MultiPeriodDiscriminator()
    y, y_hat = torch.randn(2, 16384), torch.randn(2, 16384)
    y_d_rs, y_d_gs, fmap_rs, fmap_gs = mpd(y, y_hat)
    assert len(y_d_rs) == len(y_d_gs) == len(fmap_rs) == len(fmap_gs) == 5
    assert y_d_rs[0].dim() == 2 and y_d_rs[0].shape[0] == 2  # DiscriminatorP flattens to 2D
    assert all(torch.isfinite(o).all() for o in y_d_rs + y_d_gs)


def test_mpd_fmap_lengths_and_shape():
    mpd = MultiPeriodDiscriminator()
    _, _, fmap_rs, _ = mpd(torch.randn(2, 16384), torch.randn(2, 16384))
    assert all(len(fm) == 5 for fm in fmap_rs)
    assert all(f.dim() == 4 and f.shape[0] == 2 for fm in fmap_rs for f in fm)


def test_mrd_forward_returns_four_lists_per_resolution():
    mrd = MultiResolutionDiscriminator()
    y_d_rs, y_d_gs, fmap_rs, fmap_gs = mrd(torch.randn(2, 16384), torch.randn(2, 16384))
    assert len(y_d_rs) == len(y_d_gs) == len(fmap_rs) == len(fmap_gs) == 3
    assert y_d_rs[0].dim() == 4  # DiscriminatorR does NOT flatten
    assert all(torch.isfinite(o).all() for o in y_d_rs + y_d_gs)


def test_mrd_fmap_lengths():
    mrd = MultiResolutionDiscriminator()
    _, _, fmap_rs, _ = mrd(torch.randn(2, 16384), torch.randn(2, 16384))
    assert all(len(fm) == 21 for fm in fmap_rs)  # 5 bands x 4 + conv_post


def test_default_periods_and_fft_sizes():
    mpd = MultiPeriodDiscriminator()
    mrd = MultiResolutionDiscriminator()
    assert [d.period for d in mpd.discriminators] == [2, 3, 5, 7, 11]
    assert [d.window_length for d in mrd.discriminators] == [2048, 1024, 512]


def test_discriminatorp_reflect_pad_on_indivisible_length():
    mpd = MultiPeriodDiscriminator()
    y = torch.randn(2, 12345)  # not divisible by any period
    y_d_rs, _, _, _ = mpd(y, y)
    assert all(torch.isfinite(o).all() for o in y_d_rs)


def test_gan_backward_flows_to_discriminator_params():
    mpd = MultiPeriodDiscriminator()
    mrd = MultiResolutionDiscriminator()
    y = torch.randn(2, 16384)
    y_hat = torch.randn(2, 16384, requires_grad=True)
    loss = torch.zeros(())
    for disc in (mpd, mrd):
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = disc(y, y_hat)
        loss = loss + sum(o.mean() for o in y_d_rs + y_d_gs)
        loss = loss + sum(f.mean() for fm in fmap_rs + fmap_gs for f in fm)
    loss.backward()
    assert y_hat.grad is not None and torch.isfinite(y_hat.grad).all()
    assert any(p.grad is not None for p in mpd.parameters())


def test_unconditional_num_embeddings_none_path():
    mpd = MultiPeriodDiscriminator()  # default num_embeddings=None
    for d in mpd.discriminators:
        assert not hasattr(d, "emb")
    y_d_rs, _, _, _ = mpd(torch.randn(2, 8192), torch.randn(2, 8192))  # no bandwidth_id
    assert len(y_d_rs) == 5
