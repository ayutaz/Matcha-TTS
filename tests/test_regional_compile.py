"""Tests for A-2 regional torch.compile (Decoder.compile_regions).

The critical safety guarantee is that compile_regions() uses nn.Module.compile()
in-place, so NO state_dict key changes -> checkpoints / EMA strict-load / resume keep
working. These run on CPU and do not require actually compiling (nn.Module.compile is
lazy; compilation happens on first forward), except the CUDA-only end-to-end test.
"""

import pytest
import torch

from matcha.models.components.decoder import Decoder


def _decoder():
    return Decoder(
        in_channels=80,
        out_channels=80,
        channels=[64, 64],
        dropout=0.0,
        attention_head_dim=32,
        n_blocks=1,
        num_mid_blocks=1,
        num_heads=2,
    )


def _inputs(batch=2, n_feats=40, length=20):
    # Decoder cats x and mu along channels -> each has in_channels // 2 = 40 channels.
    x = torch.randn(batch, n_feats, length)
    mu = torch.randn(batch, n_feats, length)
    mask = torch.ones(batch, 1, length)
    t = torch.rand(batch)
    return x, mask, mu, t


def test_default_not_compiled():
    assert _decoder()._regional_compiled is False


def test_compile_regions_preserves_state_dict_keys():
    """Core guarantee: in-place compile must not add/rename any state_dict key."""
    dec = _decoder()
    before = list(dec.state_dict().keys())
    dec.compile_regions()
    after = list(dec.state_dict().keys())
    assert before == after, "compile_regions() changed state_dict keys -> ckpt/EMA/resume would break"
    assert dec._regional_compiled is True


def test_compile_regions_count():
    # channels=[64,64], n_blocks=1, num_mid_blocks=1 -> down 2 + mid 1 + up 2 = 5 transformer blocks
    assert _decoder().compile_regions() == 5


def test_default_forward_finite():
    """Default path (no compile) must run and produce finite output (forward edits intact)."""
    dec = _decoder().eval()
    x, mask, mu, t = _inputs()
    with torch.no_grad():
        out = dec(x, mask, mu, t)
    assert out.shape == (2, 80, 20)
    assert torch.isfinite(out).all()
    assert dec._regional_compiled is False


def test_state_dict_reloadable_after_compile():
    """A compiled decoder's state_dict must strict-load into a fresh eager decoder."""
    dec = _decoder()
    dec.compile_regions()
    sd = dec.state_dict()
    fresh = _decoder()
    fresh.load_state_dict(sd, strict=True)  # must not raise


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="regional compile forward exercised on CUDA")
def test_compiled_forward_matches_eager_cuda():
    torch.manual_seed(0)
    dec = _decoder().cuda().eval()
    ref = _decoder()
    ref.load_state_dict(dec.state_dict())
    ref = ref.cuda().eval()
    x, mask, mu, t = (v.cuda() for v in _inputs())
    dec.compile_regions()
    with torch.no_grad():
        out_c = dec(x, mask, mu, t)
        out_e = ref(x, mask, mu, t)
    assert out_c.shape == out_e.shape
    assert torch.allclose(out_c, out_e, atol=1e-4)
