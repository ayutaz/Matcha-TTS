"""Tests for selective torch.compile in train.py.

Verifies that compile_model=true only compiles decoder.estimator (not encoder),
because the encoder contains einops.rearrange which causes graph breaks.
Also verifies that compiled decoder.estimator can run a forward pass without
Inductor errors (e.g. SnakeBeta sympy assertion).
"""

from types import SimpleNamespace

import pytest
import torch

from matcha.models.components.decoder import Decoder
from matcha.models.matcha_tts import MatchaTTS


def _make_encoder_config(
    n_feats=80,
    n_channels=64,
    filter_channels=128,
    filter_channels_dp=64,
    n_heads=2,
    n_layers=1,
    kernel_size=3,
    p_dropout=0.0,
    spk_emb_dim=64,
    n_spks=1,
    prenet=True,
):
    encoder_params = SimpleNamespace(
        n_feats=n_feats,
        n_channels=n_channels,
        filter_channels=filter_channels,
        filter_channels_dp=filter_channels_dp,
        n_heads=n_heads,
        n_layers=n_layers,
        kernel_size=kernel_size,
        p_dropout=p_dropout,
        spk_emb_dim=spk_emb_dim,
        n_spks=n_spks,
        prenet=prenet,
    )
    duration_predictor_params = SimpleNamespace(
        filter_channels_dp=filter_channels_dp,
        kernel_size=kernel_size,
        p_dropout=p_dropout,
    )
    return SimpleNamespace(
        encoder_type="RoPE Encoder",
        encoder_params=encoder_params,
        duration_predictor_params=duration_predictor_params,
    )


def _make_decoder_config():
    return {
        "channels": [64, 64],
        "dropout": 0.0,
        "attention_head_dim": 32,
        "n_blocks": 1,
        "num_mid_blocks": 1,
        "num_heads": 2,
        "act_fn": "snakebeta",
    }


def _make_cfm_config():
    return SimpleNamespace(
        name="CFM",
        solver="euler",
        sigma_min=1e-4,
    )


def _build_model():
    """Instantiate a minimal MatchaTTS model for compile testing."""
    return MatchaTTS(
        n_vocab=55,
        n_spks=1,
        spk_emb_dim=64,
        n_feats=80,
        encoder=_make_encoder_config(),
        decoder=_make_decoder_config(),
        cfm=_make_cfm_config(),
        data_statistics={"mel_mean": 0.0, "mel_std": 1.0},
        out_size=None,
    )


def _is_compiled(module):
    """Check if a module has been wrapped by torch.compile."""
    return isinstance(module, torch._dynamo.eval_frame.OptimizedModule)


def _apply_compile(model, compile_model, compile_mode="default"):
    """Replicate the compile logic from matcha/train.py."""
    if compile_model:
        model.decoder.estimator = torch.compile(model.decoder.estimator, mode=compile_mode)
    return model


class TestSelectiveCompile:
    """Verify that compile_model only compiles decoder.estimator."""

    @pytest.mark.slow
    def test_compile_true_decoder_estimator_is_compiled(self):
        """compile_model=true should compile decoder.estimator."""
        model = _build_model()
        _apply_compile(model, compile_model=True)
        assert _is_compiled(model.decoder.estimator), "decoder.estimator should be compiled when compile_model=true"

    @pytest.mark.slow
    def test_compile_true_encoder_is_not_compiled(self):
        """compile_model=true should NOT compile the encoder."""
        model = _build_model()
        _apply_compile(model, compile_model=True)
        assert not _is_compiled(model.encoder), "encoder should NOT be compiled (einops causes graph breaks)"

    @pytest.mark.slow
    def test_compile_false_nothing_compiled(self):
        """compile_model=false should leave all modules uncompiled."""
        model = _build_model()
        _apply_compile(model, compile_model=False)
        assert not _is_compiled(model.encoder), "encoder should not be compiled when compile_model=false"
        assert not _is_compiled(model.decoder.estimator), (
            "decoder.estimator should not be compiled when compile_model=false"
        )


class TestCompiledDecoderForward:
    """Verify that a compiled decoder.estimator with SnakeBeta can run forward.

    The Decoder's forward() concatenates x and mu along dim=1, so the
    Decoder is constructed with in_channels = 2 * n_feats (accounting for
    the cat), while x and mu each have n_feats channels.
    """

    @staticmethod
    def _build_decoder(n_feats=80, channels=(64, 64)):
        """Build a minimal Decoder with snakebeta activation."""
        return Decoder(
            in_channels=2 * n_feats,
            out_channels=n_feats,
            channels=channels,
            dropout=0.0,
            attention_head_dim=32,
            n_blocks=1,
            num_mid_blocks=1,
            num_heads=2,
            act_fn="snakebeta",
        )

    @staticmethod
    def _make_inputs(n_feats, batch=1, time=20, device=None):
        """Create dummy inputs matching Decoder.forward(x, mask, mu, t) signature."""
        kwargs = {"device": device} if device else {}
        x = torch.randn(batch, n_feats, time, **kwargs)
        mask = torch.ones(batch, 1, time, **kwargs)
        mu = torch.randn(batch, n_feats, time, **kwargs)
        t = torch.tensor([0.5], **kwargs)
        return x, mask, mu, t

    def test_compiled_decoder_forward_cpu(self):
        """Compiled decoder.estimator should complete a forward pass on CPU.

        This catches Inductor/sympy errors caused by SnakeBeta's exp/sin/pow
        ops when torch.compile traces through them without the
        @torch.compiler.disable guard.
        """
        n_feats = 80
        decoder = self._build_decoder(n_feats=n_feats)
        decoder.eval()

        compiled_decoder = torch.compile(decoder, backend="eager")
        x, mask, mu, t = self._make_inputs(n_feats)

        with torch.no_grad():
            out = compiled_decoder(x, mask, mu, t)

        assert out.shape == (1, n_feats, 20), f"Expected output shape (1, {n_feats}, 20), got {out.shape}"
        assert torch.isfinite(out).all(), "Output contains non-finite values"

    def test_compiled_decoder_matches_uncompiled(self):
        """Compiled and uncompiled decoder should produce identical outputs."""
        n_feats = 80
        decoder = self._build_decoder(n_feats=n_feats)
        decoder.eval()

        compiled_decoder = torch.compile(decoder, backend="eager")

        torch.manual_seed(42)
        x, mask, mu, t = self._make_inputs(n_feats)

        with torch.no_grad():
            out_compiled = compiled_decoder(x, mask, mu, t)
            out_eager = decoder(x, mask, mu, t)

        assert torch.allclose(out_compiled, out_eager, atol=1e-6), "Compiled and uncompiled decoder outputs differ"

    @pytest.mark.slow
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_compiled_decoder_forward_cuda_inductor(self):
        """Compiled decoder.estimator should complete a forward pass on CUDA
        using the inductor backend (the default for torch.compile).

        This is the scenario that triggers the original SnakeBeta sympy
        assertion error without the @torch.compiler.disable fix.
        """
        n_feats = 80
        device = torch.device("cuda")
        decoder = self._build_decoder(n_feats=n_feats).to(device)
        decoder.eval()

        compiled_decoder = torch.compile(decoder, mode="reduce-overhead")
        x, mask, mu, t = self._make_inputs(n_feats, device=device)

        with torch.no_grad():
            out = compiled_decoder(x, mask, mu, t)

        assert out.shape == (1, n_feats, 20), f"Expected output shape (1, {n_feats}, 20), got {out.shape}"
        assert torch.isfinite(out).all(), "Output contains non-finite values"
