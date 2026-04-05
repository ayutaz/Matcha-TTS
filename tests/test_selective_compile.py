"""Tests for selective torch.compile in train.py.

Verifies that compile_model=true only compiles decoder.estimator (not encoder),
because the encoder contains einops.rearrange which causes graph breaks.
"""

from types import SimpleNamespace

import pytest
import torch

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
        model.decoder.estimator = torch.compile(
            model.decoder.estimator, mode=compile_mode
        )
    return model


class TestSelectiveCompile:
    """Verify that compile_model only compiles decoder.estimator."""

    @pytest.mark.slow
    def test_compile_true_decoder_estimator_is_compiled(self):
        """compile_model=true should compile decoder.estimator."""
        model = _build_model()
        _apply_compile(model, compile_model=True)
        assert _is_compiled(model.decoder.estimator), (
            "decoder.estimator should be compiled when compile_model=true"
        )

    @pytest.mark.slow
    def test_compile_true_encoder_is_not_compiled(self):
        """compile_model=true should NOT compile the encoder."""
        model = _build_model()
        _apply_compile(model, compile_model=True)
        assert not _is_compiled(model.encoder), (
            "encoder should NOT be compiled (einops causes graph breaks)"
        )

    @pytest.mark.slow
    def test_compile_false_nothing_compiled(self):
        """compile_model=false should leave all modules uncompiled."""
        model = _build_model()
        _apply_compile(model, compile_model=False)
        assert not _is_compiled(model.encoder), (
            "encoder should not be compiled when compile_model=false"
        )
        assert not _is_compiled(model.decoder.estimator), (
            "decoder.estimator should not be compiled when compile_model=false"
        )
