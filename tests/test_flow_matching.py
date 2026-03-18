"""Tests for the Matcha-TTS flow matching (CFM) module."""

import types

import pytest
import torch

from matcha.models.components.flow_matching import BASECFM, CFM

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _cfm_params(**overrides):
    """Return a minimal cfm_params namespace accepted by BASECFM / CFM."""
    defaults = {"solver": "euler", "sigma_min": 1e-4}
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


def _decoder_params():
    """Return the smallest viable Decoder kwargs for fast CPU tests."""
    return {
        "channels": [64, 64],
        "dropout": 0.0,
        "attention_head_dim": 32,
        "n_blocks": 1,
        "num_mid_blocks": 1,
        "num_heads": 2,
    }


# Tensor dimensions shared across tests.
BATCH = 2
N_FEATS = 80  # mel feature dimension
MEL_LEN = 20  # time-steps in the mel spectrogram
IN_CHANNELS = 2 * N_FEATS  # Decoder packs (x, mu) along channel dim
OUT_CHANNEL = N_FEATS


@pytest.fixture()
def cfm_model():
    """Instantiate a small CFM model for testing."""
    model = CFM(
        in_channels=IN_CHANNELS,
        out_channel=OUT_CHANNEL,
        cfm_params=_cfm_params(),
        decoder_params=_decoder_params(),
        n_spks=1,
        spk_emb_dim=64,
    )
    model.eval()
    return model


@pytest.fixture()
def sample_tensors():
    """Return (mu, mask, x1) tensors used by most tests."""
    mu = torch.randn(BATCH, N_FEATS, MEL_LEN)
    mask = torch.ones(BATCH, 1, MEL_LEN)
    x1 = torch.randn(BATCH, N_FEATS, MEL_LEN)
    return mu, mask, x1


# ---------------------------------------------------------------------------
# 1. BASECFM instantiation
# ---------------------------------------------------------------------------


class TestBASECFMInstantiation:
    """Verify that BASECFM stores its configuration correctly."""

    def test_default_sigma_min(self):
        params = _cfm_params()
        del params.sigma_min  # let the class fall back to default
        model = CFM(
            in_channels=IN_CHANNELS,
            out_channel=OUT_CHANNEL,
            cfm_params=params,
            decoder_params=_decoder_params(),
        )
        assert model.sigma_min == 1e-4, "Default sigma_min should be 1e-4"

    def test_custom_sigma_min(self):
        model = CFM(
            in_channels=IN_CHANNELS,
            out_channel=OUT_CHANNEL,
            cfm_params=_cfm_params(sigma_min=0.01),
            decoder_params=_decoder_params(),
        )
        assert model.sigma_min == 0.01

    def test_n_feats_stored(self):
        model = CFM(
            in_channels=IN_CHANNELS,
            out_channel=OUT_CHANNEL,
            cfm_params=_cfm_params(),
            decoder_params=_decoder_params(),
        )
        assert model.n_feats == IN_CHANNELS

    def test_solver_stored(self):
        model = CFM(
            in_channels=IN_CHANNELS,
            out_channel=OUT_CHANNEL,
            cfm_params=_cfm_params(solver="euler"),
            decoder_params=_decoder_params(),
        )
        assert model.solver == "euler"

    def test_single_speaker_defaults(self):
        model = CFM(
            in_channels=IN_CHANNELS,
            out_channel=OUT_CHANNEL,
            cfm_params=_cfm_params(),
            decoder_params=_decoder_params(),
        )
        assert model.n_spks == 1

    def test_estimator_is_decoder(self):
        model = CFM(
            in_channels=IN_CHANNELS,
            out_channel=OUT_CHANNEL,
            cfm_params=_cfm_params(),
            decoder_params=_decoder_params(),
        )
        assert model.estimator is not None
        from matcha.models.components.decoder import Decoder

        assert isinstance(model.estimator, Decoder)


# ---------------------------------------------------------------------------
# 2. forward pass (inference mode) — output shape
# ---------------------------------------------------------------------------


class TestForwardPass:
    """The forward method runs the Euler ODE solver and returns a mel tensor."""

    def test_output_shape(self, cfm_model, sample_tensors):
        mu, mask, _ = sample_tensors
        n_timesteps = 2
        output = cfm_model(mu, mask, n_timesteps=n_timesteps, temperature=1.0)
        assert output.shape == (BATCH, N_FEATS, MEL_LEN)

    def test_output_dtype_float32(self, cfm_model, sample_tensors):
        mu, mask, _ = sample_tensors
        output = cfm_model(mu, mask, n_timesteps=2)
        assert output.dtype == torch.float32

    def test_partial_mask_produces_valid_output(self, cfm_model, sample_tensors):
        mu, _, _ = sample_tensors
        # Mask out the last 5 time-steps.
        mask = torch.ones(BATCH, 1, MEL_LEN)
        mask[:, :, -5:] = 0.0
        output = cfm_model(mu, mask, n_timesteps=2)
        assert output.shape == (BATCH, N_FEATS, MEL_LEN)
        assert torch.isfinite(output).all()


# ---------------------------------------------------------------------------
# 3. compute_loss
# ---------------------------------------------------------------------------


class TestComputeLoss:
    """compute_loss returns (scalar loss, interpolated sample y)."""

    def test_loss_is_scalar(self, cfm_model, sample_tensors):
        mu, mask, x1 = sample_tensors
        loss, _ = cfm_model.compute_loss(x1=x1, mask=mask, mu=mu)
        assert loss.dim() == 0, "Loss must be a scalar tensor"

    def test_loss_is_finite(self, cfm_model, sample_tensors):
        mu, mask, x1 = sample_tensors
        loss, _ = cfm_model.compute_loss(x1=x1, mask=mask, mu=mu)
        assert torch.isfinite(loss), "Loss must be finite"

    def test_loss_is_nonnegative(self, cfm_model, sample_tensors):
        mu, mask, x1 = sample_tensors
        loss, _ = cfm_model.compute_loss(x1=x1, mask=mask, mu=mu)
        assert loss.item() >= 0.0, "MSE-based loss cannot be negative"

    def test_y_shape(self, cfm_model, sample_tensors):
        mu, mask, x1 = sample_tensors
        _, y = cfm_model.compute_loss(x1=x1, mask=mask, mu=mu)
        assert y.shape == (BATCH, N_FEATS, MEL_LEN)

    def test_loss_requires_grad(self, cfm_model, sample_tensors):
        """Loss must be differentiable so it can drive training."""
        cfm_model.train()
        mu, mask, x1 = sample_tensors
        loss, _ = cfm_model.compute_loss(x1=x1, mask=mask, mu=mu)
        assert loss.requires_grad

    def test_loss_backward(self, cfm_model, sample_tensors):
        """Gradient can flow back through the estimator."""
        cfm_model.train()
        mu, mask, x1 = sample_tensors
        loss, _ = cfm_model.compute_loss(x1=x1, mask=mask, mu=mu)
        loss.backward()
        # At least one parameter in the estimator should have a gradient.
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in cfm_model.estimator.parameters() if p.requires_grad
        )
        assert has_grad, "Gradients must flow to the estimator parameters"


# ---------------------------------------------------------------------------
# 4. Euler ODE solver — shape and determinism
# ---------------------------------------------------------------------------


class TestSolveEuler:
    """Direct tests for solve_euler (called internally by forward)."""

    def test_output_shape(self, cfm_model, sample_tensors):
        mu, mask, _ = sample_tensors
        x = torch.randn(BATCH, N_FEATS, MEL_LEN)
        t_span = torch.linspace(0, 1, 3)  # 2 steps
        with torch.inference_mode():
            out = cfm_model.solve_euler(x, t_span=t_span, mu=mu, mask=mask, spks=None, cond=None)
        assert out.shape == (BATCH, N_FEATS, MEL_LEN)

    def test_more_steps_changes_output(self, cfm_model, sample_tensors):
        """Using more ODE steps should generally produce a different result."""
        mu, mask, _ = sample_tensors
        torch.manual_seed(0)
        x = torch.randn(BATCH, N_FEATS, MEL_LEN)
        with torch.inference_mode():
            out_2 = cfm_model.solve_euler(
                x.clone(),
                t_span=torch.linspace(0, 1, 3),
                mu=mu,
                mask=mask,
                spks=None,
                cond=None,
            )
            out_5 = cfm_model.solve_euler(
                x.clone(),
                t_span=torch.linspace(0, 1, 6),
                mu=mu,
                mask=mask,
                spks=None,
                cond=None,
            )
        assert not torch.allclose(out_2, out_5, atol=1e-5), "Different step counts should yield different outputs"

    def test_single_step(self, cfm_model, sample_tensors):
        """Even a single Euler step should produce a valid tensor."""
        mu, mask, _ = sample_tensors
        x = torch.randn(BATCH, N_FEATS, MEL_LEN)
        t_span = torch.linspace(0, 1, 2)  # 1 step
        with torch.inference_mode():
            out = cfm_model.solve_euler(x, t_span=t_span, mu=mu, mask=mask, spks=None, cond=None)
        assert out.shape == (BATCH, N_FEATS, MEL_LEN)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# 5. Temperature parameter
# ---------------------------------------------------------------------------


class TestTemperature:
    """Temperature scales the initial noise variance in the forward pass."""

    def test_zero_temperature_starts_from_zero_noise(self, cfm_model, sample_tensors):
        """With temperature=0 the initial noise is all zeros, so the ODE
        evolves from the origin. The output should differ from temperature=1."""
        mu, mask, _ = sample_tensors
        torch.manual_seed(42)
        out_t0 = cfm_model(mu, mask, n_timesteps=2, temperature=0.0)
        torch.manual_seed(42)
        out_t1 = cfm_model(mu, mask, n_timesteps=2, temperature=1.0)
        assert not torch.allclose(out_t0, out_t1, atol=1e-5), (
            "temperature=0 and temperature=1 should produce different outputs"
        )

    def test_temperature_scales_noise(self, cfm_model, sample_tensors):
        """Higher temperature should generally increase the magnitude of
        the initial noise and therefore change the output."""
        mu, mask, _ = sample_tensors
        torch.manual_seed(7)
        out_low = cfm_model(mu, mask, n_timesteps=2, temperature=0.1)
        torch.manual_seed(7)
        out_high = cfm_model(mu, mask, n_timesteps=2, temperature=2.0)
        assert not torch.allclose(out_low, out_high, atol=1e-5)

    def test_negative_temperature_inverts_noise(self, cfm_model, sample_tensors):
        """temperature=-1 flips the sign of the initial noise relative to
        temperature=1, so outputs should differ."""
        mu, mask, _ = sample_tensors
        torch.manual_seed(0)
        out_pos = cfm_model(mu, mask, n_timesteps=2, temperature=1.0)
        torch.manual_seed(0)
        out_neg = cfm_model(mu, mask, n_timesteps=2, temperature=-1.0)
        assert not torch.allclose(out_pos, out_neg, atol=1e-5)

    def test_same_temperature_same_seed_is_deterministic(self, cfm_model, sample_tensors):
        """Repeated calls with the same seed and temperature must match."""
        mu, mask, _ = sample_tensors
        torch.manual_seed(123)
        out_a = cfm_model(mu, mask, n_timesteps=2, temperature=0.5)
        torch.manual_seed(123)
        out_b = cfm_model(mu, mask, n_timesteps=2, temperature=0.5)
        assert torch.allclose(out_a, out_b, atol=1e-6)
