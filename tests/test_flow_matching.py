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


# ---------------------------------------------------------------------------
# Helpers for solver / loss tests
# ---------------------------------------------------------------------------


def _make_cfm(solver="euler", n_spks=1, spk_emb_dim=64, seed=None):
    """Build a small CFM; seeding lets two builds share identical weights."""
    if seed is not None:
        torch.manual_seed(seed)
    model = CFM(
        in_channels=IN_CHANNELS,
        out_channel=OUT_CHANNEL,
        cfm_params=_cfm_params(solver=solver),
        decoder_params=_decoder_params(),
        n_spks=n_spks,
        spk_emb_dim=spk_emb_dim,
    )
    model.eval()
    return model


class _TimeFieldEstimator(torch.nn.Module):
    """Stub estimator with the time-dependent field dphi/dt = 2t.

    Its exact integral over t in [0, 1] is 1.0. The midpoint rule integrates
    linear-in-t fields exactly while Euler (left endpoint) underestimates.
    The signature matches the estimator call inside solve_euler/solve_midpoint:
    estimator(x, mask, mu, t, spks, cond).
    """

    def forward(self, x, mask, mu, t, spks=None, cond=None):
        return 2.0 * t * torch.ones_like(x)


class _HalfPrecisionEstimator(torch.nn.Module):
    """Stub returning FP16 values large enough that squaring overflows FP16."""

    def forward(self, x, mask, mu, t, spks=None, cond=None):
        return torch.full_like(x, 6.0e4).half()


class _RecordingZeroEstimator(torch.nn.Module):
    """Stub that records the (y, t) it receives and predicts zeros."""

    def __init__(self):
        super().__init__()
        self.recorded_y = None
        self.recorded_t = None

    def forward(self, x, mask, mu, t, spks=None, cond=None):
        self.recorded_y = x.detach().clone()
        self.recorded_t = t.detach().clone()
        return torch.zeros_like(x)


# ---------------------------------------------------------------------------
# 6. Midpoint ODE solver and forward() dispatch
# ---------------------------------------------------------------------------


class TestSolveMidpoint:
    """solve_midpoint plus the solver dispatch inside forward()."""

    def test_forward_output_shape_and_finite(self, sample_tensors):
        model = _make_cfm(solver="midpoint")
        mu, mask, _ = sample_tensors
        out = model(mu, mask, n_timesteps=3)
        assert out.shape == (BATCH, N_FEATS, MEL_LEN)
        assert torch.isfinite(out).all()

    def test_forward_dispatches_to_solve_midpoint(self, sample_tensors, monkeypatch):
        """With solver='midpoint', forward() must route to solve_midpoint."""
        model = _make_cfm(solver="midpoint")
        mu, mask, _ = sample_tensors
        called = {}

        def fake_midpoint(z, t_span, mu, mask, spks, cond):
            called["midpoint"] = True
            return z

        monkeypatch.setattr(model, "solve_midpoint", fake_midpoint)
        model(mu, mask, n_timesteps=2)
        assert called.get("midpoint"), "forward() should dispatch to solve_midpoint"

    def test_midpoint_differs_from_euler_with_same_seed(self, sample_tensors):
        """Identically initialised models with different solvers disagree."""
        mu, mask, _ = sample_tensors
        euler_model = _make_cfm(solver="euler", seed=0)
        midpoint_model = _make_cfm(solver="midpoint", seed=0)
        torch.manual_seed(42)
        out_euler = euler_model(mu, mask, n_timesteps=2)
        torch.manual_seed(42)
        out_midpoint = midpoint_model(mu, mask, n_timesteps=2)
        assert not torch.allclose(out_euler, out_midpoint, atol=1e-5), (
            "midpoint and euler solvers should produce different outputs"
        )

    def test_single_timestep(self, sample_tensors):
        """Even a single midpoint step should produce a valid tensor."""
        model = _make_cfm(solver="midpoint")
        mu, mask, _ = sample_tensors
        out = model(mu, mask, n_timesteps=1)
        assert out.shape == (BATCH, N_FEATS, MEL_LEN)
        assert torch.isfinite(out).all()

    def test_midpoint_exact_on_linear_time_field(self, cfm_model, sample_tensors):
        """Integrating dphi/dt = 2t from x=0 over [0, 1] gives exactly 1.0
        under the midpoint rule, while Euler is visibly biased (0.75 at 4 steps)."""
        mu, mask, _ = sample_tensors
        cfm_model.estimator = _TimeFieldEstimator()
        x0 = torch.zeros(BATCH, N_FEATS, MEL_LEN)
        t_span = torch.linspace(0, 1, 5)  # 4 steps
        out_midpoint = cfm_model.solve_midpoint(x0.clone(), t_span, mu=mu, mask=mask, spks=None, cond=None)
        out_euler = cfm_model.solve_euler(x0.clone(), t_span, mu=mu, mask=mask, spks=None, cond=None)
        assert torch.allclose(out_midpoint, torch.ones_like(out_midpoint), atol=1e-5), (
            "midpoint must integrate a linear-in-t field exactly"
        )
        assert (out_euler - 1.0).abs().max().item() > 0.05, "euler should show first-order bias on dphi/dt = 2t"


# ---------------------------------------------------------------------------
# 7. Solver validation at construction time
# ---------------------------------------------------------------------------


class TestSolverValidation:
    """BASECFM.__init__ validates the solver name and defaults to euler."""

    def test_unknown_solver_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown solver 'rk4'"):
            CFM(
                in_channels=IN_CHANNELS,
                out_channel=OUT_CHANNEL,
                cfm_params=_cfm_params(solver="rk4"),
                decoder_params=_decoder_params(),
            )

    def test_missing_solver_attribute_defaults_to_euler(self, sample_tensors):
        """A params namespace without a solver attribute falls back to euler."""
        params = types.SimpleNamespace(sigma_min=1e-4)
        model = CFM(
            in_channels=IN_CHANNELS,
            out_channel=OUT_CHANNEL,
            cfm_params=params,
            decoder_params=_decoder_params(),
        )
        model.eval()
        assert model.solver == "euler"
        mu, mask, _ = sample_tensors
        out = model(mu, mask, n_timesteps=2)
        assert out.shape == (BATCH, N_FEATS, MEL_LEN)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# 8. compute_loss float32 cast (FP16 overflow safety)
# ---------------------------------------------------------------------------


class TestComputeLossF32Cast:
    """compute_loss must upcast FP16 estimator output before squaring in MSE."""

    def test_loss_dtype_is_float32(self, cfm_model, sample_tensors):
        mu, mask, x1 = sample_tensors
        cfm_model.estimator = _HalfPrecisionEstimator()
        loss, _ = cfm_model.compute_loss(x1=x1, mask=mask, mu=mu)
        assert loss.dtype == torch.float32

    def test_loss_finite_despite_fp16_overflow_values(self, cfm_model, sample_tensors):
        # Premise: squaring 6e4 overflows FP16 (max ~65504); the production
        # .float() cast in compute_loss must prevent the inf.
        assert torch.isinf(torch.tensor(6.0e4, dtype=torch.float16) ** 2)
        mu, mask, x1 = sample_tensors
        cfm_model.estimator = _HalfPrecisionEstimator()
        loss, _ = cfm_model.compute_loss(x1=x1, mask=mask, mu=mu)
        assert torch.isfinite(loss), "FP16 estimator output must not overflow the loss"


# ---------------------------------------------------------------------------
# 9. compute_loss interpolation formula and uniform timestep
# ---------------------------------------------------------------------------

T_FIXED = 0.3  # fixed uniform timestep injected via monkeypatched torch.rand


class TestInterpolationFormula:
    """compute_loss implements y = (1 - (1 - sigma_min) t) z + t x1 with uniform t."""

    def _run_patched_compute_loss(self, cfm_model, sample_tensors, monkeypatch):
        """Run compute_loss with fixed t/z and a recording zero estimator."""
        mu, mask, x1 = sample_tensors
        torch.manual_seed(0)
        z_fixed = torch.randn(BATCH, N_FEATS, MEL_LEN)

        def fake_rand(*args, **kwargs):
            shape = args[0] if len(args) == 1 and isinstance(args[0], (list, tuple, torch.Size)) else args
            return torch.full(tuple(shape), T_FIXED, device=kwargs.get("device"), dtype=kwargs.get("dtype"))

        def fake_randn_like(input_tensor, **kwargs):
            dtype = kwargs.get("dtype") or input_tensor.dtype
            return z_fixed.clone().to(dtype)

        monkeypatch.setattr(torch, "rand", fake_rand)
        monkeypatch.setattr(torch, "randn_like", fake_randn_like)
        recorder = _RecordingZeroEstimator()
        cfm_model.estimator = recorder
        loss, y = cfm_model.compute_loss(x1=x1, mask=mask, mu=mu)
        return loss, y, recorder, z_fixed, (mu, mask, x1)

    def test_y_matches_interpolation_formula(self, cfm_model, sample_tensors, monkeypatch):
        _, y, recorder, z, (_, _, x1) = self._run_patched_compute_loss(cfm_model, sample_tensors, monkeypatch)
        t = torch.full((BATCH, 1, 1), T_FIXED)
        expected_y = (1 - (1 - cfm_model.sigma_min) * t) * z + t * x1
        assert torch.allclose(recorder.recorded_y, expected_y, atol=1e-6), (
            "estimator must receive the OT-CFM interpolant y"
        )
        assert torch.allclose(y, expected_y, atol=1e-6), "compute_loss must return the interpolant y"

    def test_loss_matches_hand_computed_value(self, cfm_model, sample_tensors, monkeypatch):
        """With a zero estimator, loss = sum(u^2) / (sum(mask) * n_feats)
        where u = x1 - (1 - sigma_min) * z."""
        loss, _, _, z, (_, mask, x1) = self._run_patched_compute_loss(cfm_model, sample_tensors, monkeypatch)
        u = x1 - (1 - cfm_model.sigma_min) * z
        expected_loss = (u**2).sum() / (mask.sum() * u.shape[1])
        assert loss.item() == pytest.approx(expected_loss.item(), rel=1e-5)

    def test_timestep_passed_unwarped_to_estimator(self, cfm_model, sample_tensors, monkeypatch):
        """The t handed to the estimator is the raw uniform sample (squeezed
        from [b, 1, 1] to [b]) — no logit-normal or other warping."""
        _, _, recorder, _, _ = self._run_patched_compute_loss(cfm_model, sample_tensors, monkeypatch)
        assert recorder.recorded_t.shape == (BATCH,)
        assert torch.allclose(recorder.recorded_t, torch.full((BATCH,), T_FIXED)), (
            "timestep must be passed through unwarped (uniform)"
        )

    def test_batch_size_one_with_real_estimator(self, cfm_model):
        """b=1 makes t.squeeze() 0-dim, exercising SinusoidalPosEmb's ndim<1 branch."""
        torch.manual_seed(0)
        mu = torch.randn(1, N_FEATS, MEL_LEN)
        mask = torch.ones(1, 1, MEL_LEN)
        x1 = torch.randn(1, N_FEATS, MEL_LEN)
        loss, y = cfm_model.compute_loss(x1=x1, mask=mask, mu=mu)
        assert loss.dim() == 0
        assert torch.isfinite(loss)
        assert y.shape == (1, N_FEATS, MEL_LEN)


# ---------------------------------------------------------------------------
# 10. Multi-speaker CFM
# ---------------------------------------------------------------------------

SPK_EMB_DIM = 64


class TestMultiSpeakerCFM:
    """CFM with n_spks > 1 widens the estimator input by spk_emb_dim."""

    def test_estimator_channel_arithmetic(self):
        multi = _make_cfm(n_spks=4, spk_emb_dim=SPK_EMB_DIM)
        single = _make_cfm(n_spks=1, spk_emb_dim=SPK_EMB_DIM)
        assert multi.estimator.in_channels == IN_CHANNELS + SPK_EMB_DIM, (
            "multi-speaker estimator input must include the speaker embedding"
        )
        assert single.estimator.in_channels == IN_CHANNELS, (
            "single-speaker estimator input must not include the speaker embedding"
        )

    def test_forward_with_speaker_embedding(self, sample_tensors):
        model = _make_cfm(n_spks=4, spk_emb_dim=SPK_EMB_DIM, seed=0)
        mu, mask, _ = sample_tensors
        torch.manual_seed(1)
        spks = torch.randn(BATCH, SPK_EMB_DIM)
        out = model(mu, mask, n_timesteps=2, spks=spks)
        assert out.shape == (BATCH, N_FEATS, MEL_LEN)
        assert torch.isfinite(out).all()

    def test_compute_loss_with_speaker_embedding(self, sample_tensors):
        model = _make_cfm(n_spks=4, spk_emb_dim=SPK_EMB_DIM, seed=0)
        mu, mask, x1 = sample_tensors
        torch.manual_seed(1)
        spks = torch.randn(BATCH, SPK_EMB_DIM)
        loss, y = model.compute_loss(x1=x1, mask=mask, mu=mu, spks=spks)
        assert loss.dim() == 0
        assert torch.isfinite(loss)
        assert y.shape == (BATCH, N_FEATS, MEL_LEN)
