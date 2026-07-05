"""Integration tests for the MatchaTTS model."""

import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn.functional as F

from matcha.models import matcha_tts as matcha_tts_module
from matcha.models.matcha_tts import LOG_2PI, MatchaTTS
from matcha.utils.model import fix_len_compatibility, generate_path, sequence_mask


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
    """Build a minimal encoder config using SimpleNamespace to mimic Hydra DictConfig."""
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
    """Build a minimal decoder config dict for the Decoder (U-Net)."""
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
    """Build a minimal CFM config."""
    return SimpleNamespace(
        name="CFM",
        solver="euler",
        sigma_min=1e-4,
    )


def _build_model(n_vocab=178, n_spks=1, spk_emb_dim=64, n_feats=80, **model_kwargs):
    """Instantiate a MatchaTTS model with minimal parameters for testing."""
    encoder = _make_encoder_config(
        n_feats=n_feats,
        n_channels=64,
        filter_channels=128,
        filter_channels_dp=64,
        n_heads=2,
        n_layers=1,
        kernel_size=3,
        p_dropout=0.0,
        spk_emb_dim=spk_emb_dim,
        n_spks=n_spks,
        prenet=True,
    )
    decoder = _make_decoder_config()
    cfm = _make_cfm_config()
    data_statistics = {"mel_mean": 0.0, "mel_std": 1.0}

    model = MatchaTTS(
        n_vocab=n_vocab,
        n_spks=n_spks,
        spk_emb_dim=spk_emb_dim,
        n_feats=n_feats,
        encoder=encoder,
        decoder=decoder,
        cfm=cfm,
        data_statistics=data_statistics,
        out_size=None,
        **model_kwargs,
    )
    return model


def _make_training_batch(x_lengths=(6, 4), y_lengths=(16, 12), n_vocab=178, n_feats=80):
    """Create a padded (x, x_lengths, y, y_lengths) training batch with synthetic data."""
    x_lengths = torch.tensor(x_lengths, dtype=torch.long)
    y_lengths = torch.tensor(y_lengths, dtype=torch.long)
    batch_size = x_lengths.shape[0]
    x = torch.randint(1, n_vocab, (batch_size, int(x_lengths.max().item())))
    for i in range(batch_size):
        x[i, x_lengths[i] :] = 0
    y = torch.randn(batch_size, n_feats, int(y_lengths.max().item()))
    return x, x_lengths, y, y_lengths


def _make_durations(x_lengths, y_lengths):
    """Create int64 durations of shape (batch, max_text_length) whose valid row sums equal y_lengths."""
    batch_size = x_lengths.shape[0]
    max_text_length = int(x_lengths.max().item())
    durations = torch.zeros(batch_size, max_text_length, dtype=torch.long)
    for i in range(batch_size):
        n_tokens = int(x_lengths[i].item())
        total = int(y_lengths[i].item())
        base = total // n_tokens
        row = torch.full((n_tokens,), base, dtype=torch.long)
        row[: total - base * n_tokens] += 1
        durations[i, :n_tokens] = row
    return durations


def _assert_finite_scalar_losses(*losses):
    """Assert each loss is a finite 0-dim tensor."""
    for loss in losses:
        assert torch.is_tensor(loss)
        assert loss.dim() == 0
        assert torch.isfinite(loss)


@pytest.mark.slow
class TestMatchaTTSInstantiation:
    """Tests for model instantiation with various configurations."""

    def test_single_speaker_instantiation(self):
        """MatchaTTS can be instantiated in single-speaker mode."""
        model = _build_model(n_spks=1)
        assert isinstance(model, MatchaTTS)
        assert model.n_vocab == 178
        assert model.n_feats == 80
        assert model.n_spks == 1

    def test_multi_speaker_instantiation(self):
        """MatchaTTS can be instantiated in multi-speaker mode with speaker embedding."""
        model = _build_model(n_spks=4, spk_emb_dim=64)
        assert isinstance(model, MatchaTTS)
        assert model.n_spks == 4
        assert hasattr(model, "spk_emb")
        assert model.spk_emb.num_embeddings == 4
        assert model.spk_emb.embedding_dim == 64

    def test_no_speaker_embedding_for_single_speaker(self):
        """Single-speaker model should not have a speaker embedding layer."""
        model = _build_model(n_spks=1)
        assert not hasattr(model, "spk_emb")

    def test_data_statistics_buffers(self):
        """Model should register mel_mean and mel_std as buffers."""
        model = _build_model()
        assert hasattr(model, "mel_mean")
        assert hasattr(model, "mel_std")
        assert model.mel_mean.item() == 0.0
        assert model.mel_std.item() == 1.0

    def test_submodules_exist(self):
        """Model should have encoder and decoder submodules."""
        model = _build_model()
        assert hasattr(model, "encoder")
        assert hasattr(model, "decoder")


@pytest.mark.slow
class TestMatchaTTSSynthesise:
    """Tests for the synthesise() method."""

    @pytest.fixture
    def model(self):
        model = _build_model(n_spks=1)
        model.eval()
        return model

    def test_synthesise_returns_expected_keys(self, model):
        """synthesise() output dict must contain all expected keys."""
        x = torch.randint(0, 178, (1, 10))
        x_lengths = torch.tensor([10])
        output = model.synthesise(x, x_lengths, n_timesteps=2)
        expected_keys = {"encoder_outputs", "decoder_outputs", "attn", "mel", "mel_lengths", "rtf", "durations"}
        assert set(output.keys()) == expected_keys

    def test_synthesise_mel_shape(self, model):
        """Output mel should have shape (batch, n_feats, mel_length)."""
        x = torch.randint(0, 178, (1, 12))
        x_lengths = torch.tensor([12])
        output = model.synthesise(x, x_lengths, n_timesteps=2)
        mel = output["mel"]
        assert mel.dim() == 3
        assert mel.shape[0] == 1
        assert mel.shape[1] == 80  # n_feats

    def test_synthesise_encoder_decoder_same_shape(self, model):
        """encoder_outputs and decoder_outputs should share the same shape."""
        x = torch.randint(0, 178, (1, 8))
        x_lengths = torch.tensor([8])
        output = model.synthesise(x, x_lengths, n_timesteps=2)
        assert output["encoder_outputs"].shape == output["decoder_outputs"].shape

    def test_synthesise_mel_lengths_matches_output(self, model):
        """mel_lengths should be consistent with the mel time dimension."""
        x = torch.randint(0, 178, (1, 10))
        x_lengths = torch.tensor([10])
        output = model.synthesise(x, x_lengths, n_timesteps=2)
        mel_len = output["mel_lengths"].item()
        assert mel_len > 0
        assert output["mel"].shape[2] == mel_len

    def test_synthesise_rtf_is_positive_float(self, model):
        """Real-time factor should be a positive number."""
        x = torch.randint(0, 178, (1, 5))
        x_lengths = torch.tensor([5])
        output = model.synthesise(x, x_lengths, n_timesteps=2)
        assert isinstance(output["rtf"], float)
        assert output["rtf"] > 0

    def test_synthesise_mel_is_finite(self, model):
        """Output mel should not contain NaN or Inf values."""
        x = torch.randint(0, 178, (1, 10))
        x_lengths = torch.tensor([10])
        output = model.synthesise(x, x_lengths, n_timesteps=2)
        assert torch.isfinite(output["mel"]).all()

    def test_synthesise_batch(self, model):
        """synthesise() should handle a batch of sequences with varying lengths."""
        x = torch.randint(0, 178, (2, 12))
        x_lengths = torch.tensor([12, 8])
        output = model.synthesise(x, x_lengths, n_timesteps=2)
        assert output["mel"].shape[0] == 2
        assert output["mel_lengths"].shape[0] == 2


@pytest.mark.slow
class TestMatchaTTSEvalMode:
    """Tests for model behavior in eval mode on CPU."""

    def test_eval_mode_on_cpu(self):
        """Model should run in eval mode on CPU without errors."""
        model = _build_model()
        model.eval()
        assert not model.training

        x = torch.randint(0, 178, (1, 6))
        x_lengths = torch.tensor([6])
        output = model.synthesise(x, x_lengths, n_timesteps=2)
        assert output["mel"].device.type == "cpu"

    def test_no_grad_during_synthesise(self):
        """synthesise() should not accumulate gradients (inference_mode)."""
        model = _build_model()
        model.eval()
        x = torch.randint(0, 178, (1, 6))
        x_lengths = torch.tensor([6])
        output = model.synthesise(x, x_lengths, n_timesteps=2)
        assert not output["mel"].requires_grad

    def test_temperature_scaling(self):
        """Different temperature values should produce different outputs."""
        model = _build_model()
        model.eval()

        torch.manual_seed(42)
        x = torch.randint(0, 178, (1, 8))
        x_lengths = torch.tensor([8])

        torch.manual_seed(0)
        out_low = model.synthesise(x, x_lengths, n_timesteps=2, temperature=0.1)
        torch.manual_seed(0)
        out_high = model.synthesise(x, x_lengths, n_timesteps=2, temperature=2.0)

        # Both should be valid mels with the same shape, but different values
        assert out_low["mel"].shape == out_high["mel"].shape
        assert not torch.allclose(out_low["mel"], out_high["mel"])


@pytest.mark.slow
class TestMatchaTTSMultiSpeaker:
    """Tests for multi-speaker model."""

    def test_multispeaker_synthesise(self):
        """Multi-speaker model should accept speaker ids and produce output."""
        model = _build_model(n_spks=4, spk_emb_dim=64)
        model.eval()

        x = torch.randint(0, 178, (1, 8))
        x_lengths = torch.tensor([8])
        spks = torch.tensor([2])

        output = model.synthesise(x, x_lengths, n_timesteps=2, spks=spks)
        assert output["mel"].shape[0] == 1
        assert output["mel"].shape[1] == 80


@pytest.mark.slow
class TestPrecomputedDurationsForward:
    """Tests for the use_precomputed_durations branch of forward() (MAS bypass)."""

    def _batch_with_durations(self):
        x, x_lengths, y, y_lengths = _make_training_batch(x_lengths=(6, 4), y_lengths=(16, 12))
        durations = _make_durations(x_lengths, y_lengths)
        return x, x_lengths, y, y_lengths, durations

    def test_forward_with_2d_durations(self):
        """int64 durations shaped (B, T_text) must drive the alignment exactly and give finite losses."""
        torch.manual_seed(0)
        model = _build_model(use_precomputed_durations=True)
        x, x_lengths, y, y_lengths, durations = self._batch_with_durations()

        dur_loss, prior_loss, diff_loss, attn = model(x, x_lengths, y, y_lengths, durations=durations)

        _assert_finite_scalar_losses(dur_loss, prior_loss, diff_loss)
        # The per-token frame counts implied by the generated alignment equal the input durations
        assert torch.equal(attn.sum(dim=-1).long(), durations)

    def test_forward_with_3d_durations(self):
        """Durations shaped (B, 1, T_text) are squeezed and produce the same alignment."""
        torch.manual_seed(0)
        model = _build_model(use_precomputed_durations=True)
        x, x_lengths, y, y_lengths, durations = self._batch_with_durations()

        dur_loss, prior_loss, diff_loss, attn = model(x, x_lengths, y, y_lengths, durations=durations.unsqueeze(1))

        _assert_finite_scalar_losses(dur_loss, prior_loss, diff_loss)
        assert torch.equal(attn.sum(dim=-1).long(), durations)

    def test_forward_rejects_1d_durations(self):
        """A 1D durations tensor violates the (B, T_text) contract and must raise AssertionError."""
        torch.manual_seed(0)
        model = _build_model(use_precomputed_durations=True)
        x, x_lengths, y, y_lengths, durations = self._batch_with_durations()

        with pytest.raises(AssertionError):
            model(x, x_lengths, y, y_lengths, durations=durations[0])

    def test_forward_rejects_4d_durations(self):
        """A 4D durations tensor is not squeezed down and must raise AssertionError."""
        torch.manual_seed(0)
        model = _build_model(use_precomputed_durations=True)
        x, x_lengths, y, y_lengths, durations = self._batch_with_durations()

        with pytest.raises(AssertionError):
            model(x, x_lengths, y, y_lengths, durations=durations[:, None, None, :])

    def test_forward_requires_durations(self):
        """durations=None fails on .float() — characterizes the get_losses batch contract."""
        torch.manual_seed(0)
        model = _build_model(use_precomputed_durations=True)
        x, x_lengths, y, y_lengths, _ = self._batch_with_durations()

        with pytest.raises(AttributeError):
            model(x, x_lengths, y, y_lengths, durations=None)


@pytest.mark.slow
class TestOutSizeCropping:
    """Tests for the out_size mel-segment cropping branch of forward().

    Regression guard for the documented torch.empty -> torch.zeros NaN bug in the
    attn_cut / y_cut buffers (uninitialized memory previously leaked NaNs into losses).
    """

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_cropping_losses_finite_across_seeds(self, seed):
        """Mixed batch (one item longer, one shorter than out_size) never yields NaN losses."""
        torch.manual_seed(seed)
        model = _build_model()
        x, x_lengths, y, y_lengths = _make_training_batch(x_lengths=(8, 5), y_lengths=(28, 12))
        out_size = fix_len_compatibility(20)

        dur_loss, prior_loss, diff_loss, attn = model(x, x_lengths, y, y_lengths, out_size=out_size)

        _assert_finite_scalar_losses(dur_loss, prior_loss, diff_loss)
        assert attn.shape[-1] == out_size

    def test_out_size_equal_to_max_length(self):
        """out_size == max(y_lengths) (max_offset == 0) must not crash and gives finite losses."""
        torch.manual_seed(0)
        model = _build_model()
        x, x_lengths, y, y_lengths = _make_training_batch(x_lengths=(8, 5), y_lengths=(28, 12))
        out_size = fix_len_compatibility(int(y_lengths.max().item()))
        assert out_size == 28

        dur_loss, prior_loss, diff_loss, attn = model(x, x_lengths, y, y_lengths, out_size=out_size)

        _assert_finite_scalar_losses(dur_loss, prior_loss, diff_loss)
        assert attn.shape[-1] == out_size


@pytest.mark.slow
class TestLogPriorEquivalence:
    """Tests for the refactored MAS log-prior computation in forward()."""

    def _run_forward_with_spies(self, monkeypatch):
        """Run forward() while recording mu_x (encoder hook) and maximum_path arguments/output."""
        torch.manual_seed(0)
        model = _build_model()
        x, x_lengths, y, y_lengths = _make_training_batch(x_lengths=(8, 5), y_lengths=(20, 12))

        captured = {}

        def encoder_hook(module, args, output):
            captured["mu_x"] = output[0].detach().clone()

        handle = model.encoder.register_forward_hook(encoder_hook)

        real_maximum_path = matcha_tts_module.monotonic_align.maximum_path
        recorded = {}

        def spy_maximum_path(value, mask):
            recorded["log_prior"] = value.detach().clone()
            path = real_maximum_path(value, mask)
            recorded["attn"] = path.detach().clone()
            return path

        monkeypatch.setattr(matcha_tts_module.monotonic_align, "maximum_path", spy_maximum_path)
        try:
            losses = model(x, x_lengths, y, y_lengths)
        finally:
            handle.remove()
        return x_lengths, y, y_lengths, captured, recorded, losses

    def test_log_prior_matches_gaussian_formula(self, monkeypatch):
        """log_prior passed to MAS equals -0.5*||y - mu_x||^2 - 0.5*n_feats*log(2*pi) at valid positions."""
        x_lengths, y, y_lengths, captured, recorded, losses = self._run_forward_with_spies(monkeypatch)
        assert "log_prior" in recorded

        mu_x = captured["mu_x"]  # (B, n_feats, T_text)
        n_feats = mu_x.shape[1]
        # diff[b, f, i, j] = y[b, f, j] - mu_x[b, f, i]
        diff = y.unsqueeze(2) - mu_x.unsqueeze(-1)
        expected = -0.5 * (diff**2).sum(dim=1) - 0.5 * n_feats * LOG_2PI  # (B, T_text, T_mel)

        valid = torch.zeros_like(expected, dtype=torch.bool)
        for b in range(expected.shape[0]):
            valid[b, : x_lengths[b], : y_lengths[b]] = True
        assert (recorded["log_prior"][valid] - expected[valid]).abs().max().item() < 1e-4

        dur_loss, prior_loss, diff_loss, _ = losses
        _assert_finite_scalar_losses(dur_loss, prior_loss, diff_loss)

    def test_mas_alignment_is_valid_monotonic_path(self, monkeypatch):
        """MAS output is a binary monotonic path covering exactly y_lengths frames per item."""
        x_lengths, _, y_lengths, _, recorded, _ = self._run_forward_with_spies(monkeypatch)
        attn = recorded["attn"]  # (B, T_text, T_mel)

        assert ((attn == 0) | (attn == 1)).all()
        for b in range(attn.shape[0]):
            x_len = int(x_lengths[b].item())
            y_len = int(y_lengths[b].item())
            # Each valid mel frame is assigned to exactly one text token
            assert torch.equal(attn[b, :, :y_len].sum(dim=0), torch.ones(y_len))
            # Nothing outside the valid mel/text region
            assert attn[b, :, y_len:].sum() == 0
            assert attn[b, x_len:, :].sum() == 0
            # Total mass equals the mel length
            assert attn[b].sum() == y_len
            # Monotonic: assigned token index never decreases over time
            token_idx = attn[b, :, :y_len].argmax(dim=0)
            assert (token_idx[1:] >= token_idx[:-1]).all()


@pytest.mark.slow
class TestPriorLossFlag:
    """Tests for the prior_loss flag and its computed value in forward()."""

    def test_prior_loss_disabled_returns_zero(self):
        """With prior_loss=False the returned prior loss is exactly 0."""
        torch.manual_seed(0)
        model = _build_model(prior_loss=False, use_precomputed_durations=True)
        x, x_lengths, y, y_lengths = _make_training_batch(x_lengths=(6, 4), y_lengths=(16, 12))
        durations = _make_durations(x_lengths, y_lengths)

        dur_loss, prior_loss, diff_loss, _ = model(x, x_lengths, y, y_lengths, durations=durations)

        assert prior_loss == 0
        _assert_finite_scalar_losses(dur_loss, diff_loss)

    def test_prior_loss_value_matches_manual_computation(self):
        """With deterministic (precomputed) alignment, prior loss equals the hand-computed value."""
        torch.manual_seed(0)
        model = _build_model(use_precomputed_durations=True)
        x, x_lengths, y, y_lengths = _make_training_batch(x_lengths=(6, 4), y_lengths=(16, 12))
        durations = _make_durations(x_lengths, y_lengths)

        captured = {}

        def encoder_hook(module, args, output):
            captured["mu_x"] = output[0].detach().clone()
            captured["x_mask"] = output[2].detach().clone()

        handle = model.encoder.register_forward_hook(encoder_hook)
        try:
            _, prior_loss, _, _ = model(x, x_lengths, y, y_lengths, durations=durations)
        finally:
            handle.remove()

        x_mask = captured["x_mask"]
        y_mask = sequence_mask(y_lengths, y.shape[-1]).unsqueeze(1).to(x_mask)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
        attn = generate_path(durations.float(), attn_mask.squeeze(1))
        mu_y = torch.matmul(attn.transpose(1, 2), captured["mu_x"].transpose(1, 2)).transpose(1, 2)
        masked_n = torch.sum(y_mask) * model.n_feats
        expected = 0.5 * (F.mse_loss(y * y_mask, mu_y * y_mask, reduction="sum") / masked_n + LOG_2PI)

        assert prior_loss.item() == pytest.approx(expected.item(), rel=1e-4)


@pytest.mark.slow
class TestMultiSpeakerTrainingForward:
    """Tests for the multi-speaker training path of forward()."""

    def test_mas_forward_with_speaker_ids(self):
        """Multi-speaker forward through the MAS branch produces finite losses."""
        torch.manual_seed(0)
        model = _build_model(n_spks=4, spk_emb_dim=64)
        x, x_lengths, y, y_lengths = _make_training_batch(x_lengths=(6, 4), y_lengths=(16, 12))
        spks = torch.tensor([0, 3], dtype=torch.long)

        dur_loss, prior_loss, diff_loss, _ = model(x, x_lengths, y, y_lengths, spks=spks)

        _assert_finite_scalar_losses(dur_loss, prior_loss, diff_loss)

    def test_precomputed_durations_forward_with_speaker_ids(self):
        """Multi-speaker forward through the precomputed-durations branch produces finite losses."""
        torch.manual_seed(0)
        model = _build_model(n_spks=4, spk_emb_dim=64, use_precomputed_durations=True)
        x, x_lengths, y, y_lengths = _make_training_batch(x_lengths=(6, 4), y_lengths=(16, 12))
        durations = _make_durations(x_lengths, y_lengths)
        spks = torch.tensor([0, 3], dtype=torch.long)

        dur_loss, prior_loss, diff_loss, _ = model(x, x_lengths, y, y_lengths, spks=spks, durations=durations)

        _assert_finite_scalar_losses(dur_loss, prior_loss, diff_loss)

    def test_float_speaker_ids_raise(self):
        """forward() does not cast spks: float32 ids break the embedding lookup (dtype contract)."""
        torch.manual_seed(0)
        model = _build_model(n_spks=4, spk_emb_dim=64)
        x, x_lengths, y, y_lengths = _make_training_batch(x_lengths=(6, 4), y_lengths=(16, 12))
        spks = torch.tensor([0.0, 3.0], dtype=torch.float32)

        with pytest.raises(RuntimeError):
            model(x, x_lengths, y, y_lengths, spks=spks)


@pytest.mark.slow
class TestClampBoundaryBlanks:
    """Tests for the clamp_boundary_blanks option of synthesise()."""

    def _model_with_constant_logw(self, monkeypatch, duration=100.0):
        """Build an eval model whose encoder predicts a constant duration everywhere."""
        model = _build_model()
        model.eval()
        real_forward = model.encoder.forward

        def constant_logw_forward(x, x_lengths, spks=None):
            mu_x, logw, x_mask = real_forward(x, x_lengths, spks)
            return mu_x, torch.full_like(logw, math.log(duration)), x_mask

        monkeypatch.setattr(model.encoder, "forward", constant_logw_forward)
        return model

    def test_clamp_enabled_clamps_per_item_boundaries(self, monkeypatch):
        """First/last valid tokens are clamped to <=3 frames per item; interior tokens keep ~100."""
        torch.manual_seed(0)
        model = self._model_with_constant_logw(monkeypatch)
        x = torch.randint(1, 178, (2, 6))
        x_lengths = torch.tensor([6, 4])

        output = model.synthesise(x, x_lengths, n_timesteps=2, clamp_boundary_blanks=True)
        durations = output["durations"]

        # First token of each item is clamped
        assert durations[0, 0] <= 3
        assert durations[1, 0] <= 3
        # Last token is clamped at each item's own x_lengths[b]-1, not the padded max index
        assert durations[0, 5] <= 3
        assert durations[1, 3] <= 3
        # Interior tokens keep the raw ~100-frame prediction
        assert (durations[0, 1:5] >= 99).all()
        assert (durations[1, 1:3] >= 99).all()
        # Padded positions of the short item stay masked at zero
        assert (durations[1, 4:] == 0).all()

    def test_clamp_disabled_keeps_boundary_durations(self, monkeypatch):
        """With clamp_boundary_blanks=False the boundary tokens keep the raw ~100-frame prediction."""
        torch.manual_seed(0)
        model = self._model_with_constant_logw(monkeypatch)
        x = torch.randint(1, 178, (2, 6))
        x_lengths = torch.tensor([6, 4])

        output = model.synthesise(x, x_lengths, n_timesteps=2, clamp_boundary_blanks=False)
        durations = output["durations"]

        assert durations[0, 0] >= 99
        assert durations[0, 5] >= 99
        assert durations[1, 0] >= 99
        assert durations[1, 3] >= 99
        # Padded positions remain masked regardless of the clamp flag
        assert (durations[1, 4:] == 0).all()


@pytest.mark.slow
class TestSynthesiseDurationInvariant:
    """Tests for the durations/mel_lengths invariant of synthesise()."""

    def test_durations_sum_matches_mel_lengths(self):
        """At length_scale=1.0, per-item duration sums equal the reported mel lengths."""
        torch.manual_seed(0)
        model = _build_model()
        model.eval()
        x = torch.randint(1, 178, (2, 10))
        x_lengths = torch.tensor([10, 7])

        output = model.synthesise(x, x_lengths, n_timesteps=2, length_scale=1.0)

        assert torch.equal(output["durations"].sum(dim=1).long(), output["mel_lengths"])

    def test_length_scale_two_doubles_mel_lengths(self):
        """length_scale=2.0 doubles the mel lengths relative to length_scale=1.0."""
        torch.manual_seed(0)
        model = _build_model()
        model.eval()
        x = torch.randint(1, 178, (2, 10))
        x_lengths = torch.tensor([10, 7])

        out_normal = model.synthesise(x, x_lengths, n_timesteps=2, length_scale=1.0)
        out_slow = model.synthesise(x, x_lengths, n_timesteps=2, length_scale=2.0)

        assert torch.equal(out_slow["mel_lengths"], out_normal["mel_lengths"] * 2)


class TestEnableGradientCheckpointing:
    """Tests for enable_gradient_checkpointing() delegation to the decoder's estimator."""

    def _model_with_decoder(self, decoder):
        """Build a model and replace its decoder submodule with an arbitrary object."""
        torch.manual_seed(0)
        model = _build_model()
        del model.decoder  # bypass nn.Module submodule type checks
        model.decoder = decoder
        return model

    def test_delegates_to_estimator_exactly_once(self):
        """The estimator hook is called exactly once when it exists."""
        estimator = MagicMock(spec=["enable_gradient_checkpointing"])
        model = self._model_with_decoder(SimpleNamespace(estimator=estimator))

        model.enable_gradient_checkpointing()

        estimator.enable_gradient_checkpointing.assert_called_once_with()

    def test_estimator_without_hook_is_silent_noop(self):
        """An estimator without the hook attribute results in a silent no-op."""
        model = self._model_with_decoder(SimpleNamespace(estimator=object()))
        model.enable_gradient_checkpointing()  # must not raise

    def test_decoder_without_estimator_is_silent_noop(self):
        """A decoder without an estimator attribute results in a silent no-op."""
        model = self._model_with_decoder(object())
        model.enable_gradient_checkpointing()  # must not raise
