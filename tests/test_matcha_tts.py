"""Integration tests for the MatchaTTS model."""

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


def _build_model(n_vocab=178, n_spks=1, spk_emb_dim=64, n_feats=80):
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
    )
    return model


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
        expected_keys = {"encoder_outputs", "decoder_outputs", "attn", "mel", "mel_lengths", "rtf"}
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
