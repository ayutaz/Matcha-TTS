"""Tests for the TextEncoder component."""

from types import SimpleNamespace

import torch

from matcha.models.components.text_encoder import TextEncoder

# ---------------------------------------------------------------------------
# Shared small-model configuration helpers
# ---------------------------------------------------------------------------


def _make_encoder_params(
    n_feats=80,
    n_channels=64,
    filter_channels=64,
    n_heads=2,
    n_layers=2,
    kernel_size=3,
    p_dropout=0.1,
    prenet=True,
):
    return SimpleNamespace(
        n_feats=n_feats,
        n_channels=n_channels,
        filter_channels=filter_channels,
        n_heads=n_heads,
        n_layers=n_layers,
        kernel_size=kernel_size,
        p_dropout=p_dropout,
        prenet=prenet,
    )


def _make_duration_predictor_params(
    filter_channels_dp=64,
    kernel_size=3,
    p_dropout=0.1,
):
    return SimpleNamespace(
        filter_channels_dp=filter_channels_dp,
        kernel_size=kernel_size,
        p_dropout=p_dropout,
    )


def _build_encoder(n_vocab=178, n_spks=1, spk_emb_dim=128, prenet=True):
    """Build a small TextEncoder for testing."""
    encoder_params = _make_encoder_params(prenet=prenet)
    dp_params = _make_duration_predictor_params()
    return TextEncoder(
        encoder_type="RoPE Encoder",
        encoder_params=encoder_params,
        duration_predictor_params=dp_params,
        n_vocab=n_vocab,
        n_spks=n_spks,
        spk_emb_dim=spk_emb_dim,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTextEncoderInstantiation:
    """TextEncoder can be instantiated with various configurations."""

    def test_instantiation_default_params(self):
        encoder = _build_encoder()
        assert isinstance(encoder, TextEncoder)

    def test_instantiation_without_prenet(self):
        encoder = _build_encoder(prenet=False)
        assert isinstance(encoder, TextEncoder)
        # prenet should be a plain lambda, not a ConvReluNorm
        assert not isinstance(encoder.prenet, torch.nn.Module)

    def test_instantiation_multispeaker(self):
        encoder = _build_encoder(n_spks=2, spk_emb_dim=64)
        assert isinstance(encoder, TextEncoder)
        assert encoder.n_spks == 2

    def test_stored_attributes(self):
        encoder = _build_encoder()
        assert encoder.n_vocab == 178
        assert encoder.n_feats == 80
        assert encoder.n_channels == 64
        assert encoder.n_spks == 1

    def test_embedding_shape(self):
        encoder = _build_encoder()
        assert encoder.emb.num_embeddings == 178
        assert encoder.emb.embedding_dim == 64


class TestTextEncoderForward:
    """Forward pass produces correct output shapes and types."""

    BATCH_SIZE = 2
    SEQ_LEN = 10
    N_FEATS = 80
    N_CHANNELS = 64

    def _run_forward(self, encoder, batch_size=None, seq_len=None):
        batch_size = batch_size or self.BATCH_SIZE
        seq_len = seq_len or self.SEQ_LEN
        x = torch.randint(0, 178, (batch_size, seq_len))
        x_lengths = torch.full((batch_size,), seq_len, dtype=torch.long)
        encoder.eval()
        with torch.no_grad():
            return encoder(x, x_lengths)

    def test_forward_returns_three_tensors(self):
        encoder = _build_encoder()
        outputs = self._run_forward(encoder)
        assert len(outputs) == 3

    def test_mu_shape(self):
        encoder = _build_encoder()
        mu, _logw, _x_mask = self._run_forward(encoder)
        assert mu.shape == (self.BATCH_SIZE, self.N_FEATS, self.SEQ_LEN)

    def test_logw_shape(self):
        encoder = _build_encoder()
        _mu, logw, _x_mask = self._run_forward(encoder)
        assert logw.shape == (self.BATCH_SIZE, 1, self.SEQ_LEN)

    def test_x_mask_shape(self):
        encoder = _build_encoder()
        _mu, _logw, x_mask = self._run_forward(encoder)
        assert x_mask.shape == (self.BATCH_SIZE, 1, self.SEQ_LEN)

    def test_x_mask_values_all_ones_for_equal_lengths(self):
        encoder = _build_encoder()
        _mu, _logw, x_mask = self._run_forward(encoder)
        assert torch.all(x_mask == 1.0)

    def test_output_dtypes_are_float(self):
        encoder = _build_encoder()
        mu, logw, x_mask = self._run_forward(encoder)
        assert mu.dtype == torch.float32
        assert logw.dtype == torch.float32
        assert x_mask.dtype == torch.float32

    def test_forward_without_prenet(self):
        encoder = _build_encoder(prenet=False)
        mu, logw, x_mask = self._run_forward(encoder)
        assert mu.shape == (self.BATCH_SIZE, self.N_FEATS, self.SEQ_LEN)
        assert logw.shape == (self.BATCH_SIZE, 1, self.SEQ_LEN)
        assert x_mask.shape == (self.BATCH_SIZE, 1, self.SEQ_LEN)


class TestTextEncoderOutputDimensions:
    """Output shapes match expected dimensions for various inputs."""

    def test_single_sample(self):
        encoder = _build_encoder()
        x = torch.randint(0, 178, (1, 5))
        x_lengths = torch.tensor([5])
        encoder.eval()
        with torch.no_grad():
            mu, logw, x_mask = encoder(x, x_lengths)
        assert mu.shape == (1, 80, 5)
        assert logw.shape == (1, 1, 5)
        assert x_mask.shape == (1, 1, 5)

    def test_variable_lengths_within_batch(self):
        encoder = _build_encoder()
        batch_size, max_len = 3, 12
        x = torch.randint(0, 178, (batch_size, max_len))
        x_lengths = torch.tensor([8, 12, 5])
        encoder.eval()
        with torch.no_grad():
            mu, logw, x_mask = encoder(x, x_lengths)
        # Outputs are padded to max_len
        assert mu.shape == (batch_size, 80, max_len)
        assert logw.shape == (batch_size, 1, max_len)
        assert x_mask.shape == (batch_size, 1, max_len)

    def test_mask_reflects_variable_lengths(self):
        encoder = _build_encoder()
        x = torch.randint(0, 178, (2, 8))
        x_lengths = torch.tensor([4, 8])
        encoder.eval()
        with torch.no_grad():
            _mu, _logw, x_mask = encoder(x, x_lengths)
        # First sample: positions 0..3 unmasked, 4..7 masked
        assert torch.all(x_mask[0, 0, :4] == 1.0)
        assert torch.all(x_mask[0, 0, 4:] == 0.0)
        # Second sample: all positions unmasked
        assert torch.all(x_mask[1, 0, :] == 1.0)

    def test_masked_positions_produce_zero_mu(self):
        encoder = _build_encoder()
        x = torch.randint(0, 178, (1, 10))
        x_lengths = torch.tensor([6])
        encoder.eval()
        with torch.no_grad():
            mu, _logw, _x_mask = encoder(x, x_lengths)
        # mu is multiplied by x_mask, so padded positions should be zero
        assert torch.all(mu[:, :, 6:] == 0.0)

    def test_masked_positions_produce_zero_logw(self):
        encoder = _build_encoder()
        x = torch.randint(0, 178, (1, 10))
        x_lengths = torch.tensor([6])
        encoder.eval()
        with torch.no_grad():
            _mu, logw, _x_mask = encoder(x, x_lengths)
        # logw is multiplied by x_mask inside DurationPredictor
        assert torch.all(logw[:, :, 6:] == 0.0)

    def test_different_n_feats(self):
        encoder_params = _make_encoder_params(n_feats=40)
        dp_params = _make_duration_predictor_params()
        encoder = TextEncoder(
            encoder_type="RoPE Encoder",
            encoder_params=encoder_params,
            duration_predictor_params=dp_params,
            n_vocab=178,
        )
        x = torch.randint(0, 178, (2, 10))
        x_lengths = torch.tensor([10, 10])
        encoder.eval()
        with torch.no_grad():
            mu, logw, x_mask = encoder(x, x_lengths)
        assert mu.shape == (2, 40, 10)
        assert logw.shape == (2, 1, 10)
