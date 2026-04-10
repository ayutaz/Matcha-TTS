"""Tests for the TextEncoder component."""

from types import SimpleNamespace

import torch
import torch.nn as nn

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


class TestDurationPredictorFiLM:
    """FiLM speaker conditioning in DurationPredictor."""

    def test_dp_film_instantiation_multispeaker(self):
        """n_spks=2でFiLM層が生成されること"""
        from matcha.models.components.text_encoder import DurationPredictor
        dp = DurationPredictor(128, 64, 3, 0.1, n_spks=2, spk_emb_dim=64)
        assert hasattr(dp, 'film_1')
        assert hasattr(dp, 'film_2')

    def test_dp_no_film_single_speaker(self):
        """n_spks=1でFiLM層が生成されないこと"""
        from matcha.models.components.text_encoder import DurationPredictor
        dp = DurationPredictor(128, 64, 3, 0.1, n_spks=1)
        assert not hasattr(dp, 'film_1')

    def test_dp_film_identity_init(self):
        """identity init: gamma=1, beta=0"""
        from matcha.models.components.text_encoder import DurationPredictor
        dp = DurationPredictor(128, 64, 3, 0.1, n_spks=2, spk_emb_dim=64)
        spks = torch.randn(2, 64)
        out = dp.film_1(spks)
        gamma, beta = out.chunk(2, dim=-1)
        assert torch.allclose(gamma, torch.ones_like(gamma))
        assert torch.allclose(beta, torch.zeros_like(beta))

    def test_dp_film_output_shape(self):
        """FiLM付きDP出力shape"""
        from matcha.models.components.text_encoder import DurationPredictor
        dp = DurationPredictor(128, 64, 3, 0.1, n_spks=2, spk_emb_dim=64)
        dp.eval()
        x = torch.randn(2, 128, 10)
        x_mask = torch.ones(2, 1, 10)
        spks = torch.randn(2, 64)
        with torch.no_grad():
            out = dp(x, x_mask, spks=spks)
        assert out.shape == (2, 1, 10)

    def test_dp_film_identity_preserves_output(self):
        """identity init状態でFiLMあり/なしの出力が一致"""
        from matcha.models.components.text_encoder import DurationPredictor
        torch.manual_seed(42)
        dp_no_film = DurationPredictor(64, 64, 3, 0.1, n_spks=1)
        torch.manual_seed(42)
        dp_film = DurationPredictor(64, 64, 3, 0.1, n_spks=2, spk_emb_dim=64)
        # conv/norm/projの重みをコピー
        dp_film.conv_1.load_state_dict(dp_no_film.conv_1.state_dict())
        dp_film.norm_1.load_state_dict(dp_no_film.norm_1.state_dict())
        dp_film.conv_2.load_state_dict(dp_no_film.conv_2.state_dict())
        dp_film.norm_2.load_state_dict(dp_no_film.norm_2.state_dict())
        dp_film.proj.load_state_dict(dp_no_film.proj.state_dict())

        dp_no_film.eval()
        dp_film.eval()
        x = torch.randn(2, 64, 10)
        x_mask = torch.ones(2, 1, 10)
        spks = torch.randn(2, 64)
        with torch.no_grad():
            out_no = dp_no_film(x, x_mask)
            out_yes = dp_film(x, x_mask, spks=spks)
        assert torch.allclose(out_no, out_yes, rtol=1e-5, atol=1e-5)

    def test_dp_film_different_speakers_different_output(self):
        """異なるspksで異なる出力（ランダム重み後）"""
        from matcha.models.components.text_encoder import DurationPredictor
        dp = DurationPredictor(64, 64, 3, 0.1, n_spks=2, spk_emb_dim=64)
        # ランダム重みでFiLMを意味のある変換にする
        nn.init.normal_(dp.film_1.weight)
        nn.init.normal_(dp.film_2.weight)
        dp.eval()
        x = torch.randn(1, 64, 10)
        x_mask = torch.ones(1, 1, 10)
        spk_a = torch.randn(1, 64)
        spk_b = torch.randn(1, 64)
        with torch.no_grad():
            out_a = dp(x, x_mask, spks=spk_a)
            out_b = dp(x, x_mask, spks=spk_b)
        assert not torch.allclose(out_a, out_b)

    def test_dp_film_parameter_count(self):
        """FiLMのパラメータ数: 2 * (spk_emb_dim * filter_channels*2 + filter_channels*2)"""
        from matcha.models.components.text_encoder import DurationPredictor
        dp = DurationPredictor(256, 256, 3, 0.1, n_spks=2, spk_emb_dim=64)
        film_params = sum(p.numel() for n, p in dp.named_parameters() if 'film' in n)
        # film_1: 64*512 + 512 = 33280, film_2: same = 33280, total = 66560
        assert film_params == 66560

    def test_dp_forward_without_spks_multispeaker(self):
        """n_spks=2でspks=None→FiLMスキップ、エラーなし"""
        from matcha.models.components.text_encoder import DurationPredictor
        dp = DurationPredictor(64, 64, 3, 0.1, n_spks=2, spk_emb_dim=64)
        dp.eval()
        x = torch.randn(2, 64, 10)
        x_mask = torch.ones(2, 1, 10)
        with torch.no_grad():
            out = dp(x, x_mask, spks=None)
        assert out.shape == (2, 1, 10)


class TestTextEncoderFiLMIntegration:
    """TextEncoder integration with FiLM-enabled DurationPredictor."""

    def test_encoder_passes_spks_to_dp(self):
        """多話者TextEncoderでspksがDPに渡ること"""
        encoder = _build_encoder(n_spks=2, spk_emb_dim=64)
        encoder.eval()
        x = torch.randint(0, 178, (2, 10))
        x_lengths = torch.tensor([10, 10])
        spks = torch.randn(2, 64)
        with torch.no_grad():
            mu, logw, x_mask = encoder(x, x_lengths, spks=spks)
        assert logw.shape == (2, 1, 10)

    def test_encoder_multispeaker_output_shape(self):
        """多話者TextEncoderの出力shape"""
        encoder = _build_encoder(n_spks=2, spk_emb_dim=64)
        encoder.eval()
        x = torch.randint(0, 178, (2, 10))
        x_lengths = torch.tensor([10, 10])
        spks = torch.randn(2, 64)
        with torch.no_grad():
            mu, logw, x_mask = encoder(x, x_lengths, spks=spks)
        assert mu.shape == (2, 80, 10)
        assert logw.shape == (2, 1, 10)
        assert x_mask.shape == (2, 1, 10)

    def test_encoder_single_speaker_unchanged(self):
        """単一話者TextEncoderの出力が変更前と同一"""
        encoder = _build_encoder(n_spks=1)
        assert not hasattr(encoder.proj_w, 'film_1')
        encoder.eval()
        x = torch.randint(0, 178, (2, 10))
        x_lengths = torch.tensor([10, 10])
        with torch.no_grad():
            mu, logw, x_mask = encoder(x, x_lengths)
        assert mu.shape == (2, 80, 10)


class TestBlankEmbeddingZeroInit:
    """Blank embedding (index 0) is zero-initialized for blank/phoneme separation."""

    def test_blank_embedding_is_zero_after_init(self):
        """初期化直後にblank embedding(index 0)が全ゼロ"""
        encoder = _build_encoder()
        assert torch.all(encoder.emb.weight.data[0] == 0.0)

    def test_phoneme_embeddings_are_nonzero(self):
        """index 1以降のembeddingがゼロでない"""
        encoder = _build_encoder()
        assert torch.any(encoder.emb.weight.data[1:] != 0.0)

    def test_blank_embedding_is_trainable(self):
        """blankのembeddingが学習可能（padding_idx未使用）"""
        encoder = _build_encoder()
        assert encoder.emb.weight.requires_grad is True
        # padding_idxが設定されていないことを確認
        assert encoder.emb.padding_idx is None

    def test_blank_embedding_receives_gradient(self):
        """forward+backward後にblank embeddingに勾配が流れること"""
        encoder = _build_encoder()
        encoder.train()
        x = torch.zeros(1, 5, dtype=torch.long)  # 全てblank (index 0)
        x_lengths = torch.tensor([5])
        mu, logw, x_mask = encoder(x, x_lengths)
        loss = mu.sum() + logw.sum()
        loss.backward()
        # blank embeddingの勾配がゼロでないことを確認
        assert encoder.emb.weight.grad is not None
        assert not torch.all(encoder.emb.weight.grad[0] == 0.0)

    def test_blank_phoneme_l2_distance(self):
        """初期化直後にblankと実音素のL2距離が0より大きい"""
        encoder = _build_encoder()
        blank = encoder.emb.weight.data[0]
        phoneme = encoder.emb.weight.data[1]
        l2 = torch.norm(blank - phoneme)
        assert l2 > 0.0
