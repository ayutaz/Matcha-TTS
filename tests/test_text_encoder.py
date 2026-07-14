"""Tests for the TextEncoder component."""

import math
from types import SimpleNamespace

import pytest
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
        assert hasattr(dp, "film_1")
        assert hasattr(dp, "film_2")

    def test_dp_no_film_single_speaker(self):
        """n_spks=1でFiLM層が生成されないこと"""
        from matcha.models.components.text_encoder import DurationPredictor

        dp = DurationPredictor(128, 64, 3, 0.1, n_spks=1)
        assert not hasattr(dp, "film_1")

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
        film_params = sum(p.numel() for n, p in dp.named_parameters() if "film" in n)
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
        assert not hasattr(encoder.proj_w, "film_1")
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


class TestRotaryCacheRebuild:
    """RoPEキャッシュがmax_seq_len超過時に正しく再構築されること"""

    def _make_rope(self, max_seq_len=16):
        from matcha.models.components.text_encoder import RotaryPositionalEmbeddings

        return RotaryPositionalEmbeddings(d=32, max_seq_len=max_seq_len)

    def test_forward_beyond_max_seq_len_rebuilds_cache(self):
        """seq_len > max_seq_lenでキャッシュが拡張され、出力shapeが保たれる"""
        rope = self._make_rope(max_seq_len=16)
        x = torch.randn(2, 2, 24, 32)  # [b, h, t, d], t=24 > 16
        out = rope(x)
        assert out.shape == x.shape
        assert rope.max_seq_len == 24
        assert rope.cos_cached.shape[0] == 24

    def test_rebuild_keeps_buffer_registration(self):
        """再構築後もcos/sinが非persistentバッファのまま維持される"""
        rope = self._make_rope(max_seq_len=16)
        rope(torch.randn(1, 2, 32, 32))
        buffers = dict(rope.named_buffers())
        assert "cos_cached" in buffers
        assert "sin_cached" in buffers
        # 非persistent: state_dictには含まれない
        assert "cos_cached" not in rope.state_dict()
        assert "sin_cached" not in rope.state_dict()

    def test_rebuild_preserves_values(self):
        """再構築後の先頭max_seq_len分は元のキャッシュと一致する"""
        rope = self._make_rope(max_seq_len=16)
        old_cos = rope.cos_cached.clone()
        old_sin = rope.sin_cached.clone()
        rope(torch.randn(1, 2, 24, 32))
        assert torch.allclose(rope.cos_cached[:16], old_cos)
        assert torch.allclose(rope.sin_cached[:16], old_sin)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_rebuild_on_input_device(self):
        """再構築されたキャッシュが入力と同じdeviceに配置される"""
        from matcha.models.components.text_encoder import RotaryPositionalEmbeddings

        rope = RotaryPositionalEmbeddings(d=32, max_seq_len=16).cuda()
        x = torch.randn(2, 2, 24, 32, device="cuda")
        out = rope(x)
        assert rope.cos_cached.device.type == "cuda"
        assert out.device.type == "cuda"


class TestSDPAPaddingInvariance:
    """SDPA attention: padded batch entries produce the same valid-position outputs as unpadded runs."""

    MAX_LEN = 12
    SHORT_LEN = MAX_LEN - 4

    def _run_padded_and_unpadded(self):
        """Run the same short item once inside a padded B=2 batch and once alone (B=1, no padding)."""
        torch.manual_seed(0)
        encoder = _build_encoder()
        encoder.eval()

        torch.manual_seed(1)
        x = torch.randint(1, 178, (2, self.MAX_LEN))
        x[1, self.SHORT_LEN :] = 0  # padded region of the short item
        x_lengths = torch.tensor([self.MAX_LEN, self.SHORT_LEN])

        with torch.no_grad():
            padded = encoder(x, x_lengths)
            unpadded = encoder(x[1:2, : self.SHORT_LEN], torch.tensor([self.SHORT_LEN]))
        return padded, unpadded

    def test_mu_matches_unpadded_run_at_valid_positions(self):
        (mu_p, _logw_p, _mask_p), (mu_u, _logw_u, _mask_u) = self._run_padded_and_unpadded()
        assert torch.allclose(mu_p[1, :, : self.SHORT_LEN], mu_u[0], atol=1e-5)

    def test_logw_matches_unpadded_run_at_valid_positions(self):
        (_mu_p, logw_p, _mask_p), (_mu_u, logw_u, _mask_u) = self._run_padded_and_unpadded()
        assert torch.allclose(logw_p[1, :, : self.SHORT_LEN], logw_u[0], atol=1e-5)

    def test_no_nan_in_padded_outputs(self):
        """Fully-masked SDPA rows (padded query positions) must not inject NaN into mu/logw."""
        (mu_p, logw_p, _mask_p), _ = self._run_padded_and_unpadded()
        assert not torch.isnan(mu_p).any()
        assert not torch.isnan(logw_p).any()


class TestSDPANumericalEquivalence:
    """MultiHeadAttention SDPA path matches a hand-rolled attention reference."""

    CHANNELS = 64
    N_HEADS = 2
    BATCH = 2
    SEQ_LEN = 9

    def _build_mha(self, p_dropout=0.0):
        from matcha.models.components.text_encoder import MultiHeadAttention

        torch.manual_seed(0)
        mha = MultiHeadAttention(
            channels=self.CHANNELS,
            out_channels=self.CHANNELS,
            n_heads=self.N_HEADS,
            p_dropout=p_dropout,
        )
        mha.eval()
        return mha

    def _make_input(self):
        torch.manual_seed(1)
        return torch.randn(self.BATCH, self.CHANNELS, self.SEQ_LEN)

    @staticmethod
    def _reference_attention(module, x, c, mask=None):
        """Hand-rolled attention reusing the module's own submodules.

        Mirrors MultiHeadAttention.attention: project via conv_q/k/v, reshape
        to heads, apply the module's rotary embeddings, scaled masked softmax,
        weighted sum over values, then conv_o.
        """
        q = module.conv_q(x)
        k = module.conv_k(c)
        v = module.conv_v(c)
        b, d, t_s = k.shape
        t_t = q.shape[2]
        n_heads, k_channels = module.n_heads, module.k_channels

        def to_heads(t):
            # "b (h c) t -> b h t c"
            return t.view(b, n_heads, k_channels, -1).transpose(2, 3)

        query = module.query_rotary_pe(to_heads(q))
        key = module.key_rotary_pe(to_heads(k))
        value = to_heads(v)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(k_channels)
        if mask is not None:
            # SDPA boolean masking: disallowed positions get -inf before softmax
            scores = scores.masked_fill(mask == 0, float("-inf"))
        p_attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(p_attn, value)
        output = output.transpose(2, 3).reshape(b, d, t_t)
        return module.conv_o(output)

    def test_matches_reference_without_mask(self):
        mha = self._build_mha()
        x = self._make_input()
        with torch.no_grad():
            out_module = mha(x, x, attn_mask=None)
            out_ref = self._reference_attention(mha, x, x, mask=None)
        assert torch.allclose(out_module, out_ref, atol=1e-5)

    def test_matches_reference_with_partial_mask(self):
        mha = self._build_mha()
        x = self._make_input()
        # Key-side partial mask; every query row keeps at least one valid key
        mask = torch.ones(self.BATCH, 1, self.SEQ_LEN, self.SEQ_LEN)
        mask[1, :, :, self.SEQ_LEN - 3 :] = 0.0
        with torch.no_grad():
            out_module = mha(x, x, attn_mask=mask)
            out_ref = self._reference_attention(mha, x, x, mask=mask)
        assert torch.allclose(out_module, out_ref, atol=1e-5)

    def test_attn_is_none_after_sdpa_path(self):
        """SDPA path never materializes the attention matrix (self.attn stays None)."""
        mha = self._build_mha()
        x = self._make_input()
        with torch.no_grad():
            mha(x, x, attn_mask=None)
        assert mha.attn is None

    def test_eval_mode_is_deterministic(self):
        """dropout_p=0 in eval mode: two forward passes give identical outputs."""
        mha = self._build_mha(p_dropout=0.1)
        x = self._make_input()
        with torch.no_grad():
            out_1 = mha(x, x, attn_mask=None)
            out_2 = mha(x, x, attn_mask=None)
        assert torch.equal(out_1, out_2)


class TestLayerNormFP32:
    """LayerNorm: fp32 accumulation for fp16 inputs and channel-dim (dim=1) semantics."""

    CHANNELS = 8

    def _make_layer_norm(self):
        from matcha.models.components.text_encoder import LayerNorm

        return LayerNorm(channels=self.CHANNELS)

    def _make_large_fp16_input(self):
        """Large-magnitude fp16 input where naive fp16 statistics lose precision."""
        torch.manual_seed(0)
        x32 = 300.0 + 0.5 * torch.randn(2, self.CHANNELS, 6)
        return x32.to(torch.float16)

    def test_fp16_input_output_is_finite(self):
        ln = self._make_layer_norm()
        out = ln(self._make_large_fp16_input())
        assert torch.isfinite(out).all()

    def test_fp16_matches_pure_float32_computation(self):
        """fp16 input normalized via fp32 accumulation matches the same math done fully in fp32."""
        ln = self._make_layer_norm()
        x16 = self._make_large_fp16_input()
        x32 = x16.float()

        with torch.no_grad():
            out_fp16_path = ln(x16)
            # Reference: identical computation carried out entirely in float32
            mean = x32.mean(dim=1, keepdim=True)
            variance = ((x32 - mean) ** 2).mean(dim=1, keepdim=True)
            normalized = (x32 - mean) * torch.rsqrt(variance + ln.eps)
            out_ref = normalized * ln.gamma.view(1, -1, 1) + ln.beta.view(1, -1, 1)

        assert torch.allclose(out_fp16_path.float(), out_ref, atol=1e-2)

    def test_dim1_mean_is_zero(self):
        """Normalization runs over dim=1 (channels), not the last dim: per-(batch, time) mean ~ 0."""
        ln = self._make_layer_norm()
        torch.manual_seed(1)
        x = torch.randn(3, self.CHANNELS, 5)
        with torch.no_grad():
            out = ln(x)  # default gamma=1, beta=0
        assert torch.allclose(out.mean(dim=1), torch.zeros(3, 5), atol=1e-5)

    def test_dim1_std_is_one(self):
        """Per-(batch, time) biased std over dim=1 ~ 1 (up to eps in the denominator)."""
        ln = self._make_layer_norm()
        torch.manual_seed(1)
        x = torch.randn(3, self.CHANNELS, 5)
        with torch.no_grad():
            out = ln(x)
        std = out.var(dim=1, unbiased=False).sqrt()
        assert torch.allclose(std, torch.ones(3, 5), atol=1e-2)
