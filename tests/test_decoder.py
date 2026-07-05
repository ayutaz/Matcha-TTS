import math
from copy import deepcopy

import pytest
import torch

from matcha.models.components.decoder import Decoder, SinusoidalPosEmb


@pytest.fixture()
def decoder_config():
    """Minimal decoder configuration for fast tests."""
    return dict(
        in_channels=80,
        out_channels=80,
        channels=[64, 64],
        attention_head_dim=32,
        n_blocks=1,
        num_mid_blocks=1,
        num_heads=2,
    )


@pytest.fixture()
def decoder(decoder_config):
    model = Decoder(**decoder_config)
    model.eval()
    return model


@pytest.fixture()
def sample_inputs():
    """Create sample inputs for the decoder.

    The decoder packs x and mu along the channel dimension (einops pack),
    so each must have in_channels // 2 = 40 channels to produce 80 packed
    channels that match the in_channels of the first ResnetBlock.
    """
    batch, mel_channels, length = 2, 40, 20
    x = torch.randn(batch, mel_channels, length)
    mu = torch.randn(batch, mel_channels, length)
    mask = torch.ones(batch, 1, length)
    t = torch.rand(batch)
    return x, mu, mask, t


class TestDecoderInstantiation:
    def test_instantiation(self, decoder, decoder_config):
        """Decoder can be instantiated with small config."""
        assert isinstance(decoder, Decoder)
        assert decoder.in_channels == decoder_config["in_channels"]
        assert decoder.out_channels == decoder_config["out_channels"]

    def test_has_expected_submodules(self, decoder):
        """Decoder contains down, mid, and up block lists."""
        assert len(decoder.down_blocks) == 2
        assert len(decoder.mid_blocks) == 1
        assert len(decoder.up_blocks) == 2


class TestDecoderForward:
    def test_forward_output_shape(self, decoder, sample_inputs):
        """Forward pass produces the correct output shape (batch, out_channels, length)."""
        x, mu, mask, t = sample_inputs
        with torch.no_grad():
            output = decoder(x, mask, mu, t)
        assert output.shape == (2, 80, 20)

    def test_output_matches_input_spatial_dims(self, decoder, sample_inputs):
        """Output has the same spatial (time) dimension as the input."""
        x, mu, mask, t = sample_inputs
        with torch.no_grad():
            output = decoder(x, mask, mu, t)
        assert output.shape[0] == x.shape[0], "Batch dimension mismatch"
        assert output.shape[2] == x.shape[2], "Time dimension mismatch"

    def test_output_channels_equal_out_channels(self, decoder, decoder_config, sample_inputs):
        """Output channel dimension equals the configured out_channels."""
        x, mu, mask, t = sample_inputs
        with torch.no_grad():
            output = decoder(x, mask, mu, t)
        assert output.shape[1] == decoder_config["out_channels"]

    def test_mask_zeros_output(self, decoder, sample_inputs):
        """An all-zeros mask produces an all-zeros output."""
        x, mu, mask, t = sample_inputs
        zero_mask = torch.zeros_like(mask)
        with torch.no_grad():
            output = decoder(x, zero_mask, mu, t)
        assert torch.allclose(output, torch.zeros_like(output))

    def test_optional_spks_input(self, decoder, sample_inputs):
        """Decoder accepts an optional spks tensor without error."""
        x, mu, mask, t = sample_inputs
        spks = torch.randn(2, 16)
        # spks are packed onto x and mu, so in_channels must account for
        # the extra speaker channels. Build a decoder that expects them.
        dec = Decoder(
            in_channels=80 + 16,
            out_channels=80,
            channels=[64, 64],
            attention_head_dim=32,
            n_blocks=1,
            num_mid_blocks=1,
            num_heads=2,
        )
        dec.eval()
        with torch.no_grad():
            output = dec(x, mask, mu, t, spks=spks)
        assert output.shape == (2, 80, 20)

    def test_cond_none_accepted(self, decoder, sample_inputs):
        """Passing cond=None (the default) works without error."""
        x, mu, mask, t = sample_inputs
        with torch.no_grad():
            output = decoder(x, mask, mu, t, spks=None, cond=None)
        assert output.shape == (2, 80, 20)


@pytest.fixture()
def parity_pair(decoder_config):
    """Two identically-weighted small Decoders: plain and gradient-checkpointed.

    dropout=0.0 removes the only source of randomness in train() mode so
    outputs and gradients must match exactly between the two code paths.
    """
    config = dict(decoder_config, dropout=0.0)
    torch.manual_seed(0)
    base = Decoder(**config)
    ckpt = Decoder(**config, use_gradient_checkpointing=True)
    ckpt.load_state_dict(deepcopy(base.state_dict()))
    base.train()
    ckpt.train()
    return base, ckpt


@pytest.fixture()
def parity_inputs():
    """Deterministic inputs with a non-trivial padding mask."""
    torch.manual_seed(42)
    batch, mel_channels, length = 2, 40, 20
    x = torch.randn(batch, mel_channels, length)
    mu = torch.randn(batch, mel_channels, length)
    mask = torch.ones(batch, 1, length)
    mask[1, :, 16:] = 0.0  # exercise attention_mask handling in both paths
    t = torch.rand(batch)
    return x, mu, mask, t


class TestGradientCheckpointingParity:
    """Checkpointed forward must be numerically identical to the plain path.

    This guards the positional-arg mapping (x, mask, None, None, t) onto
    BasicTransformerBlock.forward(hidden_states, attention_mask,
    encoder_hidden_states, encoder_attention_mask, timestep) used by
    torch.utils.checkpoint in Decoder.forward.
    """

    def test_constructor_flag(self, decoder_config):
        """use_gradient_checkpointing is stored from the constructor argument."""
        assert Decoder(**decoder_config).use_gradient_checkpointing is False
        assert Decoder(**decoder_config, use_gradient_checkpointing=True).use_gradient_checkpointing is True

    def test_enable_gradient_checkpointing_flips_flag(self, decoder_config):
        """enable_gradient_checkpointing() sets the flag on a plain Decoder."""
        model = Decoder(**decoder_config)
        assert model.use_gradient_checkpointing is False
        model.enable_gradient_checkpointing()
        assert model.use_gradient_checkpointing is True

    def test_forward_outputs_match(self, parity_pair, parity_inputs):
        """Train-mode outputs are identical with and without checkpointing."""
        base, ckpt = parity_pair
        x, mu, mask, t = parity_inputs
        out_base = base(x, mask, mu, t)
        out_ckpt = ckpt(x, mask, mu, t)
        assert torch.allclose(out_base, out_ckpt, atol=1e-6)

    def test_gradients_match(self, parity_pair, parity_inputs):
        """Backward through the checkpointed path reproduces every gradient."""
        base, ckpt = parity_pair
        x, mu, mask, t = parity_inputs

        x_base = x.clone().requires_grad_(True)
        x_ckpt = x.clone().requires_grad_(True)

        base(x_base, mask, mu, t).sum().backward()
        ckpt(x_ckpt, mask, mu, t).sum().backward()

        assert x_base.grad is not None and x_ckpt.grad is not None
        assert torch.allclose(x_base.grad, x_ckpt.grad, atol=1e-6), "Input gradient mismatch"

        ckpt_grads = dict(ckpt.named_parameters())
        for name, param in base.named_parameters():
            grad_base = param.grad
            grad_ckpt = ckpt_grads[name].grad
            assert (grad_base is None) == (grad_ckpt is None), f"Gradient presence mismatch for {name}"
            if grad_base is not None:
                assert torch.allclose(grad_base, grad_ckpt, atol=1e-6), f"Gradient mismatch for {name}"

    def test_eval_inference_mode_bypasses_checkpointing(self, parity_pair, parity_inputs):
        """In eval() mode checkpointing is skipped and inference_mode works."""
        base, ckpt = parity_pair
        base.eval()
        ckpt.eval()
        x, mu, mask, t = parity_inputs
        with torch.inference_mode():
            out_ckpt = ckpt(x, mask, mu, t)
            out_base = base(x, mask, mu, t)
        assert out_ckpt.shape == (2, 80, 20)
        assert torch.allclose(out_base, out_ckpt, atol=1e-6)


def _reference_sinusoidal_emb(t, dim, scale):
    """Reference implementation of the original einops-based SinusoidalPosEmb."""
    half_dim = dim // 2
    emb = torch.exp(torch.arange(half_dim).float() * -(math.log(10000) / (half_dim - 1)))
    arg = scale * t.unsqueeze(1) * emb.unsqueeze(0)
    return torch.cat((arg.sin(), arg.cos()), dim=-1)


class TestSinusoidalPosEmb:
    """Value/edge-case tests for the register_buffer rewrite of SinusoidalPosEmb."""

    def test_matches_reference_formula_default_scale(self):
        """Default scale (1000) output matches the closed-form reference."""
        dim = 64
        module = SinusoidalPosEmb(dim)
        t = torch.tensor([0.0, 1.0, 5.5, 999.0])
        out = module(t)
        expected = _reference_sinusoidal_emb(t, dim, scale=1000)
        assert out.shape == (4, dim)
        assert torch.allclose(out, expected, atol=1e-6)

    @pytest.mark.parametrize("scale", [1, 250])
    def test_custom_scale_matches_reference(self, scale):
        """A non-default scale changes the output consistently with the formula."""
        dim = 32
        module = SinusoidalPosEmb(dim)
        t = torch.tensor([0.5, 2.0, 100.0])
        out = module(t, scale=scale)
        expected = _reference_sinusoidal_emb(t, dim, scale=scale)
        assert torch.allclose(out, expected, atol=1e-6)
        assert not torch.allclose(out, module(t), atol=1e-6), "Custom scale should differ from default"

    def test_scalar_input_returns_batch_of_one(self):
        """A 0-dim scalar tensor is promoted to a batch of one."""
        dim = 16
        module = SinusoidalPosEmb(dim)
        out = module(torch.tensor(3.0))
        assert out.shape == (1, dim)
        expected = _reference_sinusoidal_emb(torch.tensor([3.0]), dim, scale=1000)
        assert torch.allclose(out, expected, atol=1e-6)

    def test_odd_dim_raises_assertion_error(self):
        """Odd dimensions are rejected at construction time."""
        with pytest.raises(AssertionError, match="even"):
            SinusoidalPosEmb(7)

    def test_dim_two_raises_assertion_error(self):
        """dim=2 (half_dim=1) is rejected with a clear message instead of ZeroDivisionError."""
        with pytest.raises(AssertionError, match="dim >= 4"):
            SinusoidalPosEmb(2)

    def test_emb_weights_buffer_values(self):
        """The cached buffer holds exp(arange(half_dim) * -log(10000)/(half_dim-1))."""
        dim = 20
        module = SinusoidalPosEmb(dim)
        half_dim = dim // 2
        expected = torch.exp(torch.arange(half_dim).float() * -(math.log(10000) / (half_dim - 1)))
        assert module.emb_weights.shape == (half_dim,)
        assert torch.allclose(module.emb_weights, expected, atol=1e-6)


class TestSinusoidalPosEmbCheckpointCompat:
    """emb_weights is a *non-persistent* buffer (matching the RoPE caches in the
    text encoder).

    Pre-branch/upstream checkpoints were saved before the register_buffer
    rewrite and lack the ``time_embeddings.emb_weights`` key; because the
    buffer is rebuilt at construction and excluded from the state_dict, strict
    loading of those checkpoints succeeds. Stale keys from checkpoints saved
    while the buffer was persistent are stripped at the Lightning level by
    BaseLightningClass.on_load_checkpoint (covered in test_matcha_tts.py).
    """

    def test_emb_weights_absent_from_state_dict(self, decoder):
        """The buffer is non-persistent and never enters the state_dict."""
        assert [k for k in decoder.state_dict() if "emb_weights" in k] == []

    def test_strict_load_without_emb_weights_succeeds(self, decoder, decoder_config, sample_inputs):
        """A pre-branch checkpoint lacking emb_weights loads cleanly with strict=True;
        the construction-time buffer stays correct."""
        state_dict = {k: v for k, v in decoder.state_dict().items() if "emb_weights" not in k}
        fresh = Decoder(**decoder_config)
        result = fresh.load_state_dict(state_dict, strict=True)
        assert not result.missing_keys
        assert not result.unexpected_keys

        # The buffer was initialized at construction, so embeddings are still correct.
        t = torch.tensor([0.0, 1.0, 5.5, 999.0])
        expected = _reference_sinusoidal_emb(t, decoder_config["in_channels"], scale=1000)
        assert torch.allclose(fresh.time_embeddings(t), expected, atol=1e-6)

        # And the fully-loaded model runs end to end.
        fresh.eval()
        x, mu, mask, t_dec = sample_inputs
        with torch.no_grad():
            output = fresh(x, mask, mu, t_dec)
        assert output.shape == (2, 80, 20)

    def test_stale_persistent_key_requires_hook_strip(self, decoder, decoder_config):
        """A checkpoint saved while the buffer was persistent carries a stale key:
        plain nn.Module strict loading rejects it, which is why the Lightning
        on_load_checkpoint hook strips it before loading."""
        state_dict = dict(decoder.state_dict())
        state_dict["time_embeddings.emb_weights"] = decoder.time_embeddings.emb_weights.clone()
        fresh = Decoder(**decoder_config)
        with pytest.raises(RuntimeError, match="emb_weights"):
            fresh.load_state_dict(state_dict, strict=True)
        result = fresh.load_state_dict(state_dict, strict=False)
        assert result.unexpected_keys == ["time_embeddings.emb_weights"]
