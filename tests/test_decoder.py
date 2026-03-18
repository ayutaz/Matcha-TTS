import pytest
import torch

from matcha.models.components.decoder import Decoder


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
