import pytest
import torch

from matcha.hifigan.config import v1
from matcha.hifigan.denoiser import Denoiser
from matcha.hifigan.env import AttrDict
from matcha.hifigan.models import Generator


@pytest.fixture()
def hifigan_config():
    """Return the v1 HiFi-GAN config wrapped in AttrDict."""
    return AttrDict(v1)


@pytest.fixture()
def generator(hifigan_config):
    """Return a Generator instance in eval mode on CPU."""
    gen = Generator(hifigan_config)
    gen.eval()
    return gen


class TestGeneratorInstantiation:
    def test_creates_with_v1_config(self, generator):
        assert isinstance(generator, Generator)

    def test_num_upsamples_matches_config(self, generator, hifigan_config):
        assert generator.num_upsamples == len(hifigan_config.upsample_rates)

    def test_num_kernels_matches_config(self, generator, hifigan_config):
        assert generator.num_kernels == len(hifigan_config.resblock_kernel_sizes)


class TestGeneratorForward:
    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_output_shape_has_single_channel(self, generator, batch_size):
        mel = torch.randn(batch_size, 80, 10)
        with torch.no_grad():
            audio = generator(mel)
        assert audio.ndim == 3
        assert audio.shape[0] == batch_size
        assert audio.shape[1] == 1

    def test_output_length_matches_hop_length(self, generator, hifigan_config):
        mel_len = 10
        hop_length = 1
        for r in hifigan_config.upsample_rates:
            hop_length *= r
        # The total upsample factor should equal 256 for v1
        assert hop_length == 256

        mel = torch.randn(1, 80, mel_len)
        with torch.no_grad():
            audio = generator(mel)
        expected_audio_len = mel_len * hop_length
        actual_audio_len = audio.shape[2]
        assert actual_audio_len == expected_audio_len

    def test_output_values_bounded_by_tanh(self, generator):
        mel = torch.randn(1, 80, 10)
        with torch.no_grad():
            audio = generator(mel)
        assert audio.min() >= -1.0
        assert audio.max() <= 1.0

    def test_different_mel_lengths_produce_proportional_audio(self, generator):
        mel_short = torch.randn(1, 80, 5)
        mel_long = torch.randn(1, 80, 20)
        with torch.no_grad():
            audio_short = generator(mel_short)
            audio_long = generator(mel_long)
        assert audio_long.shape[2] == 4 * audio_short.shape[2]


class TestDenoiser:
    def test_instantiation_from_generator(self, generator):
        denoiser = Denoiser(generator)
        assert isinstance(denoiser, Denoiser)
        assert hasattr(denoiser, "bias_spec")

    def test_forward_preserves_length(self, generator):
        mel = torch.randn(1, 80, 10)
        with torch.no_grad():
            audio = generator(mel)
        denoiser = Denoiser(generator)
        # Denoiser expects squeezed audio (remove channel dim)
        denoised = denoiser(audio.squeeze(1))
        assert denoised.shape[-1] == audio.shape[-1]
