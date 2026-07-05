"""Tests for Matcha-TTS utility functions."""

import matplotlib
import numpy as np
import pytest
import torch

matplotlib.use("Agg")

from matcha.utils.audio import mel_spectrogram
from matcha.utils.model import (
    denormalize,
    duration_loss,
    fix_len_compatibility,
    generate_path,
    normalize,
    sequence_mask,
)
from matcha.utils.monotonic_align import maximum_path
from matcha.utils.utils import (
    get_user_data_dir,
    intersperse,
    plot_tensor,
    save_figure_to_numpy,
)

# ---------------------------------------------------------------------------
# intersperse
# ---------------------------------------------------------------------------


class TestIntersperse:
    def test_basic(self):
        assert intersperse([1, 2, 3], 0) == [0, 1, 0, 2, 0, 3, 0]

    def test_empty_list(self):
        assert intersperse([], 0) == [0]

    def test_single_element(self):
        assert intersperse([5], 0) == [0, 5, 0]

    def test_strings(self):
        assert intersperse(["a", "b"], "_") == ["_", "a", "_", "b", "_"]

    def test_length_relation(self):
        lst = list(range(10))
        result = intersperse(lst, -1)
        assert len(result) == 2 * len(lst) + 1

    def test_blank_positions(self):
        """Blank items should sit at every even index."""
        result = intersperse([1, 2, 3], 0)
        for i in range(0, len(result), 2):
            assert result[i] == 0

    def test_original_items_preserved(self):
        """Original items should sit at every odd index, in order."""
        lst = [10, 20, 30]
        result = intersperse(lst, 0)
        assert result[1::2] == lst


# ---------------------------------------------------------------------------
# sequence_mask
# ---------------------------------------------------------------------------


class TestSequenceMask:
    def test_shape(self):
        lengths = torch.tensor([3, 5, 2])
        mask = sequence_mask(lengths)
        assert mask.shape == (3, 5)

    def test_shape_with_max_length(self):
        lengths = torch.tensor([2, 4])
        mask = sequence_mask(lengths, max_length=6)
        assert mask.shape == (2, 6)

    def test_values(self):
        lengths = torch.tensor([3, 1])
        mask = sequence_mask(lengths)
        expected = torch.tensor([[True, True, True], [True, False, False]])
        assert torch.equal(mask, expected)

    def test_all_zeros(self):
        lengths = torch.tensor([0, 0])
        mask = sequence_mask(lengths, max_length=3)
        assert mask.sum().item() == 0

    def test_full_mask(self):
        lengths = torch.tensor([4, 4])
        mask = sequence_mask(lengths, max_length=4)
        assert mask.all()

    def test_dtype_matches_length(self):
        lengths = torch.tensor([2, 3], dtype=torch.long)
        mask = sequence_mask(lengths)
        assert mask.dtype == torch.bool


# ---------------------------------------------------------------------------
# generate_path
# ---------------------------------------------------------------------------


class TestGeneratePath:
    def test_output_shape(self):
        batch, t_x, t_y = 2, 3, 10
        duration = torch.tensor([[3, 4, 3], [2, 5, 3]], dtype=torch.long)
        mask = torch.ones(batch, t_x, t_y)
        path = generate_path(duration, mask)
        assert path.shape == (batch, t_x, t_y)

    def test_path_sums_to_t_y(self):
        """Each sample's path should assign every output frame exactly once."""
        batch, t_x, t_y = 1, 3, 10
        duration = torch.tensor([[3, 4, 3]])
        mask = torch.ones(batch, t_x, t_y)
        path = generate_path(duration, mask)
        # Sum over the input-phone axis; each output frame should be 1.
        assert torch.allclose(path.sum(dim=1), torch.ones(batch, t_y))

    def test_each_phone_occupies_correct_frames(self):
        duration = torch.tensor([[2, 3]])
        mask = torch.ones(1, 2, 5)
        path = generate_path(duration, mask)
        # phone-0 should cover frames 0-1, phone-1 frames 2-4
        assert path[0, 0, :2].sum().item() == 2
        assert path[0, 0, 2:].sum().item() == 0
        assert path[0, 1, 2:5].sum().item() == 3

    def test_masked_positions_are_zero(self):
        duration = torch.tensor([[2, 3]])
        mask = torch.ones(1, 2, 5)
        mask[0, :, 3:] = 0  # mask out last two frames
        path = generate_path(duration, mask)
        assert path[0, :, 3:].sum().item() == 0

    def test_zero_duration_tokens_get_empty_rows(self):
        """Tokens with duration 0 must not claim any frame; coverage stays exact."""
        duration = torch.tensor([[0, 3, 0, 2, 0]])
        mask = torch.ones(1, 5, 5)
        path = generate_path(duration, mask)
        # Zero-duration tokens (indices 0, 2, 4) get all-zero rows.
        assert path[0, 0].sum().item() == 0
        assert path[0, 2].sum().item() == 0
        assert path[0, 4].sum().item() == 0
        # Non-zero tokens keep their frame counts.
        assert path[0, 1].sum().item() == 3
        assert path[0, 3].sum().item() == 2
        # Every output frame is assigned to exactly one token.
        assert torch.allclose(path.sum(dim=1), torch.ones(1, 5))
        assert path.sum().item() == 5

    def test_trailing_shortfall_leaves_frames_unassigned(self):
        """Durations summing to less than t_y leave the trailing frames uncovered."""
        duration = torch.tensor([[2, 1]])
        mask = torch.ones(1, 2, 5)
        path = generate_path(duration, mask)
        assert path.sum().item() == 3
        # Frames 3-4 belong to no token (column sums are 0).
        assert path[0, :, 3:].sum().item() == 0
        # Frames 0-2 are each covered exactly once.
        assert torch.allclose(path[0, :, :3].sum(dim=0), torch.ones(3))


# ---------------------------------------------------------------------------
# fix_len_compatibility
# ---------------------------------------------------------------------------


class TestFixLenCompatibility:
    def test_already_compatible(self):
        # 8 is divisible by 2^2 = 4
        assert fix_len_compatibility(8, num_downsamplings_in_unet=2) == 8

    def test_rounds_up(self):
        # 5 -> ceil(5/4)*4 = 8
        assert fix_len_compatibility(5, num_downsamplings_in_unet=2) == 8

    def test_one_downsampling(self):
        # factor = 2^1 = 2; 3 -> ceil(3/2)*2 = 4
        assert fix_len_compatibility(3, num_downsamplings_in_unet=1) == 4

    def test_zero_downsamplings(self):
        # factor = 2^0 = 1; any length is compatible
        assert fix_len_compatibility(7, num_downsamplings_in_unet=0) == 7

    def test_result_is_int(self):
        result = fix_len_compatibility(10, num_downsamplings_in_unet=2)
        assert isinstance(result, int)

    def test_large_value(self):
        result = fix_len_compatibility(1000, num_downsamplings_in_unet=3)
        assert result % 8 == 0
        assert result >= 1000

    @pytest.mark.parametrize("num_downsamplings", [1, 2, 3])
    @pytest.mark.parametrize("n", [1, 5, 63, 64, 100, 257])
    def test_tensor_branch_matches_int_branch(self, n, num_downsamplings):
        """A 0-dim tensor input must round up to the same value as a plain int."""
        int_result = fix_len_compatibility(n, num_downsamplings_in_unet=num_downsamplings)
        tensor_result = fix_len_compatibility(torch.tensor(n), num_downsamplings_in_unet=num_downsamplings)
        assert tensor_result == int_result

    def test_tensor_branch_accepts_long_dtype(self):
        """synthesise() passes y_lengths.max(), a 0-dim torch.long tensor."""
        y_lengths = torch.tensor([10, 25, 17], dtype=torch.long)
        result = fix_len_compatibility(y_lengths.max())
        assert result == fix_len_compatibility(25)

    def test_tensor_branch_returns_python_int(self):
        """Outside ONNX export the tensor branch returns a plain Python int.

        synthesise() uses the result as a tensor size, so pin the .item() contract.
        """
        result = fix_len_compatibility(torch.tensor(10), num_downsamplings_in_unet=2)
        assert isinstance(result, int)
        assert not isinstance(result, torch.Tensor)


# ---------------------------------------------------------------------------
# normalize / denormalize
# ---------------------------------------------------------------------------


def _make_mel_batch():
    torch.manual_seed(42)
    return torch.randn(2, 80, 17)


def _make_mu_std(form):
    """Return (mu, std) in one of the accepted argument forms."""
    mu_values = np.linspace(-6.0, -1.0, 80).astype(np.float32)
    std_values = np.linspace(0.5, 2.5, 80).astype(np.float32)
    if form == "float":
        return -6.550095, 2.383771
    if form == "list":
        return mu_values.tolist(), std_values.tolist()
    if form == "ndarray":
        return mu_values, std_values
    if form == "tensor":
        return torch.from_numpy(mu_values), torch.from_numpy(std_values)
    raise ValueError(form)


class TestNormalizeDenormalize:
    @pytest.mark.parametrize("form", ["float", "list", "ndarray", "tensor"])
    def test_roundtrip(self, form):
        """denormalize(normalize(x)) must recover x for every accepted mu/std form."""
        data = _make_mel_batch()
        mu, std = _make_mu_std(form)
        out = denormalize(normalize(data, mu, std), mu, std)
        assert out.shape == data.shape
        assert torch.allclose(out, data, atol=1e-5)

    def test_scalar_normalize_value(self):
        data = torch.full((1, 2, 3), 5.0)
        out = normalize(data, 3.0, 2.0)
        assert torch.allclose(out, torch.full((1, 2, 3), 1.0))

    def test_per_channel_mu_shifts_channels_independently(self):
        """Non-scalar mu is broadcast along time via unsqueeze(-1), per channel."""
        data = torch.zeros(1, 3, 4)
        mu = [1.0, 2.0, 3.0]
        out = normalize(data, mu, 1.0)
        expected = torch.tensor([-1.0, -2.0, -3.0]).view(1, 3, 1).expand(1, 3, 4)
        assert torch.allclose(out, expected)

    def test_per_channel_denormalize_broadcasts(self):
        data = torch.zeros(1, 3, 4)
        mu = torch.tensor([1.0, 2.0, 3.0])
        std = torch.tensor([2.0, 4.0, 8.0])
        out = denormalize(data, mu, std)  # 0 * std + mu
        assert torch.allclose(out, mu.view(1, 3, 1).expand(1, 3, 4))

    def test_normalize_accepts_int_scalar(self):
        data = torch.full((1, 2, 3), 5.0)
        out = normalize(data, 3, 2)
        assert torch.allclose(out, torch.full((1, 2, 3), 1.0))

    def test_denormalize_accepts_int_scalar(self):
        """denormalize() must special-case (float, int) scalars symmetrically with
        normalize(); an int mu/std used to fall through to mu.unsqueeze(-1) and
        raise AttributeError.
        """
        data = torch.randn(1, 2, 3)
        out = denormalize(data, 3, 2)
        assert torch.allclose(out, data * 2 + 3)

    def test_float64_ndarray_stats_keep_data_dtype(self):
        """float64 ndarray mu/std must be cast to data.dtype (like the list branch)
        instead of silently promoting the float32 mel batch to float64."""
        data = _make_mel_batch()
        mu = np.linspace(-6.0, -1.0, 80)  # float64 by default
        std = np.linspace(0.5, 2.5, 80)
        normed = normalize(data, mu, std)
        assert normed.dtype == torch.float32
        out = denormalize(normed, mu, std)
        assert out.dtype == torch.float32
        assert torch.allclose(out, data, atol=1e-4)


# ---------------------------------------------------------------------------
# duration_loss
# ---------------------------------------------------------------------------


class TestDurationLoss:
    def test_hand_computed_value(self):
        """loss = sum((logw - logw_)**2) / sum(lengths)."""
        logw = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        logw_ = torch.zeros(2, 2)
        lengths = torch.tensor([2, 2])
        loss = duration_loss(logw, logw_, lengths)
        # sum of squares = 1 + 4 + 9 + 16 = 30; sum(lengths) = 4
        assert loss.item() == pytest.approx(30.0 / 4.0)

    def test_zero_for_identical_inputs(self):
        torch.manual_seed(0)
        logw = torch.randn(2, 5)
        lengths = torch.tensor([5, 5])
        loss = duration_loss(logw, logw.clone(), lengths)
        assert loss.item() == pytest.approx(0.0)

    def test_halving_lengths_doubles_loss(self):
        torch.manual_seed(1)
        logw = torch.randn(2, 4)
        logw_ = torch.randn(2, 4)
        full = duration_loss(logw, logw_, torch.tensor([4, 4]))
        half = duration_loss(logw, logw_, torch.tensor([2, 2]))
        assert half.item() == pytest.approx(2 * full.item())


# ---------------------------------------------------------------------------
# mel_spectrogram cache keys
# ---------------------------------------------------------------------------


def _reference_mel(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax):
    """Cache-free reimplementation of mel_spectrogram with fresh librosa filters."""
    from librosa.filters import mel as librosa_mel_fn

    mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
    mel = torch.from_numpy(mel).float()
    pad = int((n_fft - hop_size) / 2)
    y = torch.nn.functional.pad(y.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=torch.hann_window(win_size),
        center=False,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = torch.view_as_real(spec)
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)
    return torch.log(torch.clamp(torch.matmul(mel, spec), min=1e-5))


class TestMelSpectrogramCache:
    """The module-level mel_basis/hann_window caches used to be keyed only by
    (fmax, device), so a later call with equal fmax but a different
    n_fft/win_size/num_mels/sampling_rate silently reused a stale filterbank
    and window. The keys must cover every shape-relevant parameter — this
    branch runs both the 22050Hz training config and a 16kHz Julius config."""

    CFG_22K = {"n_fft": 1024, "num_mels": 80, "sampling_rate": 22050, "hop_size": 256, "win_size": 1024}
    CFG_16K = {"n_fft": 512, "num_mels": 80, "sampling_rate": 16000, "hop_size": 160, "win_size": 512}

    def test_second_config_with_equal_fmax_not_served_stale_cache(self):
        torch.manual_seed(7)
        y22 = torch.randn(1, 22050).clamp(-1.0, 1.0)
        y16 = torch.randn(1, 16000).clamp(-1.0, 1.0)
        out22 = mel_spectrogram(y22, fmin=0, fmax=None, **self.CFG_22K)
        out16 = mel_spectrogram(y16, fmin=0, fmax=None, **self.CFG_16K)
        assert torch.allclose(out22, _reference_mel(y22, fmin=0, fmax=None, **self.CFG_22K), atol=1e-5)
        assert torch.allclose(out16, _reference_mel(y16, fmin=0, fmax=None, **self.CFG_16K), atol=1e-5)

    def test_same_shape_different_sampling_rate_uses_fresh_filterbank(self):
        """Same n_fft/win_size but a different sampling_rate must not share a
        mel filterbank — the shapes match, so stale reuse is silent."""
        torch.manual_seed(8)
        y = torch.randn(1, 22050).clamp(-1.0, 1.0)
        out_22k = mel_spectrogram(
            y, sampling_rate=22050, n_fft=1024, num_mels=80, hop_size=256, win_size=1024, fmin=0, fmax=None
        )
        out_16k = mel_spectrogram(
            y, sampling_rate=16000, n_fft=1024, num_mels=80, hop_size=256, win_size=1024, fmin=0, fmax=None
        )
        ref_16k = _reference_mel(
            y, sampling_rate=16000, n_fft=1024, num_mels=80, hop_size=256, win_size=1024, fmin=0, fmax=None
        )
        assert torch.allclose(out_16k, ref_16k, atol=1e-5)
        assert not torch.allclose(out_22k, out_16k)


# ---------------------------------------------------------------------------
# maximum_path CUDA dispatch
# ---------------------------------------------------------------------------


def _make_variable_length_mas_batch():
    """Batch of 3 with per-sample variable-length 2D masks (sample 2 heavily padded)."""
    torch.manual_seed(1234)
    batch, t_x, t_y = 3, 5, 10
    value = torch.randn(batch, t_x, t_y)
    mask = torch.ones(batch, t_x, t_y)
    # Sample 0: full 5x10; Sample 1: 4x8; Sample 2: 2x5 (deliberate heavy padding).
    mask[1, 4:, :] = 0
    mask[1, :, 8:] = 0
    mask[2, 2:, :] = 0
    mask[2, :, 5:] = 0
    return value, mask


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestMaximumPathCudaDispatch:
    """maximum_path dispatches to the PyTorch kernel on CUDA and Cython on CPU;
    both must yield identical binary paths."""

    def test_gpu_result_stays_on_cuda(self):
        value, mask = _make_variable_length_mas_batch()
        gpu_result = maximum_path(value.cuda(), mask.cuda())
        assert gpu_result.device.type == "cuda"

    def test_gpu_matches_cpu_exactly(self):
        value, mask = _make_variable_length_mas_batch()
        cpu_result = maximum_path(value, mask)
        gpu_result = maximum_path(value.cuda(), mask.cuda())
        assert gpu_result.dtype == cpu_result.dtype
        assert torch.equal(gpu_result.cpu(), cpu_result)

    def test_padded_sample_respects_mask(self):
        value, mask = _make_variable_length_mas_batch()
        gpu_result = maximum_path(value.cuda(), mask.cuda()).cpu()
        # Sample 2 is effectively 2x5: nothing outside the valid region...
        assert gpu_result[2, 2:, :].sum().item() == 0
        assert gpu_result[2, :, 5:].sum().item() == 0
        # ...and every valid frame is assigned exactly once.
        assert torch.allclose(gpu_result[2, :2, :5].sum(dim=0), torch.ones(5))


# ---------------------------------------------------------------------------
# get_user_data_dir
# ---------------------------------------------------------------------------


class TestGetUserDataDir:
    def test_returns_path(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MATCHA_HOME", str(tmp_path))
        result = get_user_data_dir()
        from pathlib import Path

        assert isinstance(result, Path)

    def test_default_appname(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MATCHA_HOME", str(tmp_path))
        result = get_user_data_dir()
        assert result.name == "matcha_tts"

    def test_custom_appname(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MATCHA_HOME", str(tmp_path))
        result = get_user_data_dir("my_custom_app")
        assert result.name == "my_custom_app"

    def test_directory_exists(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MATCHA_HOME", str(tmp_path))
        result = get_user_data_dir("test_matcha_dir")
        assert result.is_dir()

    def test_respects_env_var(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MATCHA_HOME", str(tmp_path))
        result = get_user_data_dir("myapp")
        assert result == tmp_path / "myapp"
        assert result.is_dir()


# ---------------------------------------------------------------------------
# plot_tensor / save_figure_to_numpy
# ---------------------------------------------------------------------------


class TestPlotTensor:
    def test_returns_numpy_array(self):
        tensor = np.random.randn(10, 20)
        result = plot_tensor(tensor)
        assert isinstance(result, np.ndarray)

    def test_output_has_3_channels(self):
        tensor = np.random.randn(10, 20)
        result = plot_tensor(tensor)
        assert result.ndim == 3
        assert result.shape[2] == 3  # RGB

    def test_output_dtype(self):
        tensor = np.random.randn(10, 20)
        result = plot_tensor(tensor)
        assert result.dtype == np.uint8


class TestSaveFigureToNumpy:
    def test_returns_correct_shape(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot([0, 1], [0, 1])
        fig.canvas.draw()
        data = save_figure_to_numpy(fig)
        plt.close(fig)
        assert isinstance(data, np.ndarray)
        assert data.ndim == 3
        assert data.shape[2] == 3  # RGB, alpha channel stripped

    def test_dtype_is_uint8(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot([0, 1], [0, 1])
        fig.canvas.draw()
        data = save_figure_to_numpy(fig)
        plt.close(fig)
        assert data.dtype == np.uint8
