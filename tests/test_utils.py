"""Tests for Matcha-TTS utility functions."""

import matplotlib
import numpy as np
import pytest
import torch

matplotlib.use("Agg")

from matcha.utils.model import fix_len_compatibility, generate_path, sequence_mask
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


# ---------------------------------------------------------------------------
# get_user_data_dir
# ---------------------------------------------------------------------------


class TestGetUserDataDir:
    def test_returns_path(self):
        result = get_user_data_dir()
        from pathlib import Path

        assert isinstance(result, Path)

    def test_default_appname(self):
        result = get_user_data_dir()
        assert result.name == "matcha_tts"

    def test_custom_appname(self):
        result = get_user_data_dir("my_custom_app")
        assert result.name == "my_custom_app"

    def test_directory_exists(self):
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
