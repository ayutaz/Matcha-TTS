import numpy as np
import pytest
import torch

from matcha.utils.monotonic_align import maximum_path, maximum_path_pytorch
from matcha.utils.monotonic_align.core import maximum_path_c


@pytest.fixture
def small_inputs():
    """Create small test inputs: batch=2, t_x=5, t_y=10."""
    batch, t_x, t_y = 2, 5, 10
    torch.manual_seed(42)
    value = torch.randn(batch, t_x, t_y)
    mask = torch.ones(batch, t_x, t_y)
    return value, mask


class TestMaximumPathShape:
    def test_output_shape_matches_input(self, small_inputs):
        value, mask = small_inputs
        path = maximum_path(value, mask)
        assert path.shape == value.shape

    def test_single_batch(self):
        value = torch.randn(1, 3, 6)
        mask = torch.ones(1, 3, 6)
        path = maximum_path(value, mask)
        assert path.shape == (1, 3, 6)

    def test_square_dimensions(self):
        value = torch.randn(2, 4, 4)
        mask = torch.ones(2, 4, 4)
        path = maximum_path(value, mask)
        assert path.shape == (2, 4, 4)


class TestMaximumPathBinary:
    def test_output_values_are_binary(self, small_inputs):
        value, mask = small_inputs
        path = maximum_path(value, mask)
        unique_vals = torch.unique(path)
        assert all(v in (0.0, 1.0) for v in unique_vals.tolist())

    def test_output_dtype_matches_input(self, small_inputs):
        value, mask = small_inputs
        path = maximum_path(value, mask)
        assert path.dtype == value.dtype


class TestMaximumPathMask:
    def test_path_is_zero_outside_mask(self):
        """Path values must be zero wherever the mask is zero."""
        batch, t_x, t_y = 2, 5, 10
        torch.manual_seed(0)
        value = torch.randn(batch, t_x, t_y)
        # Mask that clips each sample to different effective lengths.
        mask = torch.ones(batch, t_x, t_y)
        # Sample 0: effective t_x=3, t_y=7
        mask[0, 3:, :] = 0
        mask[0, :, 7:] = 0
        # Sample 1: effective t_x=4, t_y=8
        mask[1, 4:, :] = 0
        mask[1, :, 8:] = 0

        path = maximum_path(value, mask)

        # Outside the masked region the path must be zero.
        assert (path[0, 3:, :] == 0).all()
        assert (path[0, :, 7:] == 0).all()
        assert (path[1, 4:, :] == 0).all()
        assert (path[1, :, 8:] == 0).all()

    def test_path_within_mask_has_ones(self):
        """There must be at least one '1' within the masked region."""
        batch, t_x, t_y = 1, 4, 8
        value = torch.randn(batch, t_x, t_y)
        mask = torch.ones(batch, t_x, t_y)
        path = maximum_path(value, mask)
        assert path.sum() > 0


class TestMonotonicity:
    def _extract_path_coordinates(self, path_2d):
        """Extract (x, y) coordinates of the path from a single 2D path tensor.

        Returns a list of (x, y) tuples sorted by y-coordinate.
        """
        coords = []
        for y in range(path_2d.shape[1]):
            xs = torch.where(path_2d[:, y] == 1)[0]
            for x in xs:
                coords.append((x.item(), y))
        coords.sort(key=lambda c: c[1])
        return coords

    def test_path_is_monotonically_increasing(self, small_inputs):
        """The x-index of the path must never decrease as y increases."""
        value, mask = small_inputs
        path = maximum_path(value, mask)

        for b in range(path.shape[0]):
            coords = self._extract_path_coordinates(path[b])
            x_values = [c[0] for c in coords]
            for i in range(1, len(x_values)):
                assert x_values[i] >= x_values[i - 1], (
                    f"Monotonicity violated at batch {b}: x went from {x_values[i - 1]} to {x_values[i]}"
                )

    def test_each_column_has_exactly_one_active_cell(self, small_inputs):
        """For a fully-masked input, every y-column must have exactly one '1'."""
        value, mask = small_inputs
        path = maximum_path(value, mask)

        for b in range(path.shape[0]):
            col_sums = path[b].sum(dim=0)  # sum over t_x for each t_y
            assert (col_sums == 1).all(), (
                f"Batch {b}: not every column has exactly one active cell. Column sums: {col_sums.tolist()}"
            )

    def test_path_covers_full_x_range(self, small_inputs):
        """The path must start at x=0 and end at x=t_x-1 (for full mask)."""
        value, mask = small_inputs
        path = maximum_path(value, mask)

        t_x = path.shape[1]
        for b in range(path.shape[0]):
            coords = self._extract_path_coordinates(path[b])
            x_values = [c[0] for c in coords]
            assert x_values[0] == 0, f"Batch {b}: path does not start at x=0"
            assert x_values[-1] == t_x - 1, f"Batch {b}: path does not end at x={t_x - 1}"


class TestBatchProcessing:
    def test_batch_results_match_individual(self):
        """Batched computation must produce the same result as processing each sample individually."""
        torch.manual_seed(123)
        batch, t_x, t_y = 3, 4, 8
        value = torch.randn(batch, t_x, t_y)
        mask = torch.ones(batch, t_x, t_y)

        batched_path = maximum_path(value, mask)

        for b in range(batch):
            single_value = value[b : b + 1].clone()
            single_mask = mask[b : b + 1].clone()
            single_path = maximum_path(single_value, single_mask)
            assert torch.equal(batched_path[b], single_path[0]), (
                f"Batch element {b} differs between batched and individual computation"
            )

    def test_independent_samples_in_batch(self):
        """Changing one sample in the batch must not affect other samples' results."""
        torch.manual_seed(7)
        batch, t_x, t_y = 2, 5, 10
        value = torch.randn(batch, t_x, t_y)
        mask = torch.ones(batch, t_x, t_y)

        path_original = maximum_path(value.clone(), mask.clone())

        value_modified = value.clone()
        value_modified[1] = torch.randn(t_x, t_y)
        path_modified = maximum_path(value_modified, mask.clone())

        assert torch.equal(path_original[0], path_modified[0]), "Modifying sample 1 changed the result for sample 0"

    def test_different_lengths_in_batch(self):
        """Samples with different effective lengths via masks are handled correctly."""
        batch, t_x, t_y = 2, 5, 10
        torch.manual_seed(99)
        value = torch.randn(batch, t_x, t_y)
        mask = torch.ones(batch, t_x, t_y)
        # Sample 0: full length; Sample 1: t_x=3, t_y=6
        mask[1, 3:, :] = 0
        mask[1, :, 6:] = 0

        path = maximum_path(value, mask)

        # Both samples should have valid paths.
        assert path[0].sum() > 0
        assert path[1].sum() > 0
        # Sample 1 must have no path outside its mask.
        assert (path[1, 3:, :] == 0).all()
        assert (path[1, :, 6:] == 0).all()


# ---------------------------------------------------------------------------
# Tests comparing PyTorch GPU implementation against Cython reference
# ---------------------------------------------------------------------------


def _run_cython(value, mask):
    """Run the Cython implementation on CPU and return float tensor."""
    value = (value * mask).clone()
    val_np = value.data.cpu().numpy().astype(np.float32)
    path_np = np.zeros_like(val_np).astype(np.int32)
    mask_np = mask.data.cpu().numpy()
    t_x_max = mask_np.sum(1)[:, 0].astype(np.int32)
    t_y_max = mask_np.sum(2)[:, 0].astype(np.int32)
    maximum_path_c(path_np, val_np, t_x_max, t_y_max)
    return torch.from_numpy(path_np).float()


def _run_pytorch(value, mask):
    """Run the pure PyTorch implementation on CPU."""
    value = (value * mask).clone()
    return maximum_path_pytorch(value, mask)


class TestPytorchVsCython:
    """Verify the PyTorch implementation produces identical results to Cython."""

    def test_basic(self):
        torch.manual_seed(42)
        v = torch.randn(2, 5, 10)
        m = torch.ones(2, 5, 10)
        assert torch.equal(_run_cython(v, m), _run_pytorch(v, m))

    def test_square(self):
        torch.manual_seed(7)
        v = torch.randn(3, 4, 4)
        m = torch.ones(3, 4, 4)
        assert torch.equal(_run_cython(v, m), _run_pytorch(v, m))

    def test_single_phoneme(self):
        torch.manual_seed(11)
        v = torch.randn(2, 1, 5)
        m = torch.ones(2, 1, 5)
        assert torch.equal(_run_cython(v, m), _run_pytorch(v, m))

    def test_masked_different_lengths(self):
        torch.manual_seed(99)
        v = torch.randn(2, 5, 10)
        m = torch.ones(2, 5, 10)
        m[0, 3:, :] = 0
        m[0, :, 7:] = 0
        m[1, 4:, :] = 0
        m[1, :, 8:] = 0
        assert torch.equal(_run_cython(v, m), _run_pytorch(v, m))

    def test_larger_batch(self):
        torch.manual_seed(123)
        v = torch.randn(8, 10, 20)
        m = torch.ones(8, 10, 20)
        assert torch.equal(_run_cython(v, m), _run_pytorch(v, m))

    def test_minimal(self):
        v = torch.randn(1, 1, 1)
        m = torch.ones(1, 1, 1)
        assert torch.equal(_run_cython(v, m), _run_pytorch(v, m))

    def test_tall_matrix(self):
        torch.manual_seed(77)
        v = torch.randn(2, 3, 15)
        m = torch.ones(2, 3, 15)
        assert torch.equal(_run_cython(v, m), _run_pytorch(v, m))

    def test_batch_consistency(self):
        torch.manual_seed(200)
        v = torch.randn(4, 6, 12)
        m = torch.ones(4, 6, 12)
        batched = _run_pytorch(v, m)
        for b in range(4):
            single = _run_pytorch(v[b : b + 1], m[b : b + 1])
            assert torch.equal(batched[b], single[0]), f"Batch element {b} differs"

    @pytest.mark.parametrize("seed", list(range(50)))
    def test_fuzz(self, seed):
        """Fuzz test: random shapes and values must match Cython output."""
        import random

        rng = random.Random(seed)
        torch.manual_seed(seed)
        b = rng.randint(1, 8)
        tx = rng.randint(1, 15)
        ty = rng.randint(tx, tx * 4)
        v = torch.randn(b, tx, ty)
        m = torch.ones(b, tx, ty)
        cy = _run_cython(v, m)
        pt = _run_pytorch(v, m)
        assert torch.equal(cy, pt), (
            f"Mismatch at seed={seed}, b={b}, tx={tx}, ty={ty}"
        )

    def test_dtype_preservation(self):
        """PyTorch implementation should return the same dtype as input."""
        torch.manual_seed(42)
        v = torch.randn(2, 5, 10)
        m = torch.ones(2, 5, 10)
        path = _run_pytorch(v, m)
        assert path.dtype == v.dtype

    def test_dispatch_cpu_uses_cython(self):
        """On CPU, maximum_path should dispatch to Cython (not error)."""
        torch.manual_seed(42)
        v = torch.randn(2, 5, 10)
        m = torch.ones(2, 5, 10)
        path = maximum_path(v, m)
        assert path.shape == v.shape
        assert path.device.type == "cpu"
