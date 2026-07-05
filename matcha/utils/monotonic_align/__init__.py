import numpy as np
import torch

from matcha.utils.monotonic_align.core import maximum_path_c


@torch.jit.script
def maximum_path_pytorch_original(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Original PyTorch implementation of Monotonic Alignment Search (MAS).

    Kept as a reference / backup. The optimized version is maximum_path_pytorch.

    Args:
        value: [b, t_x, t_y] — alignment scores (already masked)
        mask: [b, t_x, t_y] — binary mask

    Returns:
        path: [b, t_x, t_y] — binary monotonic alignment path
    """
    device = value.device
    dtype = value.dtype
    b, t_x, t_y = value.shape

    # Compute per-sample effective lengths from the mask
    t_x_max = mask[:, :, 0].sum(dim=1).to(torch.long)  # [b]
    t_y_max = mask[:, 0, :].sum(dim=1).to(torch.long)  # [b]

    # Work in float32 for numerical stability
    value = value.float()
    neg_inf = torch.tensor(-1e9, device=device, dtype=torch.float32)

    for y in range(1, t_y):
        x_lo = 0 if (t_x + y - t_y) < 0 else (t_x + y - t_y)
        x_hi = min(t_x, y + 1)
        for x in range(x_lo, x_hi):
            x_valid_lo = torch.clamp(t_x_max + y - t_y_max, min=0)  # [b]
            x_valid_hi = torch.where(
                torch.tensor(y + 1, device=device) < t_x_max,
                torch.tensor(y + 1, device=device, dtype=torch.long),
                t_x_max,
            )
            valid = (x >= x_valid_lo) & (x < x_valid_hi)  # [b]

            if x == y:
                v_cur = neg_inf.expand(b)
            else:
                v_cur = value[:, x, y - 1]

            if x == 0:
                v_prev = neg_inf.expand(b)
            else:
                v_prev = value[:, x - 1, y - 1]

            new_val = torch.max(v_cur, v_prev) + value[:, x, y]
            value[:, x, y] = torch.where(valid, new_val, value[:, x, y])

    # Backward traceback
    path = torch.zeros(b, t_x, t_y, device=device, dtype=dtype)
    index = t_x_max - 1  # [b], starting x index per sample

    batch_idx = torch.arange(b, device=device)

    for y in range(t_y - 1, -1, -1):
        y_valid = torch.tensor(y, device=device) < t_y_max  # [b]
        path[batch_idx, index, y] = torch.where(
            y_valid, torch.ones(1, device=device, dtype=dtype), path[batch_idx, index, y]
        )

        if y > 0:
            can_step = (index > 0) & y_valid
            at_diagonal = index == y
            val_cur = value[batch_idx, index, y - 1]
            val_prev = value[batch_idx, torch.clamp(index - 1, min=0), y - 1]
            prefer_prev = val_cur < val_prev

            do_step = can_step & (at_diagonal | prefer_prev)
            index = torch.where(do_step, index - 1, index)

    return path


@torch.jit.script
def maximum_path_pytorch(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Optimized PyTorch implementation of Monotonic Alignment Search (MAS).

    Vectorizes the inner x-loop in the forward DP phase so that all valid x
    positions for a given y are updated in a single batched tensor operation,
    drastically reducing the number of small CUDA kernel launches.

    The backward traceback is also streamlined with fewer tensor allocations
    per loop iteration.

    Stays entirely on GPU, avoiding CPU-GPU data transfers.
    Produces identical results to the Cython maximum_path_c.

    Args:
        value: [b, t_x, t_y] — alignment scores (already masked)
        mask: [b, t_x, t_y] — binary mask

    Returns:
        path: [b, t_x, t_y] — binary monotonic alignment path
    """
    device = value.device
    dtype = value.dtype
    b, t_x, t_y = value.shape

    # Compute per-sample effective lengths from the mask
    t_x_max = mask[:, :, 0].sum(dim=1).to(torch.long)  # [b]
    t_y_max = mask[:, 0, :].sum(dim=1).to(torch.long)  # [b]

    # Work in float32 for numerical stability
    value = value.float()
    neg_inf_val: float = -1e9

    # ---------------------------------------------------------------
    # Forward DP — vectorized over x for each y step
    # ---------------------------------------------------------------
    # Pre-build an x-index vector [0, 1, ..., t_x-1] for broadcasting
    x_idx = torch.arange(t_x, device=device, dtype=torch.long)  # [t_x]

    for y in range(1, t_y):
        # Global x range that could be valid for any sample at this y
        x_lo = max(0, t_x + y - t_y)
        x_hi = min(t_x, y + 1)
        n_x = x_hi - x_lo
        if n_x <= 0:
            continue

        # Slice the x range we need to update: x_lo..x_hi-1
        # x_range: [n_x]   (the actual x indices in this slice)
        x_range = x_idx[x_lo:x_hi]  # [n_x]

        # Per-sample validity: x must be in [max(0, t_x_max+y-t_y_max), min(t_x_max, y+1))
        x_valid_lo = torch.clamp(t_x_max + y - t_y_max, min=0)  # [b]
        x_valid_hi = torch.where(
            t_x_max < (y + 1),
            t_x_max,
            torch.tensor(y + 1, device=device, dtype=torch.long),
        )  # [b]
        # valid[b, n_x]: whether position x_range[j] is valid for sample b
        valid = (x_range.unsqueeze(0) >= x_valid_lo.unsqueeze(1)) & (
            x_range.unsqueeze(0) < x_valid_hi.unsqueeze(1)
        )  # [b, n_x]

        # ---- v_cur: value[:, x, y-1] but -inf where x == y ----
        v_cur = value[:, x_lo:x_hi, y - 1]  # [b, n_x]
        # Mask positions where x == y (on the diagonal)
        diag_mask = x_range == y  # [n_x]  (at most one True)
        if diag_mask.any():
            v_cur = torch.where(diag_mask.unsqueeze(0), torch.tensor(neg_inf_val, device=device), v_cur)

        # ---- v_prev: value[:, x-1, y-1] but -inf where x == 0 ----
        if x_lo == 0:
            # First position in slice is x=0 -> needs -inf for v_prev
            # Rest get value[:, x-1, y-1]
            if n_x == 1:
                v_prev = torch.full((b, 1), neg_inf_val, device=device)
            else:
                v_prev_rest = value[:, x_lo : x_hi - 1, y - 1]  # [b, n_x-1]
                v_prev_first = torch.full((b, 1), neg_inf_val, device=device)
                v_prev = torch.cat([v_prev_first, v_prev_rest], dim=1)  # [b, n_x]
        else:
            # All positions have x >= 1, so x-1 >= 0
            v_prev = value[:, x_lo - 1 : x_hi - 1, y - 1]  # [b, n_x]

        # ---- DP update ----
        new_val = torch.max(v_cur, v_prev) + value[:, x_lo:x_hi, y]  # [b, n_x]
        value[:, x_lo:x_hi, y] = torch.where(valid, new_val, value[:, x_lo:x_hi, y])

    # ---------------------------------------------------------------
    # Backward traceback — vectorized over batch
    # ---------------------------------------------------------------
    path = torch.zeros(b, t_x, t_y, device=device, dtype=dtype)
    index = t_x_max - 1  # [b], starting x index per sample
    batch_idx = torch.arange(b, device=device)
    one_val = torch.tensor(1.0, device=device, dtype=dtype)
    zero_idx = torch.tensor(0, device=device, dtype=torch.long)

    for y in range(t_y - 1, -1, -1):
        # Only set path if y < t_y_max for this sample
        y_valid = y < t_y_max  # [b]
        path[batch_idx, index, y] = torch.where(y_valid, one_val, path[batch_idx, index, y])

        if y > 0:
            # Decide whether to step index back
            can_step = (index > 0) & y_valid
            at_diagonal = index == y
            val_cur = value[batch_idx, index, y - 1]
            val_prev = value[batch_idx, torch.max(index - 1, zero_idx), y - 1]
            prefer_prev = val_cur < val_prev

            do_step = can_step & (at_diagonal | prefer_prev)
            index = torch.where(do_step, index - 1, index)

    return path


def maximum_path(value, mask):
    """Cython optimised version.
    value: [b, t_x, t_y]
    mask: [b, t_x, t_y]
    """
    value = value * mask
    device = value.device
    dtype = value.dtype

    if device.type == "cuda":
        return maximum_path_pytorch(value, mask)

    value = value.data.cpu().numpy().astype(np.float32)
    path = np.zeros_like(value).astype(np.int32)
    mask = mask.data.cpu().numpy()

    t_x_max = mask.sum(1)[:, 0].astype(np.int32)
    t_y_max = mask.sum(2)[:, 0].astype(np.int32)
    maximum_path_c(path, value, t_x_max, t_y_max)
    return torch.from_numpy(path).to(device=device, dtype=dtype)
