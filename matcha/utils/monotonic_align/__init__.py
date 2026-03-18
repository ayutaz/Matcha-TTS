import numpy as np
import torch

from matcha.utils.monotonic_align.core import maximum_path_c


@torch.jit.script
def maximum_path_pytorch(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Pure PyTorch implementation of Monotonic Alignment Search (MAS).

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
    neg_inf = torch.tensor(-1e9, device=device, dtype=torch.float32)

    # Forward DP: accumulate best path values in-place
    # For y=0, only x=0 is valid (range is max(0, t_x+0-t_y)..min(t_x, 1) which
    # includes x=0 for typical cases). At y=0,x=0 the Cython code sets v_prev=0
    # and v_cur=max_neg_val, so value[0,0] = max(max_neg_val, 0) + value[0,0] = value[0,0].
    # This is already correct (no update needed for y=0, x=0 as max(neg_inf, 0)=0).
    for y in range(1, t_y):
        # Valid x range per sample: [max(0, t_x_max + y - t_y_max), min(t_x_max, y+1))
        # We iterate over all possible x positions and use masking for per-sample validity.
        x_lo = 0 if (t_x + y - t_y) < 0 else (t_x + y - t_y)
        x_hi = min(t_x, y + 1)
        for x in range(x_lo, x_hi):
            # Per-sample validity: x must be in [max(0, t_x_max+y-t_y_max), min(t_x_max, y+1))
            x_valid_lo = torch.clamp(t_x_max + y - t_y_max, min=0)  # [b]
            x_valid_hi = torch.clamp(t_y_max, max=t_x)  # not needed past y+1
            x_valid_hi = torch.where(
                torch.tensor(y + 1, device=device) < t_x_max,
                torch.tensor(y + 1, device=device, dtype=torch.long),
                t_x_max,
            )
            valid = (x >= x_valid_lo) & (x < x_valid_hi)  # [b]

            # v_cur: value[:, x, y-1], but -inf if x == y
            if x == y:
                v_cur = neg_inf.expand(b)
            else:
                v_cur = value[:, x, y - 1]

            # v_prev: value[:, x-1, y-1], but -inf if x == 0 (and 0 if x==0,y==0 — not reached since y>=1)
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
        # Only set path if y < t_y_max for this sample
        y_valid = torch.tensor(y, device=device) < t_y_max  # [b]
        path[batch_idx, index, y] = torch.where(y_valid, torch.ones(1, device=device, dtype=dtype), path[batch_idx, index, y])

        if y > 0:
            # Decide whether to step index back:
            # if index != 0 and (index == y or value[index, y-1] < value[index-1, y-1])
            can_step = (index > 0) & y_valid
            at_diagonal = index == y
            # value at [index, y-1] vs [index-1, y-1]
            val_cur = value[batch_idx, index, y - 1]
            val_prev = value[batch_idx, torch.clamp(index - 1, min=0), y - 1]
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
