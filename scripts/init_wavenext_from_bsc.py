"""Build a WaveNeXt training init state_dict from the BSC-LT/wavenext-mel weights.

The BSC checkpoint (fmax=8000) has no fmax-dependent parameters, so its backbone./head.
weights are a valid warm start for fmax=11025 training (input_channels=80 unchanged) —
far better than random init. Discriminators are not published, so they start from scratch.

Usage:
    uv run python scripts/init_wavenext_from_bsc.py --bsc <BSC pytorch_model.bin> --out init.bin
"""

import argparse
from collections import OrderedDict

import torch


def build_init_state_dict(bsc):
    """Keep only backbone.* / head.* (drop the BSC feature_extractor.*)."""
    if isinstance(bsc, dict) and "state_dict" in bsc:
        bsc = bsc["state_dict"]
    return OrderedDict((k, v) for k, v in bsc.items() if k.startswith(("backbone.", "head.")))


def main(argv=None):
    p = argparse.ArgumentParser(description="BSC wavenext weights -> training init state_dict")
    p.add_argument("--bsc", required=True, help="BSC-LT/wavenext-mel pytorch_model.bin")
    p.add_argument("--out", required=True, help="output init state_dict (.bin)")
    args = p.parse_args(argv)
    sd = build_init_state_dict(torch.load(args.bsc, map_location="cpu", weights_only=True))
    torch.save(sd, args.out)
    print(f"[+] wrote {args.out}: {len(sd)} backbone./head. tensors")


if __name__ == "__main__":
    main()
