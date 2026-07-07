"""Extract a deployable WaveNeXt generator bin from a training checkpoint.

A training .ckpt holds the whole WaveNeXtExp (backbone./head./multiperioddisc./
multiresddisc./n_batches). matcha.cli.load_wavenext asserts NO unexpected keys, so a raw
training ckpt cannot be loaded directly. This keeps only backbone.* / head.* — exactly the
keys of matcha.wavenext.WaveNeXtVocoder — so the output round-trips through load_wavenext.
EMA weights are preferred when present.

Usage:
    uv run python scripts/extract_wavenext_generator.py --ckpt <train.ckpt> --out wavenext_ja_11025.bin
"""

import argparse
from collections import OrderedDict

import torch


def extract(ckpt, prefer_ema=True):
    """Return an OrderedDict of backbone.*/head.* only (EMA-preferred)."""
    if prefer_ema and isinstance(ckpt, dict) and ckpt.get("ema_state_dict"):
        source = ckpt["ema_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        source = ckpt["state_dict"]
    else:
        source = ckpt
    return OrderedDict((k, v) for k, v in source.items() if k.startswith(("backbone.", "head.")))


def main(argv=None):
    p = argparse.ArgumentParser(description="training .ckpt -> deployable WaveNeXt generator bin")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--no-ema", action="store_true", help="use raw weights even if EMA is present")
    args = p.parse_args(argv)
    sd = extract(torch.load(args.ckpt, map_location="cpu", weights_only=False), prefer_ema=not args.no_ema)
    torch.save(sd, args.out)
    print(f"[+] wrote {args.out}: {len(sd)} backbone./head. tensors")
    # Round-trip verify against the deployment loader.
    try:
        from matcha.cli import load_wavenext

        load_wavenext(args.out, "cpu")
        print("[+] load_wavenext round-trip OK")
    except Exception as exc:  # noqa: BLE001
        print(f"[!] load_wavenext round-trip FAILED: {exc}")
        raise


if __name__ == "__main__":
    main()
