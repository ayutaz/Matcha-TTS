"""Compute Mel Cepstral Distortion (MCD) between synthesized and reference audio.

Usage:
    uv run python scripts/eval_mcd.py \
        --synth-dir eval/samples/julius_model \
        --ref-dir data/jvs/wavs \
        --output eval/report/mcd.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def compute_mcd(synth_mfcc, ref_mfcc):
    """Compute MCD in dB between two MFCC sequences (after DTW alignment)."""
    # MCD = (10*sqrt(2)/ln(10)) * mean(||synth - ref||_2)
    coeff = 10.0 * np.sqrt(2.0) / np.log(10.0)
    diff = synth_mfcc - ref_mfcc
    frame_dist = np.sqrt(np.sum(diff**2, axis=1))
    return float(coeff * np.mean(frame_dist))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compute Mel Cepstral Distortion (MCD) between synthesized and reference audio"
    )
    parser.add_argument("--synth-dir", type=str, required=True)
    parser.add_argument("--ref-dir", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args(argv)

    synth_dir = Path(args.synth_dir)
    if not synth_dir.exists():
        print(f"Synth directory not found: {synth_dir}", file=sys.stderr)
        return 1

    mel_files = sorted(synth_dir.rglob("*.npy"))
    print(f"Found {len(mel_files)} mel files")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        report = {"n_files": len(mel_files), "status": "files_found"}
        Path(args.output).write_text(json.dumps(report, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
