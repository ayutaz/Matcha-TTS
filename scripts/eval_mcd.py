"""Compute Mel Cepstral Distortion (MCD) between synthesized and reference features.

Synth/ref .npy feature files are paired by identical relative paths and
frame-aligned by truncation to the shorter sequence.

Usage:
    uv run python scripts/eval_mcd.py \
        --synth-dir eval/samples/julius_model \
        --ref-dir eval/samples/mas_baseline \
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


def load_features(path):
    """Load a .npy feature array as (frames, dims).

    Mels saved by generate_eval_samples.py are (1, n_mels, T); squeeze the
    batch dim and transpose mel-first arrays to frames-first.
    """
    arr = np.squeeze(np.load(path))
    if arr.ndim != 2:
        raise ValueError(f"Expected 2-D feature array in {path}, got shape {arr.shape}")
    if arr.shape[0] == 80 and arr.shape[1] != 80:
        arr = arr.T
    return arr


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

    if args.ref_dir is None:
        print("No --ref-dir given; cannot compute MCD", file=sys.stderr)
        return 1
    ref_dir = Path(args.ref_dir)
    if not ref_dir.exists():
        print(f"Reference directory not found: {ref_dir}", file=sys.stderr)
        return 1

    # Pair synth/ref files by path relative to their root dirs
    per_file = []
    skipped = 0
    for synth_path in mel_files:
        rel = synth_path.relative_to(synth_dir)
        ref_path = ref_dir / rel
        if not ref_path.exists():
            skipped += 1
            continue
        synth = load_features(synth_path)
        ref = load_features(ref_path)
        if synth.shape[1] != ref.shape[1]:
            print(f"Skipping {rel}: dim mismatch {synth.shape} vs {ref.shape}", file=sys.stderr)
            skipped += 1
            continue
        n_frames = min(len(synth), len(ref))
        per_file.append({"file": str(rel), "mcd": compute_mcd(synth[:n_frames], ref[:n_frames])})

    if not per_file:
        print("No synth/ref pairs found", file=sys.stderr)
        return 1

    mcds = [entry["mcd"] for entry in per_file]
    report = {
        "n_files": len(mel_files),
        "n_pairs": len(per_file),
        "n_skipped": skipped,
        "mcd_mean": float(np.mean(mcds)),
        "mcd_std": float(np.std(mcds)),
        "per_file": per_file,
        "status": "complete",
    }
    print(f"MCD: {report['mcd_mean']:.3f} +/- {report['mcd_std']:.3f} dB over {len(per_file)} pairs")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    sys.exit(main())
