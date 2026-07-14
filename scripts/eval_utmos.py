"""UTMOS automatic MOS evaluation.

Usage:
    uv run python scripts/eval_utmos.py \
        --wav-dir eval/samples/julius_model \
        --output eval/report/utmos.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

UTMOS_HUB_REPO = "tarepan/SpeechMOS:v1.2.0"
UTMOS_HUB_MODEL = "utmos22_strong"


def load_predictor():
    """Load the UTMOS predictor via torch.hub (lazy import keeps CLI startup cheap)."""
    import torch

    return torch.hub.load(UTMOS_HUB_REPO, UTMOS_HUB_MODEL, trust_repo=True)


def score_wav_files(wav_files, predictor):
    """Score wav files with a UTMOS predictor, returning per-file scores."""
    import soundfile as sf
    import torch

    scores = []
    for wav_path in wav_files:
        audio, sr = sf.read(str(wav_path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        with torch.no_grad():
            score = predictor(torch.from_numpy(audio).unsqueeze(0), sr)
        scores.append({"path": str(wav_path), "utmos": float(score)})
    return scores


def main(argv=None):
    parser = argparse.ArgumentParser(description="UTMOS automatic MOS evaluation")
    parser.add_argument("--wav-dir", type=str, required=True)
    parser.add_argument("--baseline-dir", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args(argv)

    wav_dir = Path(args.wav_dir)
    if not wav_dir.exists():
        print(f"WAV directory not found: {wav_dir}", file=sys.stderr)
        return 1

    wav_files = sorted(wav_dir.rglob("*.wav"))
    print(f"Found {len(wav_files)} wav files")
    if not wav_files:
        print("No wav files to score", file=sys.stderr)
        return 1

    try:
        predictor = load_predictor()
    except Exception as e:  # torch.hub load can fail in many ways (network, cache, import)
        print(f"Failed to load UTMOS predictor via torch.hub: {e}", file=sys.stderr)
        return 1

    scores = score_wav_files(wav_files, predictor)
    utmos_values = [s["utmos"] for s in scores]

    report = {
        "n_wav_files": len(wav_files),
        "utmos_mean": float(np.mean(utmos_values)),
        "utmos_std": float(np.std(utmos_values)),
        "per_file": scores,
        "status": "complete",
    }
    print(f"UTMOS: {report['utmos_mean']:.3f} +/- {report['utmos_std']:.3f} over {len(scores)} files")

    if args.baseline_dir:
        baseline_dir = Path(args.baseline_dir)
        if baseline_dir.exists():
            baseline_scores = score_wav_files(sorted(baseline_dir.rglob("*.wav")), predictor)
            if baseline_scores:
                report["baseline_utmos_mean"] = float(np.mean([s["utmos"] for s in baseline_scores]))
                report["baseline_per_file"] = baseline_scores
        else:
            print(f"Baseline directory not found: {baseline_dir}", file=sys.stderr)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    sys.exit(main())
