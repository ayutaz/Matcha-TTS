"""Evaluate degeneration rate of predicted durations.

Uses matcha.utils.alignment_metrics shared library (M1 T-M1-04).

Usage:
    uv run python scripts/eval_degeneration_rate.py \
        --pred-dir eval/samples/julius_model \
        --output eval/report/degeneration.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from matcha.utils.alignment_metrics import is_degenerate


def load_durations_from_json(pred_dir):
    """Load duration arrays from _dur.json files."""
    all_durations = []
    for json_path in sorted(Path(pred_dir).rglob("*_dur.json")):
        data = json.loads(json_path.read_text(encoding="utf-8"))
        if "predicted_durations" in data:
            dur = np.array(data["predicted_durations"], dtype=np.int64)
            all_durations.append(
                {
                    "path": str(json_path),
                    "speaker_id": data.get("speaker_id", -1),
                    "durations": dur,
                }
            )
    return all_durations


def compute_degeneration_report(all_durations):
    """Compute degeneration statistics."""
    if not all_durations:
        return {"error": "No data"}

    total = len(all_durations)
    degenerate_count = sum(1 for d in all_durations if is_degenerate(d["durations"]))

    # Aggregate phoneme stats
    all_phoneme_durs = []
    all_blank0_durs = []
    for d in all_durations:
        dur = d["durations"]
        all_phoneme_durs.extend(dur[1::2].tolist())
        if len(dur) > 0:
            all_blank0_durs.append(int(dur[0]))

    ph = np.array(all_phoneme_durs, dtype=float)

    return {
        "total_samples": total,
        "degenerate_count": degenerate_count,
        "degenerate_rate": degenerate_count / total,
        "phoneme_le1_frame_rate": float((ph <= 1).mean()) if len(ph) > 0 else 0,
        "phoneme_le2_frame_rate": float((ph <= 2).mean()) if len(ph) > 0 else 0,
        "phoneme_median_duration": float(np.median(ph)) if len(ph) > 0 else 0,
        "blank0_duration_mean": float(np.mean(all_blank0_durs)) if all_blank0_durs else 0,
        "blank0_duration_max": int(np.max(all_blank0_durs)) if all_blank0_durs else 0,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate degeneration rate of predicted durations")
    parser.add_argument("--pred-dir", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args(argv)

    pred_dir = Path(args.pred_dir)
    if not pred_dir.exists():
        print(f"Directory not found: {pred_dir}", file=sys.stderr)
        return 1

    all_durations = load_durations_from_json(pred_dir)
    print(f"Loaded {len(all_durations)} samples")

    if not all_durations:
        print("No duration data found")
        return 1

    report = compute_degeneration_report(all_durations)

    print("\n=== Degeneration Report ===")
    print(f"Total samples:        {report['total_samples']}")
    print(f"Degenerate:           {report['degenerate_count']} ({report['degenerate_rate']:.1%})")
    print(f"Phoneme <= 1 frame:   {report['phoneme_le1_frame_rate']:.1%}")
    print(f"Phoneme <= 2 frames:  {report['phoneme_le2_frame_rate']:.1%}")
    print(f"Phoneme median dur:   {report['phoneme_median_duration']:.1f}")
    print(f"Blank[0] mean dur:    {report['blank0_duration_mean']:.1f}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    sys.exit(main())
