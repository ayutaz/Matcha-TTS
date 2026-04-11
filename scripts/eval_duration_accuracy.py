"""Evaluate Duration Predictor accuracy.

Usage:
    uv run python scripts/eval_duration_accuracy.py \
        --pred-dir eval/samples/julius_model \
        --gt-dir /dev/shm/jvs_precomputed_aligned/val \
        --output eval/report/duration_accuracy.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats


def load_predicted_durations(pred_dir):
    """Load predicted durations from _dur.json files."""
    results = []
    for json_path in sorted(Path(pred_dir).rglob("*_dur.json")):
        data = json.loads(json_path.read_text())
        if "predicted_durations" in data:
            results.append(
                {
                    "path": str(json_path),
                    "speaker_id": data.get("speaker_id", -1),
                    "durations": data["predicted_durations"],
                }
            )
    return results


def compute_accuracy(pred_durations, gt_durations):
    """Compute duration accuracy metrics (phoneme positions only)."""
    pred = np.array(pred_durations, dtype=float)
    gt = np.array(gt_durations, dtype=float)

    # Phoneme positions (odd indices in blank-interspersed)
    min_len = min(len(pred), len(gt))
    pred_ph = pred[1:min_len:2]
    gt_ph = gt[1:min_len:2]

    if len(pred_ph) == 0 or len(gt_ph) == 0:
        return None

    mae = float(np.abs(pred_ph - gt_ph).mean())
    rmse = float(np.sqrt(np.mean((pred_ph - gt_ph) ** 2)))

    if len(pred_ph) > 2 and np.std(gt_ph) > 0:
        corr, _ = stats.pearsonr(pred_ph, gt_ph)
        corr = float(corr)
    else:
        corr = float("nan")

    # Relative error
    nonzero_mask = gt_ph > 0
    if nonzero_mask.any():
        rel_err = float(
            (np.abs(pred_ph[nonzero_mask] - gt_ph[nonzero_mask]) / gt_ph[nonzero_mask]).mean()
        )
    else:
        rel_err = float("nan")

    return {"mae": mae, "rmse": rmse, "pearson_r": corr, "relative_error": rel_err}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate Duration Predictor accuracy")
    parser.add_argument("--pred-dir", type=str, required=True)
    parser.add_argument("--gt-dir", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args(argv)

    pred_dir = Path(args.pred_dir)
    if not pred_dir.exists():
        print(f"Prediction directory not found: {pred_dir}", file=sys.stderr)
        return 1

    predictions = load_predicted_durations(pred_dir)
    print(f"Loaded {len(predictions)} predicted duration files")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        report = {"n_predictions": len(predictions), "status": "predictions_loaded"}
        Path(args.output).write_text(json.dumps(report, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
