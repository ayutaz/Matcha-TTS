"""Evaluate Duration Predictor accuracy.

Usage:
    uv run python scripts/eval_duration_accuracy.py \
        --pred-dir eval/samples/julius_model \
        --gt-dir eval/reference_durations \
        --output eval/report/duration_accuracy.json

Both --pred-dir and --gt-dir must contain *_dur.json files (as written by
scripts/generate_eval_samples.py); predictions and references are paired by
their path relative to each root directory.
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
        data = json.loads(json_path.read_text(encoding="utf-8"))
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
        rel_err = float((np.abs(pred_ph[nonzero_mask] - gt_ph[nonzero_mask]) / gt_ph[nonzero_mask]).mean())
    else:
        rel_err = float("nan")

    return {"mae": mae, "rmse": rmse, "pearson_r": corr, "relative_error": rel_err}


def load_reference_durations(gt_dir):
    """Load ground-truth durations from _dur.json files, keyed by path relative to gt_dir."""
    gt_dir = Path(gt_dir)
    references = {}
    for json_path in sorted(gt_dir.rglob("*_dur.json")):
        data = json.loads(json_path.read_text(encoding="utf-8"))
        durations = data.get("durations", data.get("predicted_durations"))
        if durations is not None:
            references[str(json_path.relative_to(gt_dir))] = durations
    return references


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

    if args.gt_dir is None:
        print("No --gt-dir given; cannot compute duration accuracy", file=sys.stderr)
        return 1
    gt_dir = Path(args.gt_dir)
    if not gt_dir.exists():
        print(f"Ground-truth directory not found: {gt_dir}", file=sys.stderr)
        return 1

    # Pair predictions with references by path relative to their root dirs
    references = load_reference_durations(gt_dir)
    per_file = []
    for pred in predictions:
        rel = str(Path(pred["path"]).relative_to(pred_dir))
        gt = references.get(rel)
        if gt is None:
            continue
        metrics = compute_accuracy(pred["durations"], gt)
        if metrics is not None:
            per_file.append({"file": rel, **metrics})

    if not per_file:
        print("No prediction/ground-truth pairs found", file=sys.stderr)
        return 1

    report = {
        "n_predictions": len(predictions),
        "n_pairs": len(per_file),
        "mae_mean": float(np.mean([m["mae"] for m in per_file])),
        "rmse_mean": float(np.mean([m["rmse"] for m in per_file])),
        "pearson_r_mean": float(np.nanmean([m["pearson_r"] for m in per_file])),
        "relative_error_mean": float(np.nanmean([m["relative_error"] for m in per_file])),
        "per_file": per_file,
        "status": "complete",
    }
    print(f"MAE: {report['mae_mean']:.2f} frames, RMSE: {report['rmse_mean']:.2f} over {len(per_file)} pairs")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    sys.exit(main())
