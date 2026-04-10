"""Evaluate Duration Predictor accuracy against precomputed duration targets.

Usage:
    uv run python scripts/evaluate_durations.py \
        --checkpoint logs/train/jvs_aligned/runs/.../checkpoints/last.ckpt \
        --data-dir /dev/shm/jvs_precomputed_aligned/val \
        --output-dir logs/eval/durations \
        --max-samples 500
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


def load_model_from_checkpoint(ckpt_path):
    """Load MatchaTTS model from checkpoint.

    Returns the model in eval mode on CPU.  The checkpoint must contain
    ``hyper_parameters`` saved by PyTorch Lightning's
    ``save_hyperparameters()``.
    """
    from matcha.models.matcha_tts import MatchaTTS

    model = MatchaTTS.load_from_checkpoint(ckpt_path, map_location="cpu")
    model.eval()
    return model


def evaluate_durations(
    model,
    data_dir: Path,
    max_samples: int = 500,
    device: str = "cpu",
) -> dict:
    """Evaluate duration prediction accuracy.

    For each sample:
    1. Load precomputed .pt (with text, mel, durations, spk)
    2. Run encoder forward to get predicted logw
    3. Convert logw to predicted durations: exp(logw)
    4. Compare predicted vs target durations

    Returns:
        dict with MAE, degenerate rate, per-phoneme stats
    """
    from matcha.utils.alignment_metrics import is_degenerate

    pt_files = sorted(data_dir.glob("*.pt"))[:max_samples]

    all_mae = []
    all_pred_durations = []
    all_target_durations = []
    degenerate_count = 0

    model = model.to(device)

    for pt_path in tqdm(pt_files, desc="Evaluating"):
        data = torch.load(pt_path, weights_only=True)
        text = data["text"].unsqueeze(0).to(device)  # (1, T)
        target_dur = data.get("durations", None)
        if target_dur is None:
            continue

        spk = data["spk"]
        spk_tensor = torch.tensor([spk], dtype=torch.long, device=device)
        x_lengths = torch.tensor([text.shape[1]], device=device)

        # Run encoder to get predicted log-durations
        with torch.no_grad():
            if model.n_spks > 1:
                spk_emb = model.spk_emb(spk_tensor)
            else:
                spk_emb = None
            mu_x, logw, x_mask = model.encoder(text, x_lengths, spks=spk_emb)

        # Predicted durations: exp(logw), rounded
        pred_dur = torch.exp(logw).squeeze(0).squeeze(0)  # (T,)
        pred_dur_rounded = torch.round(pred_dur).long().cpu()

        # Target durations (already blank-interspersed)
        target = target_dur.long()

        # MAE (phoneme positions only, odd indices)
        if len(pred_dur_rounded) == len(target):
            phoneme_pred = pred_dur_rounded[1::2].float()
            phoneme_target = target[1::2].float()
            mae = (phoneme_pred - phoneme_target).abs().mean().item()
            all_mae.append(mae)

            all_pred_durations.extend(pred_dur_rounded[1::2].tolist())
            all_target_durations.extend(target[1::2].tolist())

            # Check degenerate on predicted durations
            if is_degenerate(pred_dur_rounded.numpy()):
                degenerate_count += 1

    n_samples = len(all_mae)
    if n_samples == 0:
        return {"error": "No valid samples found"}

    pred_arr = np.array(all_pred_durations, dtype=float)
    target_arr = np.array(all_target_durations, dtype=float)

    return {
        "n_samples": n_samples,
        "mae_mean": float(np.mean(all_mae)),
        "mae_median": float(np.median(all_mae)),
        "mae_std": float(np.std(all_mae)),
        "degenerate_count": degenerate_count,
        "degenerate_rate": degenerate_count / n_samples,
        "pred_duration_stats": {
            "mean": float(pred_arr.mean()),
            "median": float(np.median(pred_arr)),
            "std": float(pred_arr.std()),
        },
        "target_duration_stats": {
            "mean": float(target_arr.mean()),
            "median": float(np.median(target_arr)),
            "std": float(target_arr.std()),
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate Duration Predictor accuracy")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args(argv)

    # Validate data directory
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}", file=sys.stderr)
        return 1

    pt_count = len(list(data_dir.glob("*.pt")))
    print(f"Found {pt_count} .pt files in {data_dir}")
    print(f"Will evaluate up to {args.max_samples} samples")
    print(f"Checkpoint: {args.checkpoint}")

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir}")

    # Full evaluation requires checkpoint loading
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        print("Skipping evaluation (checkpoint required)")
        return 0

    # Load model and run evaluation
    print("Loading model from checkpoint...")
    model = load_model_from_checkpoint(str(ckpt_path))
    print(f"Model loaded: n_spks={model.n_spks}, n_vocab={model.n_vocab}")

    results = evaluate_durations(
        model,
        data_dir,
        max_samples=args.max_samples,
        device=args.device,
    )

    # Print results
    print(f"\n{'='*50}")
    print("Duration Prediction Evaluation Results")
    print(f"{'='*50}")
    if "error" in results:
        print(f"Error: {results['error']}")
        return 1

    print(f"Samples evaluated:    {results['n_samples']}")
    print(f"MAE (mean):           {results['mae_mean']:.2f} frames")
    print(f"MAE (median):         {results['mae_median']:.2f} frames")
    print(f"MAE (std):            {results['mae_std']:.2f} frames")
    print(f"Degenerate count:     {results['degenerate_count']}")
    print(f"Degenerate rate:      {results['degenerate_rate']:.1%}")
    print(f"\nPredicted duration stats:")
    print(f"  mean={results['pred_duration_stats']['mean']:.2f}, "
          f"median={results['pred_duration_stats']['median']:.2f}, "
          f"std={results['pred_duration_stats']['std']:.2f}")
    print(f"Target duration stats:")
    print(f"  mean={results['target_duration_stats']['mean']:.2f}, "
          f"median={results['target_duration_stats']['median']:.2f}, "
          f"std={results['target_duration_stats']['std']:.2f}")

    # Save results as JSON if output_dir specified
    if args.output_dir:
        output_path = Path(args.output_dir) / "duration_eval_results.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
