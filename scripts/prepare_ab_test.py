"""Prepare A/B test sample pairs for subjective evaluation.

Usage:
    uv run python scripts/prepare_ab_test.py \
        --model-a-dir eval/samples/julius_model \
        --model-b-dir eval/samples/mas_baseline \
        --output-dir eval/ab_test \
        --n-pairs 50
"""
import argparse
import json
import random
import sys
from pathlib import Path


def select_pairs(model_a_dir, model_b_dir, n_pairs=50, seed=42):
    """Select matching wav pairs from two model directories."""
    rng = random.Random(seed)

    a_wavs = {p.relative_to(model_a_dir): p for p in sorted(Path(model_a_dir).rglob("*.wav"))}
    b_wavs = {p.relative_to(model_b_dir): p for p in sorted(Path(model_b_dir).rglob("*.wav"))}

    common = sorted(set(a_wavs.keys()) & set(b_wavs.keys()))
    if len(common) < n_pairs:
        n_pairs = len(common)

    selected = rng.sample(common, n_pairs)

    pairs = []
    for rel_path in selected:
        # Randomize A/B order
        if rng.random() < 0.5:
            pairs.append(
                {
                    "A": str(a_wavs[rel_path]),
                    "B": str(b_wavs[rel_path]),
                    "A_is": "julius",
                    "B_is": "mas",
                }
            )
        else:
            pairs.append(
                {
                    "A": str(b_wavs[rel_path]),
                    "B": str(a_wavs[rel_path]),
                    "A_is": "mas",
                    "B_is": "julius",
                }
            )

    return pairs


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Prepare A/B test sample pairs for subjective evaluation"
    )
    parser.add_argument("--model-a-dir", type=str, required=True)
    parser.add_argument("--model-b-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--n-pairs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    for d in [args.model_a_dir, args.model_b_dir]:
        if not Path(d).exists():
            print(f"Directory not found: {d}", file=sys.stderr)
            return 1

    pairs = select_pairs(args.model_a_dir, args.model_b_dir, args.n_pairs, args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "pairs.json").write_text(json.dumps(pairs, indent=2))
    print(f"Selected {len(pairs)} A/B test pairs")

    return 0


if __name__ == "__main__":
    sys.exit(main())
