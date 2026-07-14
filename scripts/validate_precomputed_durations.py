"""Validate precomputed .pt files contain correctly shaped durations.

Usage:
    uv run python scripts/validate_precomputed_durations.py \
        --pt-dir data/jvs_precomputed_v2/train \
        --expect-durations
"""

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm


def validate(pt_dir: Path, expect_durations: bool) -> dict:
    """Validate .pt files and return stats dict."""
    pt_files = sorted(pt_dir.glob("*.pt"))

    stats = {
        "total": 0,
        "with_durations": 0,
        "without_durations": 0,
        "errors": [],
        "dur_sum_mismatches": 0,
        "negative_durations": 0,
        "nan_mels": 0,
    }

    for pt_path in tqdm(pt_files, desc="Validating"):
        stats["total"] += 1
        try:
            data = torch.load(pt_path, weights_only=True)
        except Exception as e:
            stats["errors"].append((pt_path.name, f"Load error: {e}"))
            continue

        # 必須キー確認
        for key in ["mel", "text", "spk", "cleaned_text"]:
            if key not in data:
                stats["errors"].append((pt_path.name, f"Missing key: {key}"))

        text = data.get("text")
        mel = data.get("mel")

        if text is None or mel is None:
            continue

        # NaN check
        if torch.isnan(mel).any():
            stats["nan_mels"] += 1
            stats["errors"].append((pt_path.name, "NaN in mel"))

        # Duration check
        if "durations" in data and data["durations"] is not None:
            dur = data["durations"]
            stats["with_durations"] += 1

            # 長さ一致
            if len(dur) != len(text):
                stats["errors"].append((pt_path.name, f"Duration len ({len(dur)}) != text len ({len(text)})"))

            # duration合計 vs mel長（generate_pathが暗黙に仮定）
            dur_sum = dur.sum().item()
            mel_len = mel.shape[-1]
            if dur_sum != mel_len:
                stats["dur_sum_mismatches"] += 1
                stats["errors"].append((pt_path.name, f"Duration sum ({dur_sum}) != mel len ({mel_len})"))

            # 負値チェック
            if (dur < 0).any():
                stats["negative_durations"] += 1
                stats["errors"].append((pt_path.name, "Negative duration values"))
        else:
            stats["without_durations"] += 1
            if expect_durations:
                stats["errors"].append((pt_path.name, "Missing 'durations' key"))

    return stats


def print_report(stats: dict, expect_durations: bool):
    """Print human-readable report."""
    print(f"\n{'=' * 50}")
    print("Validation Report")
    print(f"{'=' * 50}")
    print(f"Total files:          {stats['total']}")
    print(f"With durations:       {stats['with_durations']}")
    print(f"Without durations:    {stats['without_durations']}")
    print(f"Duration sum != mel:  {stats['dur_sum_mismatches']}")
    print(f"Negative durations:   {stats['negative_durations']}")
    print(f"NaN in mel:           {stats['nan_mels']}")
    print(f"Errors:               {len(stats['errors'])}")

    if stats["errors"]:
        print("\nErrors (first 20):")
        for name, msg in stats["errors"][:20]:
            print(f"  {name}: {msg}")
        if len(stats["errors"]) > 20:
            print(f"  ... and {len(stats['errors']) - 20} more")

    if expect_durations and stats["without_durations"] > 0:
        print(f"\nWARNING: {stats['without_durations']} files missing durations!")

    if len(stats["errors"]) == 0:
        print(f"\nAll {stats['total']} files passed validation.")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Validate precomputed .pt files")
    parser.add_argument("--pt-dir", type=str, required=True, help="Directory with .pt files")
    parser.add_argument("--expect-durations", action="store_true", help="Require all .pt files to contain durations")
    args = parser.parse_args(argv)

    pt_dir = Path(args.pt_dir)
    if not pt_dir.exists():
        print(f"ERROR: Directory not found: {pt_dir}", file=sys.stderr)
        return 1

    stats = validate(pt_dir, args.expect_durations)
    print_report(stats, args.expect_durations)

    if stats["errors"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
