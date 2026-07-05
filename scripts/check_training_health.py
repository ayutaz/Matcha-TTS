"""Check training health from TensorBoard event files.

Usage:
    uv run python scripts/check_training_health.py \
        --log-dir logs/train/jvs_aligned/runs/<run_dir>/
"""

import argparse
import sys
from pathlib import Path


def check_health(log_dir: Path) -> dict:
    """Check training health from log directory.

    Checks:
    1. Checkpoint existence and last.ckpt presence
    2. TensorBoard event file existence
    3. Number of epoch checkpoints saved

    Returns:
        dict with checkpoint counts, TensorBoard event counts, and issues list
    """
    results = {
        "log_dir": str(log_dir),
        "checkpoints_found": 0,
        "last_ckpt_exists": False,
        "tensorboard_events": 0,
        "issues": [],
    }

    # Check checkpoints
    ckpt_dir = log_dir / "checkpoints"
    if ckpt_dir.exists():
        ckpts = list(ckpt_dir.glob("*.ckpt"))
        results["checkpoints_found"] = len(ckpts)
        results["last_ckpt_exists"] = (ckpt_dir / "last.ckpt").exists()
    else:
        results["issues"].append("No checkpoints directory found")

    # Check TensorBoard events -- look in tensorboard/ subdir first,
    # then fall back to searching the whole log dir if none were found there
    tb_dir = log_dir / "tensorboard"
    events = []
    if tb_dir.exists():
        events = list(tb_dir.rglob("events.out.tfevents.*"))
    if not events:
        events = list(log_dir.rglob("events.out.tfevents.*"))
    results["tensorboard_events"] = len(events)

    if results["tensorboard_events"] == 0:
        results["issues"].append("No TensorBoard event files found")

    return results


def print_health_report(results: dict):
    """Print training health report to stdout."""
    print(f"\n{'=' * 50}")
    print("Training Health Check")
    print(f"{'=' * 50}")
    print(f"Log directory:        {results['log_dir']}")
    print(f"Checkpoints found:    {results['checkpoints_found']}")
    print(f"last.ckpt exists:     {results['last_ckpt_exists']}")
    print(f"TensorBoard events:   {results['tensorboard_events']}")

    if results["issues"]:
        print(f"\nIssues ({len(results['issues'])}):")
        for issue in results["issues"]:
            print(f"  - {issue}")
    else:
        print("\nNo issues found.")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Check training health")
    parser.add_argument("--log-dir", type=str, required=True)
    args = parser.parse_args(argv)

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        print(f"Log directory not found: {log_dir}", file=sys.stderr)
        return 1

    results = check_health(log_dir)
    print_health_report(results)

    return 1 if results["issues"] else 0


if __name__ == "__main__":
    sys.exit(main())
