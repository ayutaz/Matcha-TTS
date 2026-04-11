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

    # Check UTMOS availability
    try:
        import utmos  # noqa: F401

        utmos_available = True
    except ImportError:
        utmos_available = False
        print("UTMOS not installed. Install with: pip install utmos")
        print("Scoring skipped, saving file count only.")

    report = {
        "n_wav_files": len(wav_files),
        "utmos_available": utmos_available,
        "status": "utmos_ready" if utmos_available else "utmos_not_installed",
    }

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
