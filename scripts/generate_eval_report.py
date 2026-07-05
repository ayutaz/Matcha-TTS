"""Generate integrated evaluation report from individual metric JSONs.

Usage:
    uv run python scripts/generate_eval_report.py \
        --report-dir eval/report \
        --output eval/report/eval_report.json
"""

import argparse
import json
import sys
from pathlib import Path


def load_json_safe(path):
    """Load JSON file, return None if not found."""
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def generate_report(report_dir):
    """Collect all metric JSONs into a single report."""
    report_dir = Path(report_dir)

    report = {
        "duration_accuracy": load_json_safe(report_dir / "duration_accuracy.json"),
        "degeneration": load_json_safe(report_dir / "degeneration.json"),
        "mcd": load_json_safe(report_dir / "mcd.json"),
        "utmos": load_json_safe(report_dir / "utmos.json"),
    }

    # Summary
    deg = report.get("degeneration") or {}
    summary = {
        "degeneration_rate": deg.get("degenerate_rate", "N/A"),
        "phoneme_median_duration": deg.get("phoneme_median_duration", "N/A"),
        "status": "complete" if any(v is not None for v in report.values()) else "no_data",
    }
    report["summary"] = summary

    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate integrated evaluation report from individual metric JSONs")
    parser.add_argument("--report-dir", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args(argv)

    report_dir = Path(args.report_dir)
    if not report_dir.exists():
        print(f"Report directory not found: {report_dir}", file=sys.stderr)
        return 1

    report = generate_report(report_dir)

    output_path = Path(args.output) if args.output else report_dir / "eval_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Report saved to {output_path}")

    # Print summary
    s = report.get("summary", {})
    print("\n=== Evaluation Summary ===")
    print(f"Degeneration rate: {s.get('degeneration_rate', 'N/A')}")
    print(f"Phoneme median dur: {s.get('phoneme_median_duration', 'N/A')}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
