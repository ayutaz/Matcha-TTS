"""Analyze A/B test results.

Usage:
    uv run python scripts/analyze_ab_test.py \
        --results eval/ab_test/results.json \
        --pairs eval/ab_test/pairs.json \
        --output eval/report/ab_test.json
"""

import argparse
import json
import sys
from pathlib import Path

from scipy.stats import binomtest


def analyze_results(results_data, pairs_data):
    """Analyze A/B test results."""
    # Count preferences
    counts = {"A": 0, "B": 0, "equal": 0}
    julius_preferred = 0
    mas_preferred = 0
    total = len(results_data)

    for result, pair in zip(results_data, pairs_data):
        choice = result.get("choice", "equal")
        counts[choice] = counts.get(choice, 0) + 1

        if choice == "A" and pair.get("A_is") == "julius" or choice == "B" and pair.get("B_is") == "julius":
            julius_preferred += 1
        elif choice == "A" and pair.get("A_is") == "mas" or choice == "B" and pair.get("B_is") == "mas":
            mas_preferred += 1

    decisive = julius_preferred + mas_preferred
    julius_rate = julius_preferred / decisive if decisive > 0 else 0.5

    # Binomial test
    if decisive > 0:
        bt = binomtest(julius_preferred, decisive, 0.5)
        p_value = float(bt.pvalue)
    else:
        p_value = 1.0

    return {
        "total_pairs": total,
        "julius_preferred": julius_preferred,
        "mas_preferred": mas_preferred,
        "equal": counts.get("equal", 0),
        "julius_preference_rate": julius_rate,
        "p_value": p_value,
        "significant": p_value < 0.05,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Analyze A/B test results")
    parser.add_argument("--results", type=str, required=True)
    parser.add_argument("--pairs", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args(argv)

    for p in [args.results, args.pairs]:
        if not Path(p).exists():
            print(f"File not found: {p}", file=sys.stderr)
            return 1

    results = json.loads(Path(args.results).read_text())
    pairs = json.loads(Path(args.pairs).read_text())

    report = analyze_results(results, pairs)

    print("\n=== A/B Test Results ===")
    print(f"Julius preferred: {report['julius_preferred']}/{report['total_pairs']}")
    print(f"MAS preferred:    {report['mas_preferred']}/{report['total_pairs']}")
    print(f"Equal:            {report['equal']}/{report['total_pairs']}")
    print(f"Julius rate:      {report['julius_preference_rate']:.1%}")
    print(f"p-value:          {report['p_value']:.4f}")
    print(f"Significant:      {report['significant']}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
