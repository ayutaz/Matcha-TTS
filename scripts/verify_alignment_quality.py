"""Verify alignment quality of Julius-generated duration arrays.

Reads .npy duration files, cross-checks against filelist text lengths,
and produces a JSON report plus a human-readable console summary.

Usage:
    uv run python scripts/verify_alignment_quality.py \
        --duration-dir data/jvs_durations \
        --filelist data/jvs/train.txt \
        --output-report data/alignment_quality_report.json

The script uses ``matcha.utils.alignment_metrics`` (shared with M5) for
all metric calculations.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from matcha.utils.alignment_metrics import (
    compute_corpus_stats,
    compute_duration_stats,
    compute_phoneme_class_stats,
)

logger = logging.getLogger(__name__)

# ---- Japanese phoneme class definitions ----
_VOWELS = {"a", "i", "u", "e", "o", "A", "I", "U", "E", "O"}
_HATSUON = {"N"}
_SOKUON = {"cl"}
_PAUSE = {"pau", "_"}
_SILENCE = {"sil", "^", "$", "?"}  # ? = interrogative-final sil (carries duration)
_PROSODY = {"#", "[", "]"}


def classify_phoneme(sym: str) -> str:
    """Return a coarse class name for a Japanese phoneme symbol."""
    if sym in _VOWELS:
        return "vowel"
    if sym in _HATSUON:
        return "hatsuon"
    if sym in _SOKUON:
        return "sokuon"
    if sym in _PAUSE:
        return "pause"
    if sym in _SILENCE:
        return "silence"
    if sym in _PROSODY:
        return "prosody"
    return "consonant"


# ---------------------------------------------------------------------------
# Filelist parsing (same format as convert_julius_to_durations.py)
# ---------------------------------------------------------------------------


def parse_filelist(filelist_path: str) -> list[tuple[str, str, str]]:
    """Parse a pipe-delimited filelist (wav_path|speaker_id|text).

    Returns:
        List of (wav_path, speaker_id, text) tuples.
    """
    entries = []
    with open(filelist_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) != 3:
                continue
            entries.append((parts[0], parts[1], parts[2]))
    return entries


def make_output_name(wav_path: str) -> str:
    """Generate output filename stem from wav path: {spk}_{utt_id}."""
    wav_p = Path(wav_path)
    spk_name = wav_p.parent.name
    utt_id = wav_p.stem
    return f"{spk_name}_{utt_id}"


# ---------------------------------------------------------------------------
# Core verification logic
# ---------------------------------------------------------------------------


def load_durations(duration_dir: Path, names: list[str]) -> dict[str, np.ndarray]:
    """Load all .npy duration files for the given names.

    Returns:
        ``{name: durations_array}`` for files that exist.
    """
    loaded = {}
    for name in names:
        npy_path = duration_dir / f"{name}.npy"
        if npy_path.exists():
            loaded[name] = np.load(str(npy_path))
    return loaded


def check_length_consistency(
    durations_dict: dict[str, np.ndarray],
    entries: list[tuple[str, str, str]],
) -> list[dict]:
    """Verify that each duration array length matches the interspersed text length.

    Returns:
        List of problem dicts ``{name, expected_len, actual_len}`` for
        mismatches.
    """
    from matcha.text import text_to_sequence
    from matcha.utils.utils import intersperse

    problems = []
    for wav_path, _spk, text in entries:
        name = make_output_name(wav_path)
        if name not in durations_dict:
            continue
        dur = durations_dict[name]

        seq, _ = text_to_sequence(text, ["japanese_cleaners"], language="ja")
        interspersed = intersperse(seq, 0)
        expected_len = len(interspersed)

        if len(dur) != expected_len:
            problems.append(
                {
                    "name": name,
                    "expected_len": expected_len,
                    "actual_len": len(dur),
                }
            )
    return problems


def build_phoneme_class_report(
    durations_dict: dict[str, np.ndarray],
    entries: list[tuple[str, str, str]],
) -> dict[str, dict]:
    """Aggregate phoneme-class statistics across the corpus.

    Returns:
        ``{class_name: {count, mean, median, pct_le1, pct_le2}}``
    """
    from matcha.text import text_to_sequence
    from matcha.text.symbols import symbols_ja

    id_to_sym = {i: s for i, s in enumerate(symbols_ja)}

    # Collect per-class durations
    class_durs: dict[str, list[int]] = {}

    for wav_path, _spk, text in entries:
        name = make_output_name(wav_path)
        if name not in durations_dict:
            continue
        dur = durations_dict[name]

        seq, _ = text_to_sequence(text, ["japanese_cleaners"], language="ja")
        per_sym = compute_phoneme_class_stats(dur, seq, id_to_sym)
        for sym, dvals in per_sym.items():
            cls = classify_phoneme(sym)
            class_durs.setdefault(cls, []).extend(dvals)

    report = {}
    for cls, dvals in sorted(class_durs.items()):
        arr = np.array(dvals, dtype=np.float64)
        n = len(arr)
        report[cls] = {
            "count": n,
            "mean": round(float(np.mean(arr)), 2) if n > 0 else 0.0,
            "median": round(float(np.median(arr)), 2) if n > 0 else 0.0,
            "pct_le1": round(float(np.sum(arr <= 1) / n), 4) if n > 0 else 0.0,
            "pct_le2": round(float(np.sum(arr <= 2) / n), 4) if n > 0 else 0.0,
        }
    return report


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_report(
    duration_dir: Path,
    filelist_path: str,
) -> dict:
    """Run all checks and return the full report as a dict.

    Args:
        duration_dir: Directory containing ``.npy`` duration files.
        filelist_path: Path to pipe-delimited filelist.

    Returns:
        Report dictionary suitable for JSON serialization.
    """
    entries = parse_filelist(filelist_path)
    names = [make_output_name(e[0]) for e in entries]

    # Load durations
    dur_dict = load_durations(duration_dir, names)

    # Missing files
    missing = [n for n in names if n not in dur_dict]

    # Corpus-level stats
    all_durs = list(dur_dict.values())
    corpus_stats = compute_corpus_stats(all_durs)

    # Length consistency
    length_problems = check_length_consistency(dur_dict, entries)

    # Per-utterance degenerate list
    degenerate_names = []
    for wav_path, _spk, _text in entries:
        name = make_output_name(wav_path)
        if name in dur_dict:
            stats = compute_duration_stats(dur_dict[name])
            if stats["is_degenerate"]:
                degenerate_names.append(name)

    # Phoneme class stats
    class_report = build_phoneme_class_report(dur_dict, entries)

    report = {
        "total_in_filelist": len(entries),
        "total_loaded": len(dur_dict),
        "missing_files": len(missing),
        "missing_file_list": missing[:50],  # truncate for readability
        "corpus_stats": {
            "total_samples": corpus_stats["total_samples"],
            "degenerate_count": corpus_stats["degenerate_count"],
            "degenerate_rate": round(corpus_stats["degenerate_rate"], 6),
            "phoneme_duration": {
                "mean": round(corpus_stats["phoneme_duration_stats"]["mean"], 2),
                "median": round(corpus_stats["phoneme_duration_stats"]["median"], 2),
                "std": round(corpus_stats["phoneme_duration_stats"]["std"], 2),
                "pct_le1": round(corpus_stats["phoneme_duration_stats"]["pct_le1"], 4),
                "pct_le2": round(corpus_stats["phoneme_duration_stats"]["pct_le2"], 4),
            },
            "blank_stats": {
                "mean_blank0": round(corpus_stats["blank_stats"]["mean_blank0"], 2),
                "all_zero_rate": round(corpus_stats["blank_stats"]["all_zero_rate"], 4),
            },
        },
        "length_mismatches": length_problems,
        "degenerate_samples": degenerate_names,
        "phoneme_class_stats": class_report,
    }
    return report


def print_summary(report: dict) -> None:
    """Print a human-readable summary to stdout."""
    cs = report["corpus_stats"]
    ph = cs["phoneme_duration"]
    bl = cs["blank_stats"]

    print("=== Alignment Quality Report ===")
    print(f"Total in filelist: {report['total_in_filelist']}")
    print(f"Total loaded: {report['total_loaded']}")
    if report["missing_files"] > 0:
        print(f"Missing duration files: {report['missing_files']}")
    print(f"Degenerate samples: {cs['degenerate_count']} ({cs['degenerate_rate'] * 100:.2f}%)")
    print(f"Phoneme duration: mean={ph['mean']}, median={ph['median']}, std={ph['std']}")
    print(f"  <= 1 frame: {ph['pct_le1'] * 100:.1f}%")
    print(f"  <= 2 frames: {ph['pct_le2'] * 100:.1f}%")
    print(f"Blank[0] mean duration: {bl['mean_blank0']} (all zero: {bl['all_zero_rate'] * 100:.1f}%)")

    if report["length_mismatches"]:
        print(f"\nLength mismatches: {len(report['length_mismatches'])}")
        for p in report["length_mismatches"][:10]:
            print(f"  {p['name']}: expected {p['expected_len']}, got {p['actual_len']}")

    if report["degenerate_samples"]:
        print(f"\nDegenerate samples ({len(report['degenerate_samples'])}):")
        for name in report["degenerate_samples"][:20]:
            print(f"  {name}")
        if len(report["degenerate_samples"]) > 20:
            print(f"  ... and {len(report['degenerate_samples']) - 20} more")

    if report["phoneme_class_stats"]:
        print("\nPhoneme class statistics:")
        for cls, st in report["phoneme_class_stats"].items():
            print(
                f"  {cls:12s}: count={st['count']:6d}, "
                f"mean={st['mean']:5.1f}, median={st['median']:5.1f}, "
                f"<=1f={st['pct_le1'] * 100:5.1f}%, <=2f={st['pct_le2'] * 100:5.1f}%"
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Entry point for CLI invocation.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code: 0 on success, 1 on error.
    """
    parser = argparse.ArgumentParser(description="Verify alignment quality of Julius-generated duration arrays.")
    parser.add_argument(
        "--duration-dir",
        type=str,
        required=True,
        help="Directory containing .npy duration files",
    )
    parser.add_argument(
        "--filelist",
        type=str,
        required=True,
        help="Path to filelist (format: wav_path|speaker_id|text)",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default=None,
        help="Path to write JSON report (optional)",
    )
    args = parser.parse_args(argv)

    duration_dir = Path(args.duration_dir)
    if not duration_dir.is_dir():
        print(f"Error: duration directory not found: {duration_dir}", file=sys.stderr)
        return 1

    report = generate_report(duration_dir, args.filelist)
    print_summary(report)

    if args.output_report:
        out_path = Path(args.output_report)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nJSON report written to: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
