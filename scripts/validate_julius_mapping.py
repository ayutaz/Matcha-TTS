#!/usr/bin/env python3
"""JVSの全.labファイルに対するJulius音素マッピングカバレッジ検証スクリプト。

Usage:
    uv run python scripts/validate_julius_mapping.py --lab-dir /path/to/lab_files

全 .lab ファイルを走査し、Julius音素の出現頻度とマッピングカバレッジを報告する。
未マッピング音素があれば警告を出力する。
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from matcha.text.julius_to_pyopenjtalk import (
    JULIUS_TO_PYOPENJTALK,
    get_unmapped_phonemes,
)


def parse_lab_file(lab_path: Path) -> list[str]:
    """Parse a Julius .lab file and extract phoneme labels.

    Julius .lab format (HTK-style):
        <start_time> <end_time> <phoneme>

    Times are in 100ns units (HTK format) or seconds depending on the tool.
    We only extract the phoneme column (3rd field).

    Args:
        lab_path: Path to a .lab file.

    Returns:
        List of phoneme strings found in the file.
    """
    phonemes = []
    with open(lab_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 3:
                phonemes.append(parts[2])
            elif len(parts) == 1:
                # Some formats may have just the phoneme
                phonemes.append(parts[0])
    return phonemes


def main():
    parser = argparse.ArgumentParser(
        description="Validate Julius phoneme mapping coverage against .lab files."
    )
    parser.add_argument(
        "--lab-dir",
        type=Path,
        required=True,
        help="Directory containing .lab files (searched recursively).",
    )
    args = parser.parse_args()

    lab_dir = args.lab_dir
    if not lab_dir.is_dir():
        print(f"Error: {lab_dir} is not a directory.", file=sys.stderr)
        sys.exit(1)

    # Collect all .lab files
    lab_files = sorted(lab_dir.rglob("*.lab"))
    if not lab_files:
        print(f"Warning: No .lab files found in {lab_dir}", file=sys.stderr)
        sys.exit(0)

    print(f"Found {len(lab_files)} .lab files in {lab_dir}")

    # Count phonemes across all files
    phoneme_counter: Counter[str] = Counter()
    files_with_unknown = []

    for lab_path in lab_files:
        phonemes = parse_lab_file(lab_path)
        phoneme_counter.update(phonemes)

        unmapped = get_unmapped_phonemes(phonemes)
        if unmapped:
            files_with_unknown.append((lab_path, unmapped))

    # Report phoneme frequency
    total_phonemes = sum(phoneme_counter.values())
    unique_phonemes = sorted(phoneme_counter.keys())

    print(f"\nTotal phoneme tokens: {total_phonemes}")
    print(f"Unique phoneme types: {len(unique_phonemes)}")

    print("\n--- Phoneme frequency ---")
    for ph, count in phoneme_counter.most_common():
        mapped = ph in JULIUS_TO_PYOPENJTALK
        status = "OK" if mapped else "UNMAPPED"
        mapped_to = f" -> {JULIUS_TO_PYOPENJTALK[ph]}" if mapped else ""
        print(f"  {ph:>8s}: {count:>8d}  [{status}]{mapped_to}")

    # Report unmapped phonemes
    all_phonemes_in_data = set(phoneme_counter.keys())
    all_unmapped = get_unmapped_phonemes(list(all_phonemes_in_data))

    if all_unmapped:
        print(f"\n*** WARNING: {len(all_unmapped)} unmapped phoneme(s) found ***")
        for ph in sorted(all_unmapped):
            count = phoneme_counter[ph]
            print(f"  {ph!r}: {count} occurrences")

        print(f"\nFiles with unmapped phonemes: {len(files_with_unknown)}")
        # Show first 10 files as examples
        for lab_path, unmapped in files_with_unknown[:10]:
            print(f"  {lab_path}: {sorted(unmapped)}")
        if len(files_with_unknown) > 10:
            print(f"  ... and {len(files_with_unknown) - 10} more files")
    else:
        print("\nAll phonemes in .lab files are covered by the mapping.")

    # Summary
    mapped_count = len(all_phonemes_in_data - all_unmapped)
    print(f"\nCoverage: {mapped_count}/{len(all_phonemes_in_data)} phoneme types mapped")

    if all_unmapped:
        sys.exit(1)
    else:
        print("Validation PASSED.")
        sys.exit(0)


if __name__ == "__main__":
    main()
