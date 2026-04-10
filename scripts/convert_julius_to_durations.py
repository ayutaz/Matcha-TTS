"""Convert Julius .lab alignment files to Matcha-TTS duration arrays.

Reads HTK-format .lab files produced by Julius segmentation-kit, aligns the
phoneme sequence with pyopenjtalk's output (including prosody symbols and
devoiced vowels), and writes per-utterance duration arrays as .npy files.

The output arrays have length 2*N+1 (blank-interspersed) and dtype int64,
suitable for use with ``model.use_precomputed_durations=True``.

Usage:
    uv run python scripts/convert_julius_to_durations.py \
        --lab-dir data/julius_alignment \
        --filelist data/jvs/train.txt \
        --output-dir data/jvs_durations \
        --mel-dir /dev/shm/jvs_precomputed/train \
        --num-workers 8
"""

from __future__ import annotations

import argparse
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

from matcha.text.julius_to_pyopenjtalk import (
    JULIUS_TO_PYOPENJTALK,
    PROSODY_SYMBOLS,
)

logger = logging.getLogger(__name__)

# Audio parameters matching Matcha-TTS defaults
SAMPLE_RATE = 22050
HOP_LENGTH = 256

# Devoiced vowels in pyopenjtalk map to their lowercase counterpart in Julius
_DEVOICED_TO_VOICED = {"A": "a", "I": "i", "U": "u", "E": "e", "O": "o"}


# ---------------------------------------------------------------------------
# .lab file parsing
# ---------------------------------------------------------------------------


def parse_lab_file(lab_path: str | Path) -> list[tuple[float, float, str]]:
    """Parse an HTK-format .lab file into (start_sec, end_sec, phoneme) tuples.

    HTK format: ``start_100ns end_100ns phoneme``
    where times are in 100-nanosecond units (1e-7 seconds).

    Args:
        lab_path: Path to the .lab file.

    Returns:
        List of (start_sec, end_sec, phoneme) tuples. Empty list for empty files.
    """
    result: list[tuple[float, float, str]] = []
    with open(lab_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            start_100ns = int(parts[0])
            end_100ns = int(parts[1])
            phoneme = parts[2]
            start_sec = start_100ns / 10_000_000
            end_sec = end_100ns / 10_000_000
            result.append((start_sec, end_sec, phoneme))
    return result


# ---------------------------------------------------------------------------
# Time-to-frame conversion (absolute timestamp based)
# ---------------------------------------------------------------------------


def time_to_frames(start_sec: float, end_sec: float) -> int:
    """Convert a time interval to the number of mel-spectrogram frames.

    Uses absolute timestamp rounding to avoid cumulative rounding errors:
      start_frame = round(start_sec * SAMPLE_RATE / HOP_LENGTH)
      end_frame   = round(end_sec   * SAMPLE_RATE / HOP_LENGTH)
      frames = max(0, end_frame - start_frame)

    Args:
        start_sec: Segment start time in seconds.
        end_sec:   Segment end time in seconds.

    Returns:
        Number of frames (non-negative integer).
    """
    start_frame = round(start_sec * SAMPLE_RATE / HOP_LENGTH)
    end_frame = round(end_sec * SAMPLE_RATE / HOP_LENGTH)
    return max(0, end_frame - start_frame)


# ---------------------------------------------------------------------------
# Phoneme alignment: pyopenjtalk <-> Julius
# ---------------------------------------------------------------------------


def _pyopenjtalk_to_julius_key(ph: str) -> str | None:
    """Map a pyopenjtalk phoneme to the corresponding Julius lookup key.

    - Prosody symbols with no Julius counterpart (``#``, ``[``, ``]``, ``?``)
      return None.
    - ``^`` and ``$`` are mapped by the caller (first/last sil).
    - ``_`` corresponds to ``pau`` in Julius.
    - Devoiced vowels (A,I,U,E,O) map to their lowercase form.
    - All other phonemes are looked up directly in JULIUS_TO_PYOPENJTALK values.

    Returns:
        The Julius phoneme string to match, or None if this pyopenjtalk symbol
        has no Julius counterpart and should receive duration=0.
    """
    if ph in {"#", "[", "]", "?"}:
        return None
    if ph in _DEVOICED_TO_VOICED:
        return _DEVOICED_TO_VOICED[ph]
    return ph


def align_julius_with_pyopenjtalk(
    julius_phonemes: list[str],
    pyopenjtalk_phonemes: list[str],
    julius_durations: list[int],
) -> list[int]:
    """Align Julius phonemes with pyopenjtalk phonemes and assign durations.

    The alignment follows these rules:
      1. ``^`` (sentence start) gets the duration of the first Julius ``sil``
         (silB).
      2. ``$`` (sentence end) gets the duration of the last Julius ``sil``
         (silE).
      3. ``_`` (pause) gets the duration of the next unmatched Julius ``pau``.
      4. ``#``, ``[``, ``]``, ``?`` get duration=0 (prosody-only markers).
      5. Devoiced vowels (A,I,U,E,O) are matched against Julius lowercase
         vowels (a,i,u,e,o). If no matching Julius phoneme remains at the
         current position, duration=0 is assigned.
      6. Regular phonemes are matched sequentially.

    If the mismatch rate exceeds 5%, falls back to DTW alignment.

    Args:
        julius_phonemes:   Phoneme labels from the .lab file, already mapped
                           via ``map_julius_sequence`` (i.e. pyopenjtalk-compatible).
        pyopenjtalk_phonemes: Phoneme list from ``japanese_cleaners`` output
                              (split on spaces).
        julius_durations:  Frame durations for each Julius phoneme segment.

    Returns:
        List of integer durations, one per pyopenjtalk phoneme.

    Raises:
        ValueError: If Julius and pyopenjtalk sequences cannot be aligned.
    """
    if not pyopenjtalk_phonemes:
        return []

    n_julius = len(julius_phonemes)
    n_pyopenjtalk = len(pyopenjtalk_phonemes)
    result = [0] * n_pyopenjtalk

    j_idx = 0  # current position in Julius sequence
    mismatches = 0

    for p_idx, ph in enumerate(pyopenjtalk_phonemes):
        # Rule 1: sentence start
        if ph == "^":
            # Find first sil in Julius (should be at index 0)
            for k in range(n_julius):
                if julius_phonemes[k] == "sil" and k == 0:
                    result[p_idx] = julius_durations[k]
                    # Only advance j_idx if it is still at or before k
                    if j_idx <= k:
                        j_idx = k + 1
                    break
            continue

        # Rule 2: sentence end
        if ph == "$":
            # Find last sil in Julius (should be the final entry)
            for k in range(n_julius - 1, -1, -1):
                if julius_phonemes[k] == "sil":
                    result[p_idx] = julius_durations[k]
                    break
            continue

        # Rule 3: pause
        if ph == "_":
            # Find next pau in Julius from current position
            for k in range(j_idx, n_julius):
                if julius_phonemes[k] == "pau":
                    result[p_idx] = julius_durations[k]
                    j_idx = k + 1
                    break
            continue

        # Rule 4: prosody-only markers
        if ph in {"#", "[", "]", "?"}:
            result[p_idx] = 0
            continue

        # Rule 5 & 6: regular phonemes and devoiced vowels
        # Determine what to look for in Julius
        julius_target = _pyopenjtalk_to_julius_key(ph)
        if julius_target is None:
            result[p_idx] = 0
            continue

        # Map devoiced vowel targets through the Julius mapping
        # pyopenjtalk "A" -> julius_target "a" -> julius may have "a"
        # pyopenjtalk "sil" -> julius may have "sil"
        # We need to find julius_target in the mapped julius sequence

        matched = False
        if j_idx < n_julius and julius_phonemes[j_idx] == julius_target:
            # Direct match at current position
            result[p_idx] = julius_durations[j_idx]
            j_idx += 1
            matched = True
        elif ph in _DEVOICED_TO_VOICED:
            # Devoiced vowel: Julius may have dropped it entirely
            # Search a small window ahead in case of minor misalignment
            found = False
            for k in range(j_idx, min(j_idx + 3, n_julius)):
                if julius_phonemes[k] == julius_target:
                    result[p_idx] = julius_durations[k]
                    j_idx = k + 1
                    found = True
                    matched = True
                    break
            if not found:
                # Devoiced vowel missing in Julius -> duration=0
                result[p_idx] = 0
                matched = True
        else:
            # Look ahead a small window for the expected phoneme
            found = False
            for k in range(j_idx, min(j_idx + 3, n_julius)):
                if julius_phonemes[k] == julius_target:
                    result[p_idx] = julius_durations[k]
                    j_idx = k + 1
                    found = True
                    matched = True
                    break
            if not found:
                mismatches += 1
                result[p_idx] = 0
                matched = True

    # Check mismatch rate and fall back to DTW if needed
    # Count only non-prosody phonemes for mismatch rate
    n_real_phonemes = sum(
        1 for ph in pyopenjtalk_phonemes if ph not in PROSODY_SYMBOLS and ph not in {"^", "$"}
    )
    if n_real_phonemes > 0 and mismatches / n_real_phonemes > 0.05:
        logger.warning(
            "High mismatch rate (%.1f%%), falling back to DTW alignment",
            100 * mismatches / n_real_phonemes,
        )
        return align_julius_with_pyopenjtalk_dtw(
            julius_phonemes, pyopenjtalk_phonemes, julius_durations
        )

    return result


def _phoneme_distance(ph_julius: str, ph_pyopenjtalk: str) -> float:
    """Compute distance between a Julius phoneme and a pyopenjtalk phoneme.

    Returns 0 for exact match, 0.5 for devoiced vowel match, 1.0 otherwise.
    """
    if ph_julius == ph_pyopenjtalk:
        return 0.0
    # Devoiced vowel: A->a, I->i, etc.
    if ph_pyopenjtalk in _DEVOICED_TO_VOICED and ph_julius == _DEVOICED_TO_VOICED[ph_pyopenjtalk]:
        return 0.5
    # sil matches ^ or $
    if ph_julius == "sil" and ph_pyopenjtalk in {"^", "$"}:
        return 0.0
    # pau matches _
    if ph_julius == "pau" and ph_pyopenjtalk == "_":
        return 0.0
    return 1.0


def align_julius_with_pyopenjtalk_dtw(
    julius_phonemes: list[str],
    pyopenjtalk_phonemes: list[str],
    julius_durations: list[int],
) -> list[int]:
    """DTW-based fallback alignment between Julius and pyopenjtalk phonemes.

    Uses dynamic time warping to find the optimal alignment between the two
    sequences, skipping prosody-only symbols (#, [, ], ?) which receive
    duration=0.

    Args:
        julius_phonemes:     Mapped Julius phonemes (pyopenjtalk-compatible).
        pyopenjtalk_phonemes: Phoneme list from japanese_cleaners.
        julius_durations:    Frame durations per Julius phoneme.

    Returns:
        List of durations aligned to the pyopenjtalk sequence.
    """
    n_pyopenjtalk = len(pyopenjtalk_phonemes)
    if not pyopenjtalk_phonemes:
        return []

    result = [0] * n_pyopenjtalk

    # Separate prosody-only symbols -- they always get duration=0
    # Build a list of (original_index, phoneme) for non-prosody pyopenjtalk phones
    non_prosody_indices = []
    non_prosody_phones = []
    for i, ph in enumerate(pyopenjtalk_phonemes):
        if ph not in {"#", "[", "]", "?"}:
            non_prosody_indices.append(i)
            non_prosody_phones.append(ph)

    n_np = len(non_prosody_phones)
    n_j = len(julius_phonemes)

    if n_np == 0 or n_j == 0:
        return result

    # Build DTW cost matrix
    INF = float("inf")
    # cost[i][j] = min cost to align non_prosody_phones[:i] with julius_phonemes[:j]
    cost = [[INF] * (n_j + 1) for _ in range(n_np + 1)]
    cost[0][0] = 0.0

    for i in range(1, n_np + 1):
        for j in range(1, n_j + 1):
            d = _phoneme_distance(julius_phonemes[j - 1], non_prosody_phones[i - 1])
            # Standard DTW transitions: match, insert (skip julius), delete (skip pyopenjtalk)
            cost[i][j] = d + min(
                cost[i - 1][j - 1],  # match
                cost[i - 1][j],      # skip julius (pyopenjtalk phone gets no duration)
                cost[i][j - 1],      # skip pyopenjtalk (julius phone unmatched)
            )

    # Backtrace
    i, j = n_np, n_j
    while i > 0 and j > 0:
        d = _phoneme_distance(julius_phonemes[j - 1], non_prosody_phones[i - 1])
        c_match = cost[i - 1][j - 1]
        c_skip_j = cost[i - 1][j]
        c_skip_p = cost[i][j - 1]

        best = min(c_match, c_skip_j, c_skip_p)
        if best == c_match:
            # Match: assign Julius duration to pyopenjtalk phone
            orig_idx = non_prosody_indices[i - 1]
            result[orig_idx] = julius_durations[j - 1]
            i -= 1
            j -= 1
        elif best == c_skip_j:
            # pyopenjtalk phone had no Julius match -> duration=0
            i -= 1
        else:
            # Julius phone unmatched (skip it)
            j -= 1

    # Remaining pyopenjtalk phones get duration=0 (already initialized)
    return result


# ---------------------------------------------------------------------------
# Duration array with blank intersperse
# ---------------------------------------------------------------------------


def build_duration_array_with_blanks(
    phoneme_durations: list[int],
    total_mel_frames: int,
) -> np.ndarray:
    """Build a blank-interspersed duration array matching Matcha-TTS conventions.

    Given N phoneme durations [d_1, d_2, ..., d_N], produces an array of
    length 2*N+1:
        [0, d_1, 0, d_2, ..., 0, d_N, 0]
    where blank positions (even indices) have duration=0.

    The sum of all durations is adjusted to match ``total_mel_frames`` by
    modifying the last non-zero phoneme duration.

    Args:
        phoneme_durations: List of N integer durations (one per phoneme).
        total_mel_frames:  Target total number of mel-spectrogram frames.

    Returns:
        np.ndarray of shape (2*N+1,) with dtype int64.
    """
    n = len(phoneme_durations)
    length = 2 * n + 1
    arr = np.zeros(length, dtype=np.int64)

    # Place phoneme durations at odd indices
    for i, dur in enumerate(phoneme_durations):
        arr[2 * i + 1] = dur

    # Adjust to match total_mel_frames
    current_sum = int(arr.sum())
    diff = total_mel_frames - current_sum
    if diff != 0:
        # Find the last non-zero entry (phoneme position) for adjustment
        last_nonzero = -1
        for i in range(length - 1, -1, -1):
            if arr[i] > 0:
                last_nonzero = i
                break

        if last_nonzero >= 0:
            arr[last_nonzero] = max(0, arr[last_nonzero] + diff)
        elif diff > 0 and n > 0:
            # All durations are zero -- put the difference in the last phoneme slot
            # This is a degenerate case but we handle it gracefully
            arr[2 * n - 1] = diff

    return arr


# ---------------------------------------------------------------------------
# Single-utterance processing pipeline
# ---------------------------------------------------------------------------


def process_single_utterance(
    lab_path: str | Path,
    text: str,
    total_mel_frames: int,
    output_path: str | Path,
) -> tuple[bool, str]:
    """Process one utterance: .lab file -> .npy duration array.

    Args:
        lab_path:          Path to Julius .lab file (HTK format).
        text:              Original Japanese text for this utterance.
        total_mel_frames:  Number of mel frames for this utterance.
        output_path:       Path to write the output .npy file.

    Returns:
        (success, message) tuple.
    """
    try:
        import torch

        from matcha.text import text_to_sequence
        from matcha.text.julius_to_pyopenjtalk import map_julius_sequence

        # 1. Parse .lab file
        segments = parse_lab_file(lab_path)
        if not segments:
            return False, f"Empty .lab file: {lab_path}"

        # 2. Extract Julius phonemes and compute frame durations
        julius_raw = [seg[2] for seg in segments]
        julius_mapped = map_julius_sequence(julius_raw)
        julius_frame_durations = [
            time_to_frames(seg[0], seg[1]) for seg in segments
        ]

        # 3. Get pyopenjtalk phoneme sequence
        _seq, clean_text = text_to_sequence(
            text, ["japanese_cleaners"], language="ja"
        )
        pyopenjtalk_phonemes = clean_text.split()

        # 4. Align Julius with pyopenjtalk
        aligned_durations = align_julius_with_pyopenjtalk(
            julius_mapped, pyopenjtalk_phonemes, julius_frame_durations
        )

        # 5. Build blank-interspersed duration array
        duration_array = build_duration_array_with_blanks(
            aligned_durations, total_mel_frames
        )

        # 6. Verify length matches interspersed text sequence
        from matcha.utils.utils import intersperse

        text_seq_interspersed = intersperse(_seq, 0)
        expected_len = len(text_seq_interspersed)
        if len(duration_array) != expected_len:
            return False, (
                f"Length mismatch: duration array {len(duration_array)} "
                f"vs interspersed text {expected_len}"
            )

        # 7. Save
        np.save(str(output_path), duration_array)
        return True, f"OK ({len(pyopenjtalk_phonemes)} phones, {total_mel_frames} frames)"

    except Exception as e:
        return False, f"Error: {e}"


# ---------------------------------------------------------------------------
# CLI: batch processing
# ---------------------------------------------------------------------------


def parse_filelist(filelist_path: str) -> list[tuple[str, str, str]]:
    """Parse a pipe-delimited filelist (wav_path|speaker_id|text)."""
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
    """Generate output filename from wav path: {spk}_{utt_id}."""
    wav_p = Path(wav_path)
    spk_name = wav_p.parent.name
    utt_id = wav_p.stem
    return f"{spk_name}_{utt_id}"


def _worker(args_tuple):
    """Worker function for parallel processing."""
    lab_path, text, total_mel_frames, output_path = args_tuple
    return process_single_utterance(lab_path, text, total_mel_frames, output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Convert Julius .lab alignment files to Matcha-TTS duration arrays."
    )
    parser.add_argument(
        "--lab-dir",
        type=str,
        required=True,
        help="Directory containing .lab files from Julius segmentation-kit",
    )
    parser.add_argument(
        "--filelist",
        type=str,
        required=True,
        help="Path to filelist (format: wav_path|speaker_id|text)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for .npy duration files",
    )
    parser.add_argument(
        "--mel-dir",
        type=str,
        required=True,
        help="Directory containing precomputed .pt files (for mel frame counts)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8)",
    )
    args = parser.parse_args()

    import torch

    lab_dir = Path(args.lab_dir)
    output_dir = Path(args.output_dir)
    mel_dir = Path(args.mel_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    entries = parse_filelist(args.filelist)
    print(f"Loaded {len(entries)} entries from {args.filelist}")

    # Build tasks
    tasks = []
    skipped_no_lab = 0
    skipped_no_mel = 0

    for wav_path, spk_id, text in entries:
        name = make_output_name(wav_path)
        lab_path = lab_dir / f"{name}.lab"
        mel_path = mel_dir / f"{name}.pt"
        out_path = output_dir / f"{name}.npy"

        if not lab_path.exists():
            skipped_no_lab += 1
            continue
        if not mel_path.exists():
            skipped_no_mel += 1
            continue

        # Load mel frame count from .pt file
        data = torch.load(str(mel_path), map_location="cpu", weights_only=True)
        total_mel_frames = data["mel"].shape[-1]

        tasks.append((str(lab_path), text, total_mel_frames, str(out_path)))

    print(f"Tasks: {len(tasks)}")
    if skipped_no_lab:
        print(f"Skipped (no .lab): {skipped_no_lab}")
    if skipped_no_mel:
        print(f"Skipped (no .pt):  {skipped_no_mel}")

    # Process
    errors = []
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(_worker, task): task[0] for task in tasks}
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Converting durations",
            unit="files",
        ):
            lab_path = futures[future]
            try:
                success, msg = future.result()
                if not success:
                    errors.append((lab_path, msg))
                    tqdm.write(f"FAIL [{Path(lab_path).stem}]: {msg}")
            except Exception as e:
                errors.append((lab_path, str(e)))
                tqdm.write(f"ERROR [{lab_path}]: {e}")

    elapsed = time.time() - start_time
    success_count = len(tasks) - len(errors)
    speed = success_count / elapsed if elapsed > 0 else 0

    print("\nResults:")
    print(f"  Success: {success_count}")
    print(f"  Errors:  {len(errors)}")
    print(f"  Speed:   {speed:.1f} files/sec ({elapsed:.1f}s total)")

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for path, msg in errors[:20]:
            print(f"  {Path(path).stem}: {msg}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")


if __name__ == "__main__":
    main()
