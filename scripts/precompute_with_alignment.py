"""Pre-compute .pt files with Julius alignment durations in a single pass.

Combines the duration conversion (lab -> duration) and mel precomputation steps
into one pass, eliminating intermediate .npy files.

Usage:
    uv run python scripts/precompute_with_alignment.py \
        --filelist data/jvs/train.txt \
        --lab-dir data/julius_work/wav \
        --output-dir data/jvs_precomputed_aligned/train \
        --mel-mean -6.550095 --mel-std 2.383771 \
        --num-workers 8
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

# Allow importing sibling scripts (convert_julius_to_durations)
_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from convert_julius_to_durations import (
    align_julius_with_pyopenjtalk,
    build_duration_array_with_blanks,
    parse_lab_file,
    time_to_frames,
)

from matcha.text import text_to_sequence
from matcha.text.julius_to_pyopenjtalk import map_julius_sequence
from matcha.utils.audio import mel_spectrogram
from matcha.utils.model import normalize
from matcha.utils.utils import intersperse

# Default mel spectrogram parameters (matching LJSpeech / JVS config)
N_FFT = 1024
N_MELS = 80
SAMPLE_RATE = 22050
HOP_LENGTH = 256
WIN_LENGTH = 1024
F_MIN = 0.0
F_MAX = 8000


def compute_duration_from_lab(
    lab_path: str | Path,
    text: str,
    total_mel_frames: int,
    align_mode: str = "auto",
) -> tuple[np.ndarray | None, str]:
    """Compute a blank-interspersed duration array from a Julius .lab file.

    This is the core computation extracted from
    ``convert_julius_to_durations.process_single_utterance``, but returns the
    numpy array directly instead of writing to disk.

    Args:
        lab_path:          Path to Julius .lab file (HTK format).
        text:              Original Japanese text for this utterance.
        total_mel_frames:  Number of mel frames for this utterance.
        align_mode:        Alignment strategy: ``sequential``, ``dtw``, or
                           ``auto`` (default).

    Returns:
        (duration_array, message) tuple.  duration_array is None on failure.
    """
    # 1. Parse .lab file
    segments = parse_lab_file(lab_path)
    if not segments:
        return None, f"Empty .lab file: {lab_path}"

    # 2. Extract Julius phonemes and compute frame durations
    julius_raw = [seg[2] for seg in segments]
    julius_mapped = map_julius_sequence(julius_raw)
    julius_frame_durations = [time_to_frames(seg[0], seg[1]) for seg in segments]

    # 3. Get pyopenjtalk phoneme sequence
    _seq, clean_text = text_to_sequence(text, ["japanese_cleaners"], language="ja")
    pyopenjtalk_phonemes = clean_text.split()

    # 4. Align Julius with pyopenjtalk
    aligned_durations = align_julius_with_pyopenjtalk(
        julius_mapped,
        pyopenjtalk_phonemes,
        julius_frame_durations,
        align_mode=align_mode,
    )

    # 5. Build blank-interspersed duration array
    duration_array = build_duration_array_with_blanks(aligned_durations, total_mel_frames)

    # 6. Verify length matches interspersed text sequence
    text_seq_interspersed = intersperse(_seq, 0)
    expected_len = len(text_seq_interspersed)
    if len(duration_array) != expected_len:
        return None, (
            f"Length mismatch: duration array {len(duration_array)} "
            f"vs interspersed text {expected_len}"
        )

    return duration_array, f"OK ({len(pyopenjtalk_phonemes)} phones, {total_mel_frames} frames)"


def parse_filelist(filelist_path: str) -> list[list[str]]:
    """Parse a pipe-delimited filelist (wav_path|speaker_id|text)."""
    with open(filelist_path, encoding="utf-8") as f:
        filepaths_and_text = [line.strip().split("|") for line in f if line.strip()]
    return filepaths_and_text


def process_sample_with_alignment(
    wav_path: str,
    spk: int,
    text: str,
    lab_path: str,
    output_path: str,
    mel_mean: float,
    mel_std: float,
    align_mode: str = "auto",
) -> tuple[str, bool, str]:
    """Process a single sample: compute mel, text, and duration from .lab in one pass.

    Args:
        wav_path:    Path to the 22050 Hz wav file.
        spk:         Speaker ID (integer).
        text:        Original Japanese text.
        lab_path:    Path to the Julius .lab alignment file.
        output_path: Path to write the output .pt file.
        mel_mean:    Mean for mel normalization.
        mel_std:     Std for mel normalization.
        align_mode:  Alignment strategy (``auto``, ``sequential``, ``dtw``).

    Returns:
        (output_path, skipped, message) tuple.
    """
    lab_p = Path(lab_path)
    out_p = Path(output_path)

    if not lab_p.exists():
        return str(out_p), True, "no .lab file"

    try:
        # 1. Mel spectrogram
        data, sr = sf.read(wav_path, dtype="float32")
        assert sr == SAMPLE_RATE, f"Expected {SAMPLE_RATE} Hz, got {sr}"
        audio = torch.from_numpy(data).unsqueeze(0)
        mel = mel_spectrogram(
            audio, N_FFT, N_MELS, SAMPLE_RATE, HOP_LENGTH, WIN_LENGTH, F_MIN, F_MAX, center=False
        ).squeeze()
        mel = normalize(mel, mel_mean, mel_std)
        mel_frames = mel.shape[-1]

        # 2. Text sequence
        text_norm, cleaned_text = text_to_sequence(text, ["japanese_cleaners"], language="ja")
        text_norm = intersperse(text_norm, 0)
        text_tensor = torch.IntTensor(text_norm)

        # 3. Duration from .lab (computed in memory, no intermediate .npy)
        duration_array, msg = compute_duration_from_lab(
            lab_path, text, mel_frames, align_mode=align_mode
        )
        if duration_array is None:
            return str(out_p), True, f"duration conversion failed: {msg}"

        duration = torch.from_numpy(duration_array).long()

        # 4. Validate
        if len(duration) != len(text_tensor):
            return str(out_p), True, (
                f"duration/text length mismatch: {len(duration)} vs {len(text_tensor)}"
            )
        if duration.sum().item() != mel_frames:
            return str(out_p), True, (
                f"duration sum mismatch: {duration.sum()} vs {mel_frames}"
            )

        # 5. Save .pt
        torch.save(
            {
                "mel": mel,
                "text": text_tensor,
                "spk": spk,
                "cleaned_text": cleaned_text,
                "durations": duration,
            },
            str(out_p),
        )

        return str(out_p), False, "ok"

    except Exception as e:
        return str(out_p), True, f"Error: {e}"


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute .pt files with Julius durations in a single pass."
    )
    parser.add_argument(
        "--filelist",
        type=str,
        required=True,
        help="Path to filelist (format: wav_path|speaker_id|text)",
    )
    parser.add_argument(
        "--lab-dir",
        type=str,
        required=True,
        help="Directory containing .lab files from Julius segmentation-kit",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write .pt files",
    )
    parser.add_argument(
        "--mel-mean",
        type=float,
        default=-6.550095,
        help="Mean for mel normalization (default: -6.550095)",
    )
    parser.add_argument(
        "--mel-std",
        type=float,
        default=2.383771,
        help="Std for mel normalization (default: 2.383771)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8)",
    )
    parser.add_argument(
        "--align-mode",
        type=str,
        default="auto",
        choices=["sequential", "dtw", "auto"],
        help=(
            "Alignment strategy: 'sequential' (sequential only), "
            "'dtw' (always use DTW), 'auto' (default; short utterances "
            "< 30 phonemes use DTW, longer ones use sequential with "
            "5%% DTW fallback)"
        ),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lab_dir = Path(args.lab_dir)

    entries = parse_filelist(args.filelist)
    print(f"Loaded {len(entries)} entries from {args.filelist}")
    print(f"Lab directory: {lab_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Mel normalization: mean={args.mel_mean}, std={args.mel_std}")
    print(f"Workers: {args.num_workers}, Align mode: {args.align_mode}")

    # Build tasks
    tasks = []
    for entry in entries:
        wav_path, spk_str, text = entry[0], entry[1], entry[2]
        wav_p = Path(wav_path)
        spk_name = wav_p.parent.name
        name = f"{spk_name}_{wav_p.stem}"
        lab_path = lab_dir / f"{name}.lab"
        out_path = output_dir / f"{name}.pt"
        tasks.append((
            wav_path,
            int(spk_str),
            text,
            str(lab_path),
            str(out_path),
            args.mel_mean,
            args.mel_std,
            args.align_mode,
        ))

    # Parallel processing
    success = 0
    skip = 0
    errors = []
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(process_sample_with_alignment, *task): task[4]
            for task in tasks
        }
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Processing",
            unit="samples",
        ):
            out_path = futures[future]
            try:
                _, skipped, msg = future.result()
                if skipped:
                    skip += 1
                    tqdm.write(f"SKIP [{Path(out_path).stem}]: {msg}")
                else:
                    success += 1
            except Exception as e:
                errors.append((out_path, str(e)))
                tqdm.write(f"ERROR [{out_path}]: {e}")

    elapsed = time.time() - start_time
    speed = (success + skip) / elapsed if elapsed > 0 else 0

    print(f"\nDone: {success} saved, {skip} skipped, {len(errors)} errors")
    print(f"Processing speed: {speed:.1f} samples/sec ({elapsed:.1f}s total)")

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for path, msg in errors[:20]:
            print(f"  {Path(path).stem}: {msg}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")


if __name__ == "__main__":
    main()
