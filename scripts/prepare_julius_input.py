"""Prepare JVS data for Julius segmentation-kit forced alignment.

Reads Matcha-TTS filelists (wav_path|spk_id|text), resamples audio to 16kHz
16-bit mono WAV, converts text to katakana via pyopenjtalk, and writes files
into the directory structure expected by segmentation-kit.

Usage:
    uv run python scripts/prepare_julius_input.py \
        --filelist data/jvs/train.txt data/jvs/val.txt \
        --output-dir data/julius_input \
        --num-workers 8

Output structure:
    data/julius_input/
        wav/
            jvs001_BASIC5000_0025.wav   (16kHz 16bit mono)
            ...
        txt/
            jvs001_BASIC5000_0025.txt   (katakana text)
            ...
"""

import argparse
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from tqdm import tqdm

# Julius segmentation-kit expects 16kHz audio
JULIUS_SAMPLE_RATE = 16000

# Punctuation / symbols to strip from katakana output.
# pyopenjtalk.g2p(kana=True) may include Japanese punctuation.
_PUNCT_RE = re.compile(
    r"[。、！？!?,.\-\s「」『』（）\(\)【】\[\]｛｝\{\}・…―─　]"
)


def parse_filelist(filelist_path):
    """Parse a pipe-delimited filelist (wav_path|speaker_id|text).

    Returns:
        list of (wav_path, spk_id_str, text) tuples.
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


def text_to_katakana(text):
    """Convert Japanese text (kanji/kana mix) to katakana via pyopenjtalk.

    Strips punctuation and whitespace from the result so that only
    katakana characters remain (what Julius segmentation-kit expects).

    Returns:
        Katakana string with punctuation removed.
    """
    import pyopenjtalk

    kana = pyopenjtalk.g2p(text, kana=True)
    # Remove punctuation and whitespace
    kana = _PUNCT_RE.sub("", kana)
    return kana


def resample_to_16k(input_path, output_path):
    """Resample a WAV file to 16kHz 16-bit mono.

    Uses soundfile for I/O and scipy.signal.resample_poly for resampling.
    Handles stereo -> mono conversion automatically.

    Args:
        input_path: Path to source WAV file.
        output_path: Path to write resampled WAV file.
    """
    data, sr = sf.read(str(input_path), dtype="float32")

    # Stereo -> mono: average channels
    if data.ndim == 2:
        data = data.mean(axis=1)

    # Resample if needed
    if sr != JULIUS_SAMPLE_RATE:
        g = gcd(sr, JULIUS_SAMPLE_RATE)
        up = JULIUS_SAMPLE_RATE // g
        down = sr // g
        data = resample_poly(data, up, down).astype(np.float32)

    # Write as 16-bit PCM WAV
    sf.write(str(output_path), data, JULIUS_SAMPLE_RATE, subtype="PCM_16")


def make_output_name(wav_path):
    """Generate output filename from wav path: {spk}_{utt_id}.

    Input path pattern: .../wavs/{spkXXX}/{utt_id}.wav
    Output: spkXXX_utt_id  (no extension)
    """
    wav_p = Path(wav_path)
    spk_name = wav_p.parent.name  # e.g. "jvs001"
    utt_id = wav_p.stem  # e.g. "BASIC5000_0025"
    return f"{spk_name}_{utt_id}"


def _process_one(args_tuple):
    """Worker function for parallel processing.

    Accepts a tuple (wav_path, text, wav_out_path, txt_out_path).
    Returns (output_name, error_or_None).
    """
    wav_path, text, wav_out_path, txt_out_path = args_tuple
    output_name = Path(wav_out_path).stem
    try:
        # Resample audio
        resample_to_16k(wav_path, wav_out_path)

        # Convert text to katakana
        kana = text_to_katakana(text)
        if not kana:
            return output_name, f"Empty katakana for text: {text!r}"

        # Write katakana text file
        Path(txt_out_path).write_text(kana, encoding="utf-8")

        return output_name, None
    except Exception as e:
        return output_name, str(e)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare JVS data for Julius segmentation-kit forced alignment."
    )
    parser.add_argument(
        "--filelist",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to filelist(s) (format: wav_path|speaker_id|text)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for Julius input files",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    wav_dir = output_dir / "wav"
    txt_dir = output_dir / "txt"
    wav_dir.mkdir(parents=True, exist_ok=True)
    txt_dir.mkdir(parents=True, exist_ok=True)

    # Collect entries from all filelists
    all_entries = []
    for filelist_path in args.filelist:
        entries = parse_filelist(filelist_path)
        print(f"Loaded {len(entries)} entries from {filelist_path}")
        all_entries.extend(entries)

    # Deduplicate by wav_path (same utterance may appear in multiple lists)
    seen = set()
    unique_entries = []
    for wav_path, spk_id, text in all_entries:
        if wav_path not in seen:
            seen.add(wav_path)
            unique_entries.append((wav_path, spk_id, text))
    if len(unique_entries) < len(all_entries):
        print(f"Deduplicated: {len(all_entries)} -> {len(unique_entries)} entries")

    print(f"Total entries to process: {len(unique_entries)}")
    print(f"Output directory: {output_dir}")

    # Build task list
    tasks = []
    for wav_path, _spk_id, text in unique_entries:
        name = make_output_name(wav_path)
        wav_out = str(wav_dir / f"{name}.wav")
        txt_out = str(txt_dir / f"{name}.txt")
        tasks.append((wav_path, text, wav_out, txt_out))

    # Process in parallel
    errors = []
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(_process_one, task): task[0] for task in tasks}
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Preparing Julius input",
            unit="files",
        ):
            wav_path = futures[future]
            try:
                name, error = future.result()
                if error:
                    errors.append((name, error))
                    tqdm.write(f"ERROR [{name}]: {error}")
            except Exception as e:
                errors.append((wav_path, str(e)))
                tqdm.write(f"ERROR [{wav_path}]: {e}")

    elapsed = time.time() - start_time
    success_count = len(tasks) - len(errors)
    speed = success_count / elapsed if elapsed > 0 else 0

    print(f"\nResults:")
    print(f"  Success: {success_count}")
    print(f"  Errors:  {len(errors)}")
    print(f"  Speed:   {speed:.1f} files/sec ({elapsed:.1f}s total)")

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for name, msg in errors[:20]:
            print(f"  {name}: {msg}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")

    # Write manifest for downstream use
    manifest_path = output_dir / "manifest.txt"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for wav_path, spk_id, text in unique_entries:
            name = make_output_name(wav_path)
            f.write(f"{name}|{spk_id}|{text}\n")
    print(f"\nManifest written: {manifest_path} ({len(unique_entries)} entries)")


if __name__ == "__main__":
    main()
