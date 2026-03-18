"""Prepare JSUT corpus for Matcha-TTS training.

Usage:
    uv run python scripts/prepare_jsut.py --jsut-dir data/jsut_ver1.1 --output-dir data/jsut

Steps:
    1. Validate the JSUT directory structure
    2. Resample audio from 48 kHz to 22050 Hz
    3. Generate Matcha-TTS file lists (audio_path|text)
    4. Split into train/val sets (90/10)
"""

import argparse
import random
from pathlib import Path

import torchaudio


def resample_audio(input_path, output_path, orig_sr=48000, target_sr=22050):
    """Resample a single audio file."""
    waveform, sr = torchaudio.load(input_path)
    if sr != orig_sr:
        print(f"  [!] Expected {orig_sr} Hz but got {sr} Hz for {input_path}")
        orig_sr = sr
    if orig_sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
        waveform = resampler(waveform)
    torchaudio.save(str(output_path), waveform, target_sr)


def parse_transcript(transcript_path):
    """Parse JSUT transcript_utf8.txt → dict of {utterance_id: text}."""
    entries = {}
    with open(transcript_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            utt_id, text = line.split(":", 1)
            entries[utt_id.strip()] = text.strip()
    return entries


def main():
    parser = argparse.ArgumentParser(description="Prepare JSUT corpus for Matcha-TTS")
    parser.add_argument("--jsut-dir", type=str, required=True, help="Path to jsut_ver1.1 directory")
    parser.add_argument("--output-dir", type=str, default="data/jsut", help="Output directory")
    parser.add_argument("--target-sr", type=int, default=22050, help="Target sample rate")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/val split")
    args = parser.parse_args()

    jsut_dir = Path(args.jsut_dir)
    output_dir = Path(args.output_dir)
    wavs_dir = output_dir / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)

    # Collect all sub-directories that contain transcript_utf8.txt
    transcript_files = sorted(jsut_dir.glob("*/transcript_utf8.txt"))
    if not transcript_files:
        raise FileNotFoundError(
            f"No transcript_utf8.txt found in {jsut_dir}/*/. "
            "Make sure you have the correct JSUT directory structure."
        )

    print(f"[*] Found {len(transcript_files)} transcript files in {jsut_dir}")

    # Parse all transcripts and resample audio
    filelist = []
    for transcript_path in transcript_files:
        subset_dir = transcript_path.parent
        subset_name = subset_dir.name
        wav_source_dir = subset_dir / "wav"

        if not wav_source_dir.exists():
            print(f"  [!] Skipping {subset_name}: no wav/ directory")
            continue

        entries = parse_transcript(transcript_path)
        print(f"  [{subset_name}] {len(entries)} utterances")

        for utt_id, text in entries.items():
            src_wav = wav_source_dir / f"{utt_id}.wav"
            if not src_wav.exists():
                print(f"    [!] Missing: {src_wav}")
                continue

            dst_wav = wavs_dir / f"{utt_id}.wav"
            if not dst_wav.exists():
                resample_audio(src_wav, dst_wav, target_sr=args.target_sr)

            filelist.append(f"{dst_wav.resolve()}|{text}")

    if not filelist:
        raise RuntimeError("No valid utterances found. Check your JSUT directory.")

    print(f"\n[*] Total utterances: {len(filelist)}")

    # Shuffle and split
    random.seed(args.seed)
    random.shuffle(filelist)

    val_size = max(1, int(len(filelist) * args.val_ratio))
    val_list = filelist[:val_size]
    train_list = filelist[val_size:]

    # Write file lists
    train_path = output_dir / "train.txt"
    val_path = output_dir / "val.txt"

    with open(train_path, "w", encoding="utf-8") as f:
        f.write("\n".join(train_list) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        f.write("\n".join(val_list) + "\n")

    print(f"[+] Train: {len(train_list)} utterances -> {train_path}")
    print(f"[+] Val:   {len(val_list)} utterances -> {val_path}")
    print("[+] Done! Next step: compute data statistics with matcha-data-stats")


if __name__ == "__main__":
    main()
