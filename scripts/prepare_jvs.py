"""Prepare JVS corpus for Matcha-TTS multi-speaker training.

Usage:
    uv run python scripts/prepare_jvs.py --jvs-dir /data/jvs_raw/jvs_ver1 --output-dir data/jvs

Steps:
    1. Collect parallel100 + nonpara30 subsets from all speakers
    2. Resample audio from 24 kHz to 22050 Hz
    3. Generate multi-speaker file lists (audio_path|speaker_id|text)
    4. Split into train/val sets (95/5)
"""

import argparse
import random
from pathlib import Path

import torchaudio


def resample_audio(input_path, output_path, orig_sr=24000, target_sr=22050):
    """Resample a single audio file."""
    waveform, sr = torchaudio.load(input_path)
    if sr != orig_sr:
        orig_sr = sr
    if orig_sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
        waveform = resampler(waveform)
    torchaudio.save(str(output_path), waveform, target_sr)


def parse_transcript(transcript_path):
    """Parse JVS transcripts_utf8.txt -> dict of {utterance_id: text}."""
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
    parser = argparse.ArgumentParser(description="Prepare JVS corpus for Matcha-TTS")
    parser.add_argument("--jvs-dir", type=str, required=True, help="Path to jvs_ver1 directory")
    parser.add_argument("--output-dir", type=str, default="data/jvs", help="Output directory")
    parser.add_argument("--target-sr", type=int, default=22050, help="Target sample rate")
    parser.add_argument("--val-ratio", type=float, default=0.05, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/val split")
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=["parallel100", "nonpara30"],
        help="Subsets to include (default: parallel100 nonpara30)",
    )
    args = parser.parse_args()

    jvs_dir = Path(args.jvs_dir)
    output_dir = Path(args.output_dir)
    wavs_dir = output_dir / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)

    # Discover speakers (jvs001 .. jvs100)
    speaker_dirs = sorted(d for d in jvs_dir.iterdir() if d.is_dir() and d.name.startswith("jvs"))
    if not speaker_dirs:
        raise FileNotFoundError(f"No speaker directories found in {jvs_dir}")

    print(f"[*] Found {len(speaker_dirs)} speakers in {jvs_dir}")

    # Map speaker names to integer IDs (0-indexed)
    spk_to_id = {d.name: i for i, d in enumerate(speaker_dirs)}

    filelist = []
    skipped = 0

    for spk_dir in speaker_dirs:
        spk_name = spk_dir.name
        spk_id = spk_to_id[spk_name]
        spk_wav_dir = wavs_dir / spk_name
        spk_wav_dir.mkdir(exist_ok=True)

        for subset in args.subsets:
            subset_dir = spk_dir / subset
            transcript_path = subset_dir / "transcripts_utf8.txt"

            if not transcript_path.exists():
                continue

            wav_source_dir = subset_dir / "wav24kHz16bit"
            if not wav_source_dir.exists():
                continue

            entries = parse_transcript(transcript_path)

            for utt_id, text in entries.items():
                src_wav = wav_source_dir / f"{utt_id}.wav"
                if not src_wav.exists():
                    skipped += 1
                    continue

                dst_wav = spk_wav_dir / f"{utt_id}.wav"
                if not dst_wav.exists():
                    resample_audio(src_wav, dst_wav, target_sr=args.target_sr)

                filelist.append(f"{dst_wav.resolve()}|{spk_id}|{text}")

        count = sum(1 for e in filelist if f"|{spk_id}|" in e)
        if count > 0:
            print(f"  [{spk_name}] spk_id={spk_id}, {count} utterances")

    if not filelist:
        raise RuntimeError("No valid utterances found.")

    print(f"\n[*] Total utterances: {len(filelist)}")
    if skipped:
        print(f"[!] Skipped {skipped} missing wav files")

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

    # Write speaker mapping
    spk_map_path = output_dir / "speakers.txt"
    with open(spk_map_path, "w", encoding="utf-8") as f:
        for name, sid in sorted(spk_to_id.items(), key=lambda x: x[1]):
            f.write(f"{sid}|{name}\n")

    print(f"\n[+] Train: {len(train_list)} utterances -> {train_path}")
    print(f"[+] Val:   {len(val_list)} utterances -> {val_path}")
    print(f"[+] Speaker map: {len(spk_to_id)} speakers -> {spk_map_path}")
    print("[+] Done! Next step: compute data statistics with matcha-data-stats")


if __name__ == "__main__":
    main()
