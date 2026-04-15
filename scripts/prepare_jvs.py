"""Prepare JVS corpus for Matcha-TTS multi-speaker training.

Usage:
    uv run python scripts/prepare_jvs.py --jvs-dir /data/jvs_raw/jvs_ver1 --output-dir data/jvs

    # With Julius 16kHz output for forced alignment:
    uv run python scripts/prepare_jvs.py --jvs-dir /data/jvs_raw/jvs_ver1 --output-dir data/jvs \
        --julius-output-dir data/julius_work

Steps:
    1. Collect parallel100 + nonpara30 subsets from all speakers
    2. Resample audio from 24 kHz to 22050 Hz
    3. Trim leading/trailing silence (JVS has ~500ms silence per utterance)
    4. Generate multi-speaker file lists (audio_path|speaker_id|text)
    5. Split into train/val sets (95/5)
    6. (Optional) Output 16kHz WAV + hiragana text for Julius alignment
"""

import argparse
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm


def trim_silence(waveform, sample_rate, top_db=30, margin_ms=50):
    """Trim leading and trailing silence from waveform.

    Uses energy-based detection with a safety margin to avoid clipping consonant onsets.

    Args:
        waveform: (channels, samples) tensor
        sample_rate: sample rate in Hz
        top_db: silence threshold in dB below peak amplitude
        margin_ms: safety margin in milliseconds to preserve around speech
    Returns:
        Trimmed waveform tensor
    """
    audio = waveform.squeeze(0)  # (samples,)
    if audio.numel() == 0:
        return waveform

    # Compute frame-level RMS energy (10ms frames)
    frame_length = int(sample_rate * 0.01)  # 10ms
    hop = frame_length
    n_frames = audio.numel() // hop

    if n_frames == 0:
        return waveform

    frames = audio[: n_frames * hop].reshape(n_frames, hop)
    rms = (frames**2).mean(dim=1).sqrt()

    # Threshold: top_db below peak RMS
    peak_rms = rms.max()
    if peak_rms <= 0:
        return waveform
    threshold = peak_rms * (10 ** (-top_db / 20))

    # Find first and last frame above threshold
    above = (rms > threshold).nonzero(as_tuple=True)[0]
    if len(above) == 0:
        return waveform

    start_frame = above[0].item()
    end_frame = above[-1].item()

    # Convert to samples with margin
    margin_samples = int(sample_rate * margin_ms / 1000)
    start_sample = max(0, start_frame * hop - margin_samples)
    end_sample = min(audio.numel(), (end_frame + 1) * hop + margin_samples)

    return waveform[:, start_sample:end_sample]


def resample_audio(input_path, output_path, orig_sr=24000, target_sr=22050, do_trim=True,
                   julius_output_path=None, julius_sr=16000):
    """Resample and optionally trim silence. Optionally output Julius 16kHz version too.

    Args:
        input_path: Source audio file path.
        output_path: Destination path for resampled (target_sr) audio.
        orig_sr: Expected original sample rate (overridden if file header differs).
        target_sr: Target sample rate for main output (default 22050).
        do_trim: Whether to trim leading/trailing silence.
        julius_output_path: If set, also write a 16kHz 16-bit PCM WAV here.
        julius_sr: Sample rate for Julius output (default 16000).
    """
    data, sr = sf.read(input_path, dtype="float32")
    if data.ndim == 1:
        data = data[None, :]  # (1, samples)
    else:
        data = data.T  # (channels, samples)
    waveform = torch.from_numpy(data)
    if sr != orig_sr:
        orig_sr = sr
    if orig_sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
        waveform = resampler(waveform)
    if do_trim:
        waveform = trim_silence(waveform, target_sr)
    sf.write(str(output_path), waveform.squeeze(0).numpy(), target_sr)

    # Julius 16kHz output (from already-trimmed 22kHz)
    if julius_output_path is not None:
        resampler_julius = torchaudio.transforms.Resample(target_sr, julius_sr)
        waveform_julius = resampler_julius(waveform)
        sf.write(str(julius_output_path), waveform_julius.squeeze(0).numpy(), julius_sr, subtype="PCM_16")


def _resample_worker(args_tuple):
    """Worker function for parallel resampling. Now supports dual output.

    Accepts a 4-tuple (src, dst, sr, trim) for standard mode,
    or a 6-tuple (src, dst, sr, trim, julius_wav, julius_sr) for dual output.
    """
    if len(args_tuple) == 6:
        src_wav, dst_wav, target_sr, do_trim, julius_wav, julius_sr = args_tuple
    else:
        src_wav, dst_wav, target_sr, do_trim = args_tuple
        julius_wav, julius_sr = None, None
    try:
        resample_audio(src_wav, dst_wav, target_sr=target_sr, do_trim=do_trim,
                       julius_output_path=julius_wav, julius_sr=julius_sr if julius_sr else 16000)
        return str(src_wav), None
    except Exception as e:
        return str(src_wav), str(e)


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
        "--num-workers",
        type=int,
        default=8,
        help="Number of parallel workers for resampling (default: 8)",
    )
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=["parallel100", "nonpara30"],
        help="Subsets to include (default: parallel100 nonpara30)",
    )
    parser.add_argument(
        "--no-trim-silence",
        action="store_true",
        help="Disable silence trimming (default: trim enabled)",
    )
    parser.add_argument(
        "--julius-output-dir",
        type=str,
        default=None,
        help="If set, also output 16kHz WAV + hiragana text for Julius alignment",
    )
    args = parser.parse_args()

    jvs_dir = Path(args.jvs_dir)
    output_dir = Path(args.output_dir)
    wavs_dir = output_dir / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)

    julius_dir = Path(args.julius_output_dir) if args.julius_output_dir else None
    if julius_dir:
        julius_dir.mkdir(parents=True, exist_ok=True)

    # Discover speakers (jvs001 .. jvs100)
    speaker_dirs = sorted(d for d in jvs_dir.iterdir() if d.is_dir() and d.name.startswith("jvs"))
    if not speaker_dirs:
        raise FileNotFoundError(f"No speaker directories found in {jvs_dir}")

    print(f"[*] Found {len(speaker_dirs)} speakers in {jvs_dir}")

    # Map speaker names to integer IDs (0-indexed)
    spk_to_id = {d.name: i for i, d in enumerate(speaker_dirs)}

    # Phase 1: collect all resample tasks and filelist entries
    resample_tasks = []  # (src_wav, dst_wav, target_sr)
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
                    if julius_dir:
                        julius_wav = julius_dir / f"{spk_name}_{utt_id}.wav"
                        resample_tasks.append((
                            str(src_wav), str(dst_wav), args.target_sr,
                            not args.no_trim_silence, str(julius_wav), 16000,
                        ))
                    else:
                        resample_tasks.append((
                            str(src_wav), str(dst_wav), args.target_sr,
                            not args.no_trim_silence,
                        ))

                filelist.append(f"{dst_wav.resolve()}|{spk_id}|{text}")

        count = sum(1 for e in filelist if f"|{spk_id}|" in e)
        if count > 0:
            print(f"  [{spk_name}] spk_id={spk_id}, {count} utterances")

    if not filelist:
        raise RuntimeError("No valid utterances found.")

    # Phase 2: parallel resampling
    resample_errors = []
    if resample_tasks:
        print(f"\n[*] Resampling {len(resample_tasks)} audio files with {args.num_workers} workers...")
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {
                executor.submit(_resample_worker, task): task[0] for task in resample_tasks
            }
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Resampling",
                unit="files",
            ):
                src_path = futures[future]
                try:
                    wav_path, error = future.result()
                    if error:
                        resample_errors.append((wav_path, error))
                        tqdm.write(f"ERROR [{wav_path}]: {error}")
                except Exception as e:
                    resample_errors.append((src_path, str(e)))
                    tqdm.write(f"ERROR [{src_path}]: {e}")
    else:
        print("\n[*] All audio files already resampled, skipping.")

    if resample_errors:
        print(f"[!] {len(resample_errors)} resampling error(s):")
        for path, msg in resample_errors:
            print(f"  {path}: {msg}")

    # Phase 3 (optional): Generate hiragana text files for Julius alignment
    if julius_dir:
        import re

        import pyopenjtalk

        _PUNCT_RE = re.compile(
            r"[。、！？!?,.\-\s「」『』（）\(\)【】\[\]｛｝\{\}・…―─　\u3000]"
        )

        def katakana_to_hiragana(text):
            return "".join(
                chr(ord(ch) - 0x60) if 0x30A1 <= ord(ch) <= 0x30F6 else ch
                for ch in text
            )

        print(f"\n[*] Generating hiragana text files for Julius in {julius_dir}...")
        generated_count = 0
        for entry in tqdm(filelist, desc="Hiragana text", unit="files"):
            parts = entry.split("|")
            wav_path, _spk_id, text = parts[0], parts[1], parts[2]
            wav_p = Path(wav_path)
            spk_name = wav_p.parent.name
            utt_id = wav_p.stem
            txt_path = julius_dir / f"{spk_name}_{utt_id}.txt"
            if not txt_path.exists():
                kana = pyopenjtalk.g2p(text, kana=True)
                kana = _PUNCT_RE.sub("", kana)
                hiragana = katakana_to_hiragana(kana)
                txt_path.write_text(hiragana, encoding="utf-8")
                generated_count += 1
        print(f"[+] Generated {generated_count} hiragana text files ({len(filelist) - generated_count} already existed)")

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
