"""Pre-compute mel spectrograms and text sequences for the JVS dataset.

Processes a filelist and saves each sample as a .pt file containing the mel
spectrogram, integer text sequence (with blank insertion), speaker ID, and
cleaned text.

Usage:
    python scripts/precompute_dataset.py \
        --filelist data/jvs/filelist_train.txt \
        --output-dir data/jvs/precomputed \
        --num-workers 8
"""

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio as ta
from tqdm import tqdm

from matcha.text import text_to_sequence
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


def parse_filelist(filelist_path: str):
    """Parse a pipe-delimited filelist (wav_path|speaker_id|text)."""
    with open(filelist_path, encoding="utf-8") as f:
        filepaths_and_text = [line.strip().split("|") for line in f if line.strip()]
    return filepaths_and_text


def load_duration(durations_dir: Path, spk_name: str, stem: str, expected_text_len: int):
    """Load a .npy duration file and validate length against text sequence."""
    npy_name = f"{spk_name}_{stem}.npy"
    npy_path = durations_dir / npy_name
    if not npy_path.exists():
        return None
    dur = np.load(str(npy_path))
    dur_tensor = torch.from_numpy(dur).int()
    if len(dur_tensor) != expected_text_len:
        raise ValueError(
            f"Duration length mismatch for {npy_name}: "
            f"duration={len(dur_tensor)}, text={expected_text_len}"
        )
    return dur_tensor


def process_sample(
    wav_path: str,
    spk: int,
    text: str,
    output_dir: Path,
    mel_mean: float,
    mel_std: float,
    durations_dir=None,  # Path | None
):
    """Compute mel + text + optional duration for a single sample and save as .pt.

    Returns:
        tuple: (out_path, skipped) where skipped=True if duration was required but missing.
    """
    # Include speaker dir name to avoid collisions (e.g., jvs001/VOICEACTRESS100_001)
    wav_p = Path(wav_path)
    spk_name = wav_p.parent.name  # e.g. "jvs001"
    out_path = output_dir / f"{spk_name}_{wav_p.stem}.pt"

    # -- mel spectrogram --
    data, sr = sf.read(wav_path, dtype="float32")
    assert sr == SAMPLE_RATE, f"Expected {SAMPLE_RATE} Hz, got {sr} Hz for {wav_path}"
    audio = torch.from_numpy(data).unsqueeze(0)  # (1, samples)
    mel = mel_spectrogram(
        audio,
        N_FFT,
        N_MELS,
        SAMPLE_RATE,
        HOP_LENGTH,
        WIN_LENGTH,
        F_MIN,
        F_MAX,
        center=False,
    ).squeeze()
    mel = normalize(mel, mel_mean, mel_std)

    # -- text sequence --
    text_norm, cleaned_text = text_to_sequence(text, ["japanese_cleaners"], language="ja")
    text_norm = intersperse(text_norm, 0)  # add_blank=True
    text_norm = torch.IntTensor(text_norm)

    # -- duration (optional) --
    duration = None
    if durations_dir is not None:
        duration = load_duration(Path(durations_dir), spk_name, wav_p.stem, len(text_norm))
        if duration is None:
            return out_path, True  # skipped

    # -- save --
    save_dict = {
        "mel": mel,
        "text": text_norm,
        "spk": spk,
        "cleaned_text": cleaned_text,
    }
    if duration is not None:
        save_dict["durations"] = duration
    torch.save(save_dict, out_path)
    return out_path, False


def process_sample_text_only(text: str):
    """Compute text sequence only (for GPU mel mode where mel is done in main thread)."""
    text_norm, cleaned_text = text_to_sequence(text, ["japanese_cleaners"], language="ja")
    text_norm = intersperse(text_norm, 0)  # add_blank=True
    text_norm = torch.IntTensor(text_norm)
    return text_norm, cleaned_text


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute mel spectrograms and text sequences for JVS dataset."
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
        help="Directory to write .pt files",
    )
    parser.add_argument(
        "--mel-mean",
        type=float,
        default=-6.963938,
        help="Mean for mel normalization (default: -6.963938)",
    )
    parser.add_argument(
        "--mel-std",
        type=float,
        default=2.478805,
        help="Std for mel normalization (default: 2.478805)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8)",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU for mel spectrogram computation (runs mel on main thread, text in parallel)",
    )
    parser.add_argument(
        "--durations-dir",
        type=str,
        default=None,
        help="Directory containing .npy duration files from Julius forced alignment. "
             "Naming: {spk_name}_{utterance_id}.npy",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    entries = parse_filelist(args.filelist)
    print(f"Loaded {len(entries)} entries from {args.filelist}")
    print(f"Output directory: {output_dir}")
    print(f"Mel normalization: mean={args.mel_mean}, std={args.mel_std}")
    print(f"Workers: {args.num_workers}")

    use_gpu = args.gpu and torch.cuda.is_available()
    if args.gpu and not torch.cuda.is_available():
        print("WARNING: --gpu specified but CUDA is not available, falling back to CPU")
    if use_gpu:
        print("GPU mel computation: enabled (mel on main thread, text processing in parallel)")

    errors = []
    skipped_count = 0
    start_time = time.time()

    if use_gpu:
        # GPU mode: compute mel on GPU in main thread, text processing in parallel
        # This avoids issues with CUDA and multiprocessing
        text_results = {}
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {}
            for entry in entries:
                wav_path, spk_str, text = entry[0], entry[1], entry[2]
                future = executor.submit(process_sample_text_only, text)
                futures[future] = (wav_path, int(spk_str), text)

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Text processing",
                unit="samples",
            ):
                wav_path, spk, text = futures[future]
                try:
                    text_norm, cleaned_text = future.result()
                    text_results[wav_path] = (spk, text_norm, cleaned_text)
                except Exception as e:
                    errors.append((wav_path, str(e)))
                    tqdm.write(f"ERROR [text] [{wav_path}]: {e}")

        # Now compute mel spectrograms on GPU in main thread
        for wav_path, (spk, text_norm, cleaned_text) in tqdm(
            text_results.items(),
            desc="GPU mel computation",
            unit="samples",
        ):
            try:
                wav_p = Path(wav_path)
                spk_name = wav_p.parent.name
                out_path = output_dir / f"{spk_name}_{wav_p.stem}.pt"

                audio, sr = ta.load(wav_path)
                assert sr == SAMPLE_RATE, f"Expected {SAMPLE_RATE} Hz, got {sr} Hz for {wav_path}"
                audio = audio.to("cuda")
                mel = mel_spectrogram(
                    audio,
                    N_FFT,
                    N_MELS,
                    SAMPLE_RATE,
                    HOP_LENGTH,
                    WIN_LENGTH,
                    F_MIN,
                    F_MAX,
                    center=False,
                ).squeeze()
                mel = mel.cpu()
                mel = normalize(mel, args.mel_mean, args.mel_std)

                duration = None
                if args.durations_dir:
                    dur_dir = Path(args.durations_dir)
                    duration = load_duration(dur_dir, spk_name, wav_p.stem, len(text_norm))
                    if duration is None:
                        skipped_count += 1
                        tqdm.write(f"SKIP: Duration not found for {wav_path}")
                        continue

                save_dict = {"mel": mel, "text": text_norm, "spk": spk, "cleaned_text": cleaned_text}
                if duration is not None:
                    save_dict["durations"] = duration
                torch.save(save_dict, out_path)
            except Exception as e:
                errors.append((wav_path, str(e)))
                tqdm.write(f"ERROR [mel] [{wav_path}]: {e}")
    else:
        # CPU mode: process everything in parallel with ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {}
            for entry in entries:
                wav_path, spk_str, text = entry[0], entry[1], entry[2]
                spk = int(spk_str)
                future = executor.submit(
                    process_sample,
                    wav_path,
                    spk,
                    text,
                    output_dir,
                    args.mel_mean,
                    args.mel_std,
                    Path(args.durations_dir) if args.durations_dir else None,
                )
                futures[future] = wav_path

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Processing",
                unit="samples",
            ):
                wav_path = futures[future]
                try:
                    result = future.result()
                    if isinstance(result, tuple):
                        out_path, skipped = result
                        if skipped:
                            skipped_count += 1
                            tqdm.write(f"SKIP: Duration not found for {wav_path}")
                except Exception as e:
                    errors.append((wav_path, str(e)))
                    tqdm.write(f"ERROR [{wav_path}]: {e}")

    elapsed = time.time() - start_time
    total_processed = len(entries) - len(errors)
    speed = total_processed / elapsed if elapsed > 0 else 0

    if errors:
        print(f"\nCompleted with {len(errors)} error(s):")
        for path, msg in errors:
            print(f"  {path}: {msg}")
    else:
        print(f"\nDone. Saved {len(entries)} .pt files to {output_dir}")
    print(f"Processing speed: {speed:.1f} samples/sec ({elapsed:.1f}s total)")
    if args.durations_dir:
        dur_count = total_processed - skipped_count
        print(f"Durations: {dur_count} embedded, {skipped_count} skipped (no .npy found)")


if __name__ == "__main__":
    main()
