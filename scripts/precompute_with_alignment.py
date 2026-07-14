"""Pre-compute .pt files with Julius alignment durations in a single pass.

Combines the duration conversion (lab -> duration) and mel precomputation steps
into one pass, eliminating intermediate .npy files.

Two execution paths are available:

1. **fast path** (default, since Phase 1-5 2026-04-15):
   - single-process + ThreadPool producer/consumer
   - shared text_to_sequence cache (pre-built for all unique texts)
   - GPU-batched mel spectrogram (optional, --device cuda)
   - ~20x faster than legacy (12,348 samples: 25.2min → 1.2min)

2. **legacy path** (--legacy):
   - ProcessPoolExecutor with N workers
   - kept for debugging and binary-compat verification

Usage (fast CPU, default and recommended — all tunables are at their defaults):
    uv run python scripts/precompute_with_alignment.py \
        --filelist data/jvs/train.txt \
        --lab-dir data/julius_work/wav \
        --output-dir data/jvs_precomputed_aligned/train \
        --mel-mean -6.550095 --mel-std 2.383771

Usage (legacy, ProcessPool):
    uv run python scripts/precompute_with_alignment.py \
        --filelist data/jvs/train.txt \
        --lab-dir data/julius_work/wav \
        --output-dir data/jvs_precomputed_aligned/train \
        --mel-mean -6.550095 --mel-std 2.383771 \
        --num-workers 8 --legacy

Note: GPU mel (--device cuda) is implemented but benchmark showed it's slower
than CPU due to H2D transfer and reflect-padding overhead. Default is CPU.
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from librosa.filters import mel as librosa_mel_fn
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


# ---------------------------------------------------------------------------
# Task dataclass (shared between legacy and fast paths)
# ---------------------------------------------------------------------------


@dataclass
class Task:
    wav_path: str
    spk: int
    text: str
    lab_path: str
    out_path: str
    name: str  # "{spk_name}_{stem}"


def parse_filelist(filelist_path: str) -> list[list[str]]:
    """Parse a pipe-delimited filelist (wav_path|speaker_id|text)."""
    with open(filelist_path, encoding="utf-8") as f:
        filepaths_and_text = [line.strip().split("|") for line in f if line.strip()]
    return filepaths_and_text


def build_tasks(entries: list[list[str]], lab_dir: Path, output_dir: Path) -> list[Task]:
    """Build the list of Task objects from filelist entries."""
    tasks: list[Task] = []
    for entry in entries:
        wav_path, spk_str, text = entry[0], entry[1], entry[2]
        wav_p = Path(wav_path)
        spk_name = wav_p.parent.name
        name = f"{spk_name}_{wav_p.stem}"
        lab_path = lab_dir / f"{name}.lab"
        out_path = output_dir / f"{name}.pt"
        tasks.append(
            Task(
                wav_path=wav_path,
                spk=int(spk_str),
                text=text,
                lab_path=str(lab_path),
                out_path=str(out_path),
                name=name,
            )
        )
    return tasks


# ---------------------------------------------------------------------------
# Legacy path (ProcessPoolExecutor) — kept intact for compatibility
# ---------------------------------------------------------------------------


def compute_duration_from_lab(
    lab_path: str | Path,
    text: str,
    total_mel_frames: int,
    align_mode: str = "auto",
) -> tuple[np.ndarray | None, str]:
    """Compute a blank-interspersed duration array from a Julius .lab file."""
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
        return None, (f"Length mismatch: duration array {len(duration_array)} vs interspersed text {expected_len}")

    return duration_array, f"OK ({len(pyopenjtalk_phonemes)} phones, {total_mel_frames} frames)"


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
    """Process a single sample: compute mel, text, and duration from .lab in one pass."""
    lab_p = Path(lab_path)
    out_p = Path(output_path)

    if not lab_p.exists():
        return str(out_p), True, "no .lab file"

    try:
        # 1. Mel spectrogram
        data, sr = sf.read(wav_path, dtype="float32")
        assert sr == SAMPLE_RATE, f"Expected {SAMPLE_RATE} Hz, got {sr}"
        if data.ndim > 1:
            data = data.mean(axis=1).astype(np.float32)  # downmix to mono, same as fast-path _load_one
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
        duration_array, msg = compute_duration_from_lab(lab_path, text, mel_frames, align_mode=align_mode)
        if duration_array is None:
            return str(out_p), True, f"duration conversion failed: {msg}"

        duration = torch.from_numpy(duration_array).long()

        # 4. Validate
        if len(duration) != len(text_tensor):
            return str(out_p), True, (f"duration/text length mismatch: {len(duration)} vs {len(text_tensor)}")
        if duration.sum().item() != mel_frames:
            return str(out_p), True, (f"duration sum mismatch: {duration.sum()} vs {mel_frames}")

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


def run_legacy_pipeline(tasks: list[Task], args) -> tuple[int, int, list]:
    """Legacy ProcessPoolExecutor-based pipeline."""
    success = 0
    skip = 0
    errors: list[tuple[str, str]] = []

    with ProcessPoolExecutor(
        max_workers=args.num_workers, initializer=_apply_fmax, initargs=(args.fmax,)
    ) as executor:
        futures = {
            executor.submit(
                process_sample_with_alignment,
                t.wav_path,
                t.spk,
                t.text,
                t.lab_path,
                t.out_path,
                args.mel_mean,
                args.mel_std,
                args.align_mode,
            ): t.out_path
            for t in tasks
        }
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Processing (legacy)",
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

    return success, skip, errors


# ---------------------------------------------------------------------------
# Fast path — Phase 2 (text cache + sequential fast loop)
# ---------------------------------------------------------------------------


@dataclass
class LoadedSample:
    task: Task
    audio: np.ndarray | None
    text_seq: list[int] | None
    cleaned_text: str | None
    lab_segments: list | None
    error: str | None = None


def _text_cache_worker(text: str) -> tuple[str, list[int], str]:
    from matcha.text import text_to_sequence as _t2s

    seq, cleaned = _t2s(text, ["japanese_cleaners"], language="ja")
    return text, seq, cleaned


def build_text_sequence_cache(texts: list[str], num_workers: int) -> dict[str, tuple[list[int], str]]:
    """Pre-compute text_to_sequence for every unique text in parallel."""
    unique_texts = list(dict.fromkeys(texts))
    n_unique = len(unique_texts)
    print(f"[text-cache] {len(texts)} total, {n_unique} unique texts (workers={num_workers})")

    t0 = time.time()
    cache: dict[str, tuple[list[int], str]] = {}

    if n_unique <= 100 or num_workers <= 1:
        for t in tqdm(unique_texts, desc="text-cache (seq)", unit="texts"):
            _, seq, cleaned = _text_cache_worker(t)
            cache[t] = (seq, cleaned)
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            futures = [ex.submit(_text_cache_worker, t) for t in unique_texts]
            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="text-cache",
                unit="texts",
            ):
                t, seq, cleaned = fut.result()
                cache[t] = (seq, cleaned)

    elapsed = time.time() - t0
    print(f"[text-cache] built in {elapsed:.1f}s ({n_unique / max(elapsed, 1e-9):.1f} texts/s)")
    return cache


def _load_one(task: Task, text_cache: dict[str, tuple[list[int], str]]) -> LoadedSample:
    """Load wav + lab + text-cache lookup for a single task."""
    lab_p = Path(task.lab_path)
    if not lab_p.exists():
        return LoadedSample(task, None, None, None, None, error="no .lab file")

    try:
        data, sr = sf.read(task.wav_path, dtype="float32")
        assert sr == SAMPLE_RATE, f"Expected {SAMPLE_RATE} Hz, got {sr}"
        if data.ndim > 1:
            data = data.mean(axis=1).astype(np.float32)

        segments = parse_lab_file(task.lab_path)
        if not segments:
            return LoadedSample(task, None, None, None, None, error=f"Empty .lab file: {task.lab_path}")

        cached = text_cache.get(task.text)
        if cached is None:
            return LoadedSample(task, None, None, None, None, error="text not in cache")
        seq, cleaned = cached

        return LoadedSample(
            task=task,
            audio=data,
            text_seq=seq,
            cleaned_text=cleaned,
            lab_segments=segments,
        )
    except Exception as e:  # noqa: BLE001
        return LoadedSample(task, None, None, None, None, error=f"load error: {e}")


# ---------------------------------------------------------------------------
# Batched GPU mel (Phase 4)
# ---------------------------------------------------------------------------


_MEL_BASIS_CACHE: dict[str, torch.Tensor] = {}
_HANN_CACHE: dict[str, torch.Tensor] = {}


def _apply_fmax(fmax: int) -> None:
    """Set the module-level mel fmax used by every mel path (CPU/GPU/fast) + basis cache.

    Called in main() for the in-process fast/GPU path and passed as the
    ProcessPoolExecutor initializer so spawned legacy workers pick up the same
    fmax (they re-import the module with the default otherwise). Default fmax
    (8000) keeps existing JVS output byte-identical.
    """
    global F_MAX
    F_MAX = fmax
    # basis cache is keyed by device, not fmax — clear so it repopulates at the new fmax
    _MEL_BASIS_CACHE.clear()
    _HANN_CACHE.clear()


def _get_mel_basis_hann(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    key = str(device)
    if key not in _MEL_BASIS_CACHE:
        mel_np = librosa_mel_fn(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS, fmin=F_MIN, fmax=F_MAX)
        _MEL_BASIS_CACHE[key] = torch.from_numpy(mel_np).float().to(device)
        _HANN_CACHE[key] = torch.hann_window(WIN_LENGTH).to(device)
    return _MEL_BASIS_CACHE[key], _HANN_CACHE[key]


def _mel_batch_gpu(
    audios: list[np.ndarray],
    device: torch.device,
    mel_mean: float,
    mel_std: float,
) -> list[torch.Tensor]:
    """Compute mel for a batch of variable-length audios on ``device``.

    Bit-exact with ``_mel_single_cpu`` within fp32 tolerance: we pre-reflect-pad
    each row by (n_fft - hop) // 2 = 384 on both sides (matching mel_spectrogram's
    internal pad), then zero-pad on the right to the batch max and run STFT with
    ``center=False``. Only frames ``[0, T_i)`` are read back where
    ``T_i = 1 + (len_i - HOP_LENGTH) // HOP_LENGTH``, which end at padded index
    ``len_i + 768`` — exactly the end of the valid reflect-padded region, so the
    zero tail never enters any returned frame.
    """
    assert len(audios) > 0
    pad = (N_FFT - HOP_LENGTH) // 2  # 384

    # Pre-reflect-pad each row individually (reflect needs the true audio bounds).
    padded_rows: list[torch.Tensor] = []
    lengths: list[int] = []
    for a in audios:
        L = len(a)
        lengths.append(L)
        t = torch.from_numpy(a).to(device=device, dtype=torch.float32)
        t = F.pad(t.view(1, 1, -1), (pad, pad), mode="reflect").view(-1)
        padded_rows.append(t)

    L_pad_max = max(r.numel() for r in padded_rows)
    B = len(padded_rows)
    batch = torch.zeros(B, L_pad_max, dtype=torch.float32, device=device)
    for i, r in enumerate(padded_rows):
        batch[i, : r.numel()] = r

    mel_basis, hann = _get_mel_basis_hann(device)

    try:
        spec = torch.stft(
            batch,
            N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            window=hann,
            center=False,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
    except RuntimeError as e:
        # OOM fallback: split batch in half
        msg = str(e).lower()
        if "out of memory" in msg or "cuda" in msg:
            torch.cuda.empty_cache() if device.type == "cuda" else None
            if B == 1:
                raise
            half = B // 2
            left = _mel_batch_gpu(audios[:half], device, mel_mean, mel_std)
            right = _mel_batch_gpu(audios[half:], device, mel_mean, mel_std)
            return left + right
        raise

    spec = torch.view_as_real(spec)
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)  # (B, F, T_max)
    spec = torch.matmul(mel_basis, spec)  # (B, N_MELS, T_max)
    spec = torch.log(torch.clamp(spec, min=1e-5))

    spec = (spec - mel_mean) / mel_std

    out: list[torch.Tensor] = []
    for i, L in enumerate(lengths):
        T_i = 1 + (L - HOP_LENGTH) // HOP_LENGTH
        mel_i = spec[i, :, :T_i].contiguous().cpu()
        out.append(mel_i)
    return out


def _mel_single_cpu(audio: np.ndarray, mel_mean: float, mel_std: float) -> torch.Tensor:
    audio_t = torch.from_numpy(audio).unsqueeze(0)
    mel = mel_spectrogram(
        audio_t,
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
    return mel


def _finalize_one(sample: LoadedSample, mel: torch.Tensor, align_mode: str) -> tuple[str, bool, str]:
    """Compute duration, validate and save .pt. No text_to_sequence call."""
    task = sample.task
    out_path = task.out_path
    try:
        mel_frames = mel.shape[-1]

        # 1. Julius → pyopenjtalk alignment (reuse cached lab_segments / text_seq)
        julius_raw = [seg[2] for seg in sample.lab_segments]  # type: ignore[arg-type]
        julius_mapped = map_julius_sequence(julius_raw)
        julius_frame_durations = [
            time_to_frames(seg[0], seg[1])
            for seg in sample.lab_segments  # type: ignore[arg-type]
        ]

        pyopenjtalk_phonemes = sample.cleaned_text.split()  # type: ignore[union-attr]

        aligned_durations = align_julius_with_pyopenjtalk(
            julius_mapped,
            pyopenjtalk_phonemes,
            julius_frame_durations,
            align_mode=align_mode,
        )

        duration_array = build_duration_array_with_blanks(aligned_durations, mel_frames)
        duration = torch.from_numpy(duration_array).long()

        # 2. Build interspersed text tensor from cached seq
        text_interspersed = intersperse(sample.text_seq, 0)  # type: ignore[arg-type]
        text_tensor = torch.IntTensor(text_interspersed)

        # 3. Validate
        if len(duration) != len(text_tensor):
            return out_path, True, (f"duration/text length mismatch: {len(duration)} vs {len(text_tensor)}")
        if duration.sum().item() != mel_frames:
            return out_path, True, (f"duration sum mismatch: {duration.sum()} vs {mel_frames}")

        # 4. Save
        torch.save(
            {
                "mel": mel,
                "text": text_tensor,
                "spk": task.spk,
                "cleaned_text": sample.cleaned_text,
                "durations": duration,
            },
            out_path,
        )
        return out_path, False, "ok"
    except Exception as e:  # noqa: BLE001
        return out_path, True, f"Error: {e}"


def run_fast_pipeline(
    tasks: list[Task],
    args,
    text_cache: dict[str, tuple[list[int], str]],
    device: torch.device,
) -> tuple[int, int, list]:
    """Producer/consumer fast pipeline (Phase 3) with optional GPU batching (Phase 4)."""
    success = 0
    skip = 0
    errors: list[tuple[str, str]] = []

    io_workers = max(1, args.io_workers)
    save_workers = max(1, args.io_workers)
    io_pool = ThreadPoolExecutor(max_workers=io_workers, thread_name_prefix="io")
    save_pool = ThreadPoolExecutor(max_workers=save_workers, thread_name_prefix="save")

    try:
        load_futures = [io_pool.submit(_load_one, t, text_cache) for t in tasks]

        save_futures: list = []
        pending: list[LoadedSample] = []
        batch_size = max(1, args.batch_size)
        use_gpu_batch = device.type == "cuda"

        pbar = tqdm(total=len(tasks), desc="Fast precompute", unit="samples")

        def _reap_saves(block: bool = False) -> None:
            nonlocal success, skip, save_futures
            still: list = []
            for sf_fut in save_futures:
                if block or sf_fut.done():
                    try:
                        _, skipped_flag, msg = sf_fut.result()
                        if skipped_flag:
                            skip += 1
                            tqdm.write(f"SKIP: {msg}")
                        else:
                            success += 1
                    except Exception as e:  # noqa: BLE001
                        errors.append(("save", str(e)))
                    pbar.update(1)
                else:
                    still.append(sf_fut)
            save_futures = still

        def _flush() -> None:
            nonlocal pending
            if not pending:
                return
            try:
                if use_gpu_batch and len(pending) >= 1:
                    mels = _mel_batch_gpu(
                        [s.audio for s in pending],  # type: ignore[misc]
                        device,
                        args.mel_mean,
                        args.mel_std,
                    )
                else:
                    mels = [
                        _mel_single_cpu(s.audio, args.mel_mean, args.mel_std)  # type: ignore[arg-type]
                        for s in pending
                    ]
            except Exception as e:  # noqa: BLE001
                for s in pending:
                    errors.append((s.task.out_path, f"mel error: {e}"))
                    pbar.update(1)
                pending = []
                return

            for s, mel in zip(pending, mels):
                save_futures.append(save_pool.submit(_finalize_one, s, mel, args.align_mode))
            pending = []

        for fut in load_futures:
            sample = fut.result()
            if sample.error is not None:
                skip += 1
                tqdm.write(f"SKIP [{sample.task.name}]: {sample.error}")
                pbar.update(1)
                continue
            pending.append(sample)
            if len(pending) >= batch_size:
                _flush()
            _reap_saves(block=False)

        _flush()
        _reap_saves(block=True)
        pbar.close()
    finally:
        io_pool.shutdown(wait=True)
        save_pool.shutdown(wait=True)

    return success, skip, errors


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Pre-compute .pt files with Julius durations in a single pass.")
    parser.add_argument("--filelist", type=str, required=True)
    parser.add_argument("--lab-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--mel-mean", type=float, default=-6.550095)
    parser.add_argument("--mel-std", type=float, default=2.383771)
    parser.add_argument(
        "--fmax",
        type=int,
        default=F_MAX,
        help="Mel fmax in Hz. Default 8000 keeps existing JVS/Julius output byte-identical. "
        "fmax=11025 pipeline (Julius fallback for moe/tsukuyomi): pass 11025.",
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--align-mode",
        type=str,
        default="auto",
        choices=["sequential", "dtw", "auto"],
    )
    # Phase 1 additions
    parser.add_argument("--legacy", action="store_true", help="Use legacy ProcessPoolExecutor path")
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu", "auto"],
        default="cpu",
        help="Device for mel computation in fast path "
        "(default: cpu — Phase 5 benchmark showed CPU is faster than GPU due to "
        "H2D transfer and reflect-padding overhead)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Mel batch size (default: 32)",
    )
    parser.add_argument(
        "--io-workers",
        type=int,
        default=16,
        help="IO worker threads for wav/lab reading and .pt saving (default: 16)",
    )
    parser.add_argument("--text-cache-workers", type=int, default=None)
    parser.add_argument(
        "--bench-only",
        type=int,
        default=0,
        help="If >0, only process the first N tasks (for benchmarking)",
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
    print(f"Mel fmax: {args.fmax} Hz")
    print(f"Workers: {args.num_workers}, Align mode: {args.align_mode}")
    print(f"Mode: {'legacy' if args.legacy else 'fast'}")

    # Set fmax for the in-process fast/GPU path (legacy workers get it via initializer below).
    _apply_fmax(args.fmax)

    tasks = build_tasks(entries, lab_dir, output_dir)
    if args.bench_only and args.bench_only > 0:
        tasks = tasks[: args.bench_only]
        print(f"[bench-only] truncated to {len(tasks)} tasks")

    start_time = time.time()

    if args.legacy:
        success, skip, errors = run_legacy_pipeline(tasks, args)
    else:
        # Resolve device
        if args.device == "auto":
            dev_str = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            dev_str = args.device
        if dev_str == "cuda" and not torch.cuda.is_available():
            print("[fast] cuda requested but unavailable, falling back to cpu")
            dev_str = "cpu"
        device = torch.device(dev_str)
        print(f"[fast] device={device} batch_size={args.batch_size} io_workers={args.io_workers}")

        # Step A: collect unique texts
        unique_texts = [t.text for t in tasks]

        # Step B: build text cache
        tc_workers = args.text_cache_workers or args.num_workers
        text_cache = build_text_sequence_cache(unique_texts, tc_workers)

        # Step C: run fast pipeline
        success, skip, errors = run_fast_pipeline(tasks, args, text_cache, device)

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
