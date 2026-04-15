"""Optimized Julius forced alignment pipeline with timing and unified precompute.

An optimized version of run_full_alignment_pipeline.py that integrates:
- T1-1: Julius parallel alignment (ProcessPoolExecutor with _align_single)
- T1-2: Duration conversion parallelization with sf.info for mel frame counts
- T1-3: Text-to-hiragana cache (pre-compute once, reuse across steps)
- T2-2: Unified precompute_with_alignment.py (merges Step 3+4 into one)
- T2-3: --use-shm option for /dev/shm output
- Pipeline-level timing instrumentation

Usage:
    uv run python scripts/run_optimized_pipeline.py \
        --filelist data/jvs/train.txt data/jvs/val.txt \
        --output-dir data/julius_work \
        --pt-output-dir data/jvs_precomputed_aligned \
        --mel-mean -6.550095 --mel-std 2.383771 \
        --num-workers 8

    # With /dev/shm acceleration:
    uv run python scripts/run_optimized_pipeline.py \
        --filelist data/jvs/train.txt data/jvs/val.txt \
        --output-dir data/julius_work \
        --pt-output-dir data/jvs_precomputed_aligned \
        --mel-mean -6.550095 --mel-std 2.383771 \
        --num-workers 8 --use-shm

    # Legacy mode (separate Step 3 + Step 4):
    uv run python scripts/run_optimized_pipeline.py \
        --filelist data/jvs/train.txt data/jvs/val.txt \
        --output-dir data/julius_work \
        --pt-output-dir data/jvs_precomputed_aligned \
        --mel-mean -6.550095 --mel-std 2.383771 \
        --num-workers 8 --legacy-precompute
"""

import argparse
import logging
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from math import gcd
from pathlib import Path

import soundfile as sf
from scipy.signal import resample_poly
from tqdm import tqdm

# Allow importing sibling scripts (convert_julius_to_durations, run_julius_alignment)
_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

JULIUS_SR = 16000
MATCHA_SR = 22050
HOP_LENGTH = 256
SEGKIT_DIR = Path("tools/segmentation-kit")

_PUNCT_RE = re.compile(r"[。、！？!?,.\-\s「」『』（）\(\)【】\[\]｛｝\{\}・…―─　\u3000]")


# ---------------------------------------------------------------------------
# T1-3: Text cache — pre-compute hiragana once for all entries
# ---------------------------------------------------------------------------

# Module-level cache: text -> hiragana (populated by build_text_cache)
_HIRAGANA_CACHE: dict[str, str] = {}


def katakana_to_hiragana(text: str) -> str:
    """Convert katakana to hiragana."""
    result = []
    for ch in text:
        cp = ord(ch)
        if 0x30A1 <= cp <= 0x30F6:
            result.append(chr(cp - 0x60))
        else:
            result.append(ch)
    return "".join(result)


def _text_to_hiragana_worker(text: str) -> tuple[str, str]:
    """Worker: convert a single text to hiragana. Returns (text, hiragana)."""
    import pyopenjtalk
    kana = pyopenjtalk.g2p(text, kana=True)
    kana = _PUNCT_RE.sub("", kana)
    return text, katakana_to_hiragana(kana)


def text_to_hiragana(text: str) -> str:
    """Convert Japanese text to hiragana, using cache if available."""
    if text in _HIRAGANA_CACHE:
        return _HIRAGANA_CACHE[text]
    import pyopenjtalk
    kana = pyopenjtalk.g2p(text, kana=True)
    kana = _PUNCT_RE.sub("", kana)
    result = katakana_to_hiragana(kana)
    _HIRAGANA_CACHE[text] = result
    return result


def build_text_cache(texts: list[str], num_workers: int) -> dict[str, str]:
    """Pre-compute hiragana for all unique texts in parallel.

    This is T1-3: pyopenjtalk g2p is expensive (~2ms per call) and many
    entries share the same text (JVS: 100 speakers x same scripts). Caching
    avoids redundant computation in Step 1 and Step 3.

    Args:
        texts: List of Japanese text strings (may contain duplicates).
        num_workers: Number of parallel workers.

    Returns:
        Dict mapping text -> hiragana.
    """
    unique_texts = list(set(texts))
    log.info("Text cache: %d unique texts from %d total entries", len(unique_texts), len(texts))

    cache: dict[str, str] = {}
    errors = 0

    if len(unique_texts) <= 100:
        # For small sets, run sequentially (process spawn overhead not worth it)
        for text in tqdm(unique_texts, desc="Text cache (seq)", unit="texts"):
            try:
                _, hiragana = _text_to_hiragana_worker(text)
                cache[text] = hiragana
            except Exception as e:
                log.warning("Text cache error for '%s': %s", text[:30], e)
                errors += 1
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(_text_to_hiragana_worker, text): text
                for text in unique_texts
            }
            for future in tqdm(as_completed(futures), total=len(futures),
                               desc="Text cache", unit="texts"):
                try:
                    text, hiragana = future.result()
                    cache[text] = hiragana
                except Exception as e:
                    orig_text = futures[future]
                    log.warning("Text cache error for '%s': %s", orig_text[:30], e)
                    errors += 1

    if errors:
        log.warning("Text cache: %d errors out of %d unique texts", errors, len(unique_texts))
    log.info("Text cache: %d entries built", len(cache))
    return cache


# ---------------------------------------------------------------------------
# Step 1 helpers: resample + hiragana file generation
# ---------------------------------------------------------------------------


def resample_wav(input_path: str, output_path: str):
    """Resample wav to 16kHz 16bit mono."""
    data, sr = sf.read(input_path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != JULIUS_SR:
        g = gcd(sr, JULIUS_SR)
        data = resample_poly(data, JULIUS_SR // g, sr // g)
    sf.write(output_path, data, JULIUS_SR, subtype="PCM_16")


def prepare_single(wav_path: str, text: str, output_dir: Path,
                    hiragana_cache: dict[str, str]) -> tuple[str, bool, str]:
    """Prepare one file: resample + hiragana text.

    Uses the pre-computed hiragana_cache (T1-3) to avoid redundant g2p calls.
    """
    wav_p = Path(wav_path)
    spk_name = wav_p.parent.name
    stem = wav_p.stem
    name = f"{spk_name}_{stem}"

    try:
        out_wav = output_dir / f"{name}.wav"
        out_txt = output_dir / f"{name}.txt"

        if out_wav.exists() and out_txt.exists():
            return name, False, "skipped (exists)"

        resample_wav(wav_path, str(out_wav))

        # T1-3: Use cached hiragana instead of calling pyopenjtalk
        if text in hiragana_cache:
            hiragana = hiragana_cache[text]
        else:
            hiragana = text_to_hiragana(text)
        out_txt.write_text(hiragana, encoding="utf-8")
        return name, False, "ok"
    except Exception as e:
        return name, True, str(e)


# ---------------------------------------------------------------------------
# Step 2: Julius alignment (T1-1 parallel)
# ---------------------------------------------------------------------------


def run_julius_alignment(wav_dir: Path, output_dir: Path, num_workers: int = 16) -> int:
    """Run Julius forced alignment with parallel workers.

    T1-1: Uses run_julius_alignment.py's _align_single() + ProcessPoolExecutor
    to run one Julius process per file across multiple cores.

    Args:
        wav_dir: Directory containing paired .wav and .txt files.
        output_dir: Directory to write .lab files (can be same as wav_dir).
        num_workers: Number of parallel worker processes.

    Returns:
        0 on success (partial errors are tolerated), 1 on total failure.
    """
    from run_julius_alignment import _align_single, check_prerequisites

    segkit_dir = SEGKIT_DIR.resolve()
    check_prerequisites(segkit_dir)

    # Discover .wav/.txt pairs
    wav_files = {p.stem: p for p in wav_dir.glob("*.wav")}
    txt_files = {p.stem: p for p in wav_dir.glob("*.txt")}
    matched = sorted(set(wav_files) & set(txt_files))

    if not matched:
        log.error("No matched .wav/.txt pairs found in %s", wav_dir)
        return 1

    # Skip already aligned
    already_done = {p.stem for p in output_dir.glob("*.lab")}
    todo = [n for n in matched if n not in already_done]

    log.info(
        "Julius alignment: %d matched pairs, %d already done, %d to process (workers=%d)",
        len(matched), len(already_done), len(todo), num_workers,
    )

    if not todo:
        log.info("Nothing to do. All files already aligned.")
        return 0

    # Build task tuples for _align_single(name, wav_path, txt_path, segkit_dir, output_dir, timeout)
    tasks = [
        (name, str(wav_files[name]), str(txt_files[name]),
         str(segkit_dir), str(output_dir), 300)
        for name in todo
    ]

    success = 0
    errors = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_align_single, task): task[0] for task in tasks}
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="Julius alignment", unit="files"):
            name = futures[future]
            try:
                result_name, error = future.result()
                if error:
                    errors.append(f"{result_name}: {error}")
                else:
                    success += 1
            except Exception as e:
                errors.append(f"{name}: {e}")

    log.info("Julius alignment: %d success, %d errors out of %d", success, len(errors), len(todo))
    if errors:
        for e in errors[:10]:
            log.warning("  %s", e)
        if len(errors) > 10:
            log.warning("  ... and %d more errors", len(errors) - 10)

    return 0


# ---------------------------------------------------------------------------
# Timing utilities
# ---------------------------------------------------------------------------


def format_timing_summary(timings: dict[str, float]) -> str:
    """Format a timing summary for logging."""
    lines = []
    total = 0.0
    for step, elapsed in timings.items():
        lines.append(f"  {step}: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
        total += elapsed
    lines.append(f"  Total: {total:.1f}s ({total / 60:.1f} min)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Optimized Julius alignment pipeline with timing and unified precompute"
    )
    parser.add_argument("--filelist", type=str, nargs="+", required=True,
                        help="One or more filelist paths (format: wav_path|speaker_id|text)")
    parser.add_argument("--output-dir", type=str, default="data/julius_work",
                        help="Working directory for Julius intermediate files")
    parser.add_argument("--pt-output-dir", type=str, default="data/jvs_precomputed_aligned",
                        help="Output directory for final .pt files")
    parser.add_argument("--mel-mean", type=float, default=-6.550095,
                        help="Mel normalization mean (default: JVS trimmed)")
    parser.add_argument("--mel-std", type=float, default=2.383771,
                        help="Mel normalization std (default: JVS trimmed)")
    parser.add_argument("--num-workers", type=int, default=16,
                        help="Number of parallel workers")

    # Step skip flags
    parser.add_argument("--skip-prepare", action="store_true",
                        help="Skip Step 1 (resample + hiragana)")
    parser.add_argument("--skip-julius", action="store_true",
                        help="Skip Step 2 (Julius alignment)")
    parser.add_argument("--skip-convert", action="store_true",
                        help="Skip Step 3 (duration conversion). "
                             "When using unified precompute (default), Step 3 is "
                             "merged into Step 4 and this flag has no effect.")
    parser.add_argument("--skip-embed", action="store_true",
                        help="Skip Step 4 (.pt file generation)")

    # T2-2: Unified vs legacy precompute
    parser.add_argument("--unified-precompute", action="store_true", default=True,
                        help="Use unified precompute_with_alignment.py (default: True). "
                             "Merges Step 3 (duration conversion) and Step 4 (.pt generation) "
                             "into a single step that reads .lab files directly.")
    parser.add_argument("--legacy-precompute", action="store_true",
                        help="Use legacy separate pipeline: Step 3 (convert_julius_to_durations) "
                             "+ Step 4 (precompute_dataset.py --durations-dir)")

    # T2-3: /dev/shm option
    parser.add_argument("--use-shm", action="store_true",
                        help="Copy final .pt output to /dev/shm for fast I/O during training. "
                             "Files are written to --pt-output-dir first, then copied.")
    parser.add_argument("--shm-dir", type=str, default="/dev/shm/jvs_precomputed_aligned",
                        help="Target directory under /dev/shm (default: /dev/shm/jvs_precomputed_aligned)")

    args = parser.parse_args()

    # Resolve precompute mode: --legacy-precompute overrides --unified-precompute
    use_unified = not args.legacy_precompute

    work_dir = Path(args.output_dir)
    wav_dir = work_dir / "wav"
    dur_dir = work_dir / "durations"
    wav_dir.mkdir(parents=True, exist_ok=True)
    dur_dir.mkdir(parents=True, exist_ok=True)

    # Load all filelists
    all_entries = []
    for fl in args.filelist:
        with open(fl, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split("|")
                    if len(parts) == 3:
                        all_entries.append(parts)
    log.info("Total entries: %d from %d filelists", len(all_entries), len(args.filelist))
    log.info("Precompute mode: %s", "unified" if use_unified else "legacy")

    timings: dict[str, float] = {}

    # ===== Step 0: Build text cache (T1-3) =====
    t0 = time.time()
    log.info("=== Step 0: Building text cache (T1-3) ===")
    all_texts = [entry[2] for entry in all_entries]
    hiragana_cache = build_text_cache(all_texts, args.num_workers)
    # Populate module-level cache for any function that uses text_to_hiragana()
    _HIRAGANA_CACHE.update(hiragana_cache)
    timings["Step 0: Text cache"] = time.time() - t0

    # ===== Step 1: Prepare Julius input =====
    if not args.skip_prepare:
        t0 = time.time()
        log.info("=== Step 1: Preparing Julius input (16kHz + hiragana) ===")
        errors = []
        skipped = 0
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {}
            for entry in all_entries:
                wav_path, spk_str, text = entry
                future = executor.submit(prepare_single, wav_path, text, wav_dir,
                                          hiragana_cache)
                futures[future] = wav_path

            for future in tqdm(as_completed(futures), total=len(futures), desc="Preparing"):
                name, is_error, msg = future.result()
                if is_error:
                    errors.append(f"{name}: {msg}")
                elif "skipped" in msg:
                    skipped += 1

        log.info("Prepared: %d ok, %d skipped, %d errors",
                 len(all_entries) - len(errors) - skipped, skipped, len(errors))
        if errors:
            for e in errors[:5]:
                log.warning("  %s", e)
        timings["Step 1: Prepare"] = time.time() - t0

    # ===== Step 2: Run Julius alignment (T1-1 parallel) =====
    if not args.skip_julius:
        t0 = time.time()
        log.info("=== Step 2: Running Julius forced alignment ===")
        ret = run_julius_alignment(wav_dir, wav_dir, args.num_workers)
        if ret != 0:
            log.error("Julius alignment failed")
            return 1
        timings["Step 2: Julius"] = time.time() - t0

    # ===== Step 3 + 4: Unified or Legacy path =====
    if use_unified:
        # T2-2: Unified precompute — .lab files are read directly by
        # precompute_with_alignment.py, which handles duration conversion
        # and .pt generation in a single pass. Step 3 is skipped entirely.
        if not args.skip_embed:
            t0 = time.time()
            log.info("=== Step 3+4 (unified): Generating .pt files with durations from .lab ===")
            for split, filelist in [("train", args.filelist[0]),
                                     ("val", args.filelist[1] if len(args.filelist) > 1 else None)]:
                if filelist is None:
                    continue
                pt_out = Path(args.pt_output_dir) / split
                pt_out.mkdir(parents=True, exist_ok=True)
                cmd = [
                    sys.executable, "scripts/precompute_with_alignment.py",
                    "--filelist", filelist,
                    "--lab-dir", str(wav_dir),
                    "--output-dir", str(pt_out),
                    "--mel-mean", str(args.mel_mean),
                    "--mel-std", str(args.mel_std),
                    "--num-workers", str(args.num_workers),
                ]
                log.info("Running: %s", " ".join(cmd))
                subprocess.run(cmd, check=True)
            timings["Step 3+4: Unified precompute"] = time.time() - t0
    else:
        # Legacy path: separate Step 3 (duration conversion) and Step 4 (precompute)

        # ===== Step 3: Convert .lab to duration .npy (T1-2 parallel + sf.info) =====
        if not args.skip_convert:
            t0 = time.time()
            log.info("=== Step 3: Converting .lab to duration .npy ===")

            # Build lookup: name -> (text, wav_path)
            entry_map = {}
            for entry in all_entries:
                wav_path, spk_str, text = entry
                wav_p = Path(wav_path)
                spk_name = wav_p.parent.name
                name = f"{spk_name}_{wav_p.stem}"
                entry_map[name] = (text, wav_path)

            # T1-2: Build tasks with mel_frames from sf.info (avoids torch.load)
            tasks = []
            skip_count = 0
            for lab_path in sorted(wav_dir.glob("*.lab")):
                name = lab_path.stem
                if name not in entry_map:
                    skip_count += 1
                    continue
                text, wav_path = entry_map[name]

                # T1-2: Compute mel_frames from wav file length directly
                try:
                    info = sf.info(wav_path)
                    mel_frames = info.frames // HOP_LENGTH
                except Exception:
                    skip_count += 1
                    continue

                npy_path = dur_dir / f"{name}.npy"
                tasks.append((str(lab_path), text, mel_frames, str(npy_path), "auto"))

            # T1-2: Parallel conversion using _worker from convert_julius_to_durations
            from convert_julius_to_durations import _worker as duration_worker

            success = 0
            fail = 0
            with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
                futures = {executor.submit(duration_worker, task): task[0] for task in tasks}
                for future in tqdm(as_completed(futures), total=len(futures), desc="Converting"):
                    try:
                        ok, msg = future.result()
                        if ok:
                            success += 1
                        else:
                            fail += 1
                    except Exception:
                        fail += 1

            log.info("Converted: %d success, %d failed, %d skipped", success, fail, skip_count)
            timings["Step 3: Convert"] = time.time() - t0

        # ===== Step 4: Re-generate .pt with durations (legacy: precompute_dataset.py) =====
        if not args.skip_embed:
            t0 = time.time()
            log.info("=== Step 4: Re-generating .pt files with durations ===")
            for split, filelist in [("train", args.filelist[0]),
                                     ("val", args.filelist[1] if len(args.filelist) > 1 else None)]:
                if filelist is None:
                    continue
                pt_out = Path(args.pt_output_dir) / split
                pt_out.mkdir(parents=True, exist_ok=True)
                cmd = [
                    sys.executable, "scripts/precompute_dataset.py",
                    "--filelist", filelist,
                    "--output-dir", str(pt_out),
                    "--mel-mean", str(args.mel_mean),
                    "--mel-std", str(args.mel_std),
                    "--durations-dir", str(dur_dir),
                    "--num-workers", str(args.num_workers),
                ]
                log.info("Running: %s", " ".join(cmd))
                subprocess.run(cmd, check=True)
            timings["Step 4: Embed"] = time.time() - t0

    # ===== T2-3: Copy to /dev/shm =====
    if args.use_shm and not args.skip_embed:
        t0 = time.time()
        shm_target = Path(args.shm_dir)
        log.info("=== Copying .pt files to /dev/shm: %s ===", shm_target)

        pt_source = Path(args.pt_output_dir)
        if not pt_source.exists():
            log.error("Source directory does not exist: %s", pt_source)
            return 1

        # Check /dev/shm available space
        try:
            shm_stat = shutil.disk_usage("/dev/shm")
            source_size = sum(f.stat().st_size for f in pt_source.rglob("*.pt"))
            log.info(
                "/dev/shm: %.1f GB free, source data: %.1f GB",
                shm_stat.free / (1024**3),
                source_size / (1024**3),
            )
            if source_size > shm_stat.free * 0.9:
                log.warning(
                    "Insufficient /dev/shm space! Need %.1f GB, have %.1f GB free. Skipping shm copy.",
                    source_size / (1024**3),
                    shm_stat.free / (1024**3),
                )
            else:
                # Copy with directory structure preserved
                if shm_target.exists():
                    log.info("Removing existing /dev/shm data at %s", shm_target)
                    shutil.rmtree(str(shm_target))
                shutil.copytree(str(pt_source), str(shm_target))
                copied_count = sum(1 for _ in shm_target.rglob("*.pt"))
                log.info("Copied %d .pt files to %s", copied_count, shm_target)
        except Exception as e:
            log.error("Failed to copy to /dev/shm: %s", e)
            # Non-fatal: training can still use disk-based files

        timings["Copy to /dev/shm"] = time.time() - t0

    # ===== Summary =====
    log.info("=== Pipeline complete ===")
    log.info("Timing summary:")
    total = 0.0
    for step, elapsed in timings.items():
        log.info("  %s: %.1fs (%.1f min)", step, elapsed, elapsed / 60)
        total += elapsed
    log.info("  Total: %.1fs (%.1f min)", total, total / 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
