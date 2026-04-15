"""Full Julius forced alignment pipeline: prepare → align → convert → embed.

Runs the complete pipeline from JVS filelists to duration-embedded .pt files:
0. Pre-compute text cache (hiragana + phoneme IDs) for unique texts
1. Resample audio to 16kHz + convert text to hiragana
2. Run Julius segmentation-kit for forced alignment
3. Convert .lab files to duration .npy arrays
4. Re-generate .pt files with durations embedded

Usage:
    uv run python scripts/run_full_alignment_pipeline.py \
        --filelist data/jvs/train.txt data/jvs/val.txt \
        --output-dir data/julius_work \
        --pt-output-dir data/jvs_precomputed_aligned \
        --mel-mean -6.550095 --mel-std 2.383771 \
        --num-workers 8
"""

import argparse
import logging
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from math import gcd
from pathlib import Path

import soundfile as sf
from scipy.signal import resample_poly
from tqdm import tqdm

# Allow importing sibling scripts (convert_julius_to_durations)
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


def text_to_hiragana(text: str) -> str:
    """Convert Japanese text to hiragana via pyopenjtalk."""
    import pyopenjtalk

    kana = pyopenjtalk.g2p(text, kana=True)  # Returns katakana
    kana = _PUNCT_RE.sub("", kana)
    return katakana_to_hiragana(kana)


def resample_wav(input_path: str, output_path: str):
    """Resample wav to 16kHz 16bit mono."""
    data, sr = sf.read(input_path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != JULIUS_SR:
        g = gcd(sr, JULIUS_SR)
        data = resample_poly(data, JULIUS_SR // g, sr // g)
    sf.write(output_path, data, JULIUS_SR, subtype="PCM_16")


def prepare_single(
    wav_path: str, text: str, output_dir: Path, hiragana: str = None
) -> tuple[str, bool, str]:
    """Prepare one file: resample + hiragana text."""
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
        if hiragana is None:
            hiragana = text_to_hiragana(text)
        out_txt.write_text(hiragana, encoding="utf-8")
        return name, False, "ok"
    except Exception as e:
        return name, True, str(e)


def run_julius_alignment(wav_dir: Path, output_dir: Path, num_workers: int = 16) -> int:
    """Run Julius forced alignment with parallel workers.

    Uses run_julius_alignment.py's _align_single() + ProcessPoolExecutor to
    run one Julius process per file across multiple cores.

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
        len(matched),
        len(already_done),
        len(todo),
        num_workers,
    )

    if not todo:
        log.info("Nothing to do. All files already aligned.")
        return 0

    # Build task tuples for _align_single(name, wav_path, txt_path, segkit_dir, output_dir, timeout)
    tasks = [
        (
            name,
            str(wav_files[name]),
            str(txt_files[name]),
            str(segkit_dir),
            str(output_dir),
            300,
        )
        for name in todo
    ]

    success = 0
    errors = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_align_single, task): task[0] for task in tasks}
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Julius alignment",
            unit="files",
        ):
            name = futures[future]
            try:
                result_name, error = future.result()
                if error:
                    errors.append(f"{result_name}: {error}")
                else:
                    success += 1
            except Exception as e:
                errors.append(f"{name}: {e}")

    log.info(
        "Julius alignment: %d success, %d errors out of %d",
        success,
        len(errors),
        len(todo),
    )
    if errors:
        for e in errors[:10]:
            log.warning("  %s", e)
        if len(errors) > 10:
            log.warning("  ... and %d more errors", len(errors) - 10)

    return 0


def main():
    parser = argparse.ArgumentParser(description="Full Julius alignment pipeline")
    parser.add_argument("--filelist", type=str, nargs="+", required=True)
    parser.add_argument("--output-dir", type=str, default="data/julius_work")
    parser.add_argument("--pt-output-dir", type=str, default="data/jvs_precomputed_aligned")
    parser.add_argument("--mel-mean", type=float, default=-6.550095)
    parser.add_argument("--mel-std", type=float, default=2.383771)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-julius", action="store_true")
    parser.add_argument("--skip-convert", action="store_true")
    parser.add_argument("--skip-embed", action="store_true")
    parser.add_argument(
        "--force-text-cache",
        action="store_true",
        help="Force rebuild text cache even if exists",
    )
    args = parser.parse_args()

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

    # ===== Step 0: Pre-compute text cache =====
    log.info("=== Step 0: Pre-computing text cache ===")
    text_cache_path = work_dir / "text_cache.pkl"

    if text_cache_path.exists() and not args.force_text_cache:
        import pickle

        with open(text_cache_path, "rb") as f:
            text_cache = pickle.load(f)
        log.info("Loaded text cache: %d entries", len(text_cache))
    else:
        import pickle

        from matcha.text import text_to_sequence
        from matcha.utils.utils import intersperse

        unique_texts = set(text for _, _, text in all_entries)
        log.info(
            "Pre-computing text for %d unique texts (%d total entries)",
            len(unique_texts),
            len(all_entries),
        )

        text_cache = {}
        for text in tqdm(unique_texts, desc="Text cache"):
            # Hiragana conversion (for Julius input)
            hiragana = text_to_hiragana(text)
            # Phoneme sequence conversion (for .pt generation)
            text_norm, cleaned_text = text_to_sequence(
                text, ["japanese_cleaners"], language="ja"
            )
            text_norm_interspersed = intersperse(text_norm, 0)
            text_cache[text] = {
                "hiragana": hiragana,
                "text_norm": text_norm_interspersed,
                "cleaned_text": cleaned_text,
            }

        with open(text_cache_path, "wb") as f:
            pickle.dump(text_cache, f)
        log.info(
            "Saved text cache: %d entries to %s", len(text_cache), text_cache_path
        )

    # ===== Step 1: Prepare Julius input =====
    if not args.skip_prepare:
        log.info("=== Step 1: Preparing Julius input (16kHz + hiragana) ===")
        errors = []
        skipped = 0
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {}
            for entry in all_entries:
                wav_path, spk_str, text = entry
                hiragana = (
                    text_cache[text]["hiragana"] if text in text_cache else None
                )
                future = executor.submit(
                    prepare_single, wav_path, text, wav_dir, hiragana
                )
                futures[future] = wav_path

            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Preparing"
            ):
                name, is_error, msg = future.result()
                if is_error:
                    errors.append(f"{name}: {msg}")
                elif "skipped" in msg:
                    skipped += 1

        log.info(
            "Prepared: %d ok, %d skipped, %d errors",
            len(all_entries) - len(errors) - skipped,
            skipped,
            len(errors),
        )
        if errors:
            for e in errors[:5]:
                log.warning("  %s", e)

    # ===== Step 2: Run Julius alignment =====
    if not args.skip_julius:
        log.info("=== Step 2: Running Julius forced alignment ===")
        ret = run_julius_alignment(wav_dir, wav_dir, args.num_workers)
        if ret != 0:
            log.error("Julius alignment failed")
            return 1

    # ===== Step 3: Convert .lab to duration .npy =====
    if not args.skip_convert:
        log.info("=== Step 3: Converting .lab to duration .npy ===")

        # Build lookup: name -> (text, wav_path)
        entry_map = {}
        for entry in all_entries:
            wav_path, spk_str, text = entry
            wav_p = Path(wav_path)
            spk_name = wav_p.parent.name
            name = f"{spk_name}_{wav_p.stem}"
            entry_map[name] = (text, wav_path)

        # Build tasks with mel_frames from sf.info (avoids torch.load)
        tasks = []
        skip_count = 0
        for lab_path in sorted(wav_dir.glob("*.lab")):
            name = lab_path.stem
            if name not in entry_map:
                skip_count += 1
                continue
            text, wav_path = entry_map[name]

            # Compute mel_frames from wav file length directly
            try:
                info = sf.info(wav_path)
                mel_frames = info.frames // HOP_LENGTH
            except Exception:
                skip_count += 1
                continue

            npy_path = dur_dir / f"{name}.npy"
            tasks.append((str(lab_path), text, mel_frames, str(npy_path), "auto"))

        # Parallel conversion using _worker from convert_julius_to_durations
        from convert_julius_to_durations import _worker as duration_worker

        success = 0
        fail = 0
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {
                executor.submit(duration_worker, task): task[0] for task in tasks
            }
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Converting"
            ):
                try:
                    ok, msg = future.result()
                    if ok:
                        success += 1
                    else:
                        fail += 1
                except Exception:
                    fail += 1

        log.info(
            "Converted: %d success, %d failed, %d skipped", success, fail, skip_count
        )

    # ===== Step 4: Re-generate .pt with durations =====
    if not args.skip_embed:
        log.info("=== Step 4: Re-generating .pt files with durations ===")
        for split, filelist in [
            ("train", args.filelist[0]),
            ("val", args.filelist[1] if len(args.filelist) > 1 else None),
        ]:
            if filelist is None:
                continue
            pt_out = Path(args.pt_output_dir) / split
            pt_out.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                "scripts/precompute_dataset.py",
                "--filelist",
                filelist,
                "--output-dir",
                str(pt_out),
                "--mel-mean",
                str(args.mel_mean),
                "--mel-std",
                str(args.mel_std),
                "--durations-dir",
                str(dur_dir),
                "--num-workers",
                str(args.num_workers),
            ]
            log.info("Running: %s", " ".join(cmd))
            subprocess.run(cmd, check=True)

    log.info("=== Pipeline complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
