"""Full Julius forced alignment pipeline: prepare → align → convert → embed.

Runs the complete pipeline from JVS filelists to duration-embedded .pt files:
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
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
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


def prepare_single(wav_path: str, text: str, output_dir: Path) -> tuple[str, bool, str]:
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
        hiragana = text_to_hiragana(text)
        out_txt.write_text(hiragana, encoding="utf-8")
        return name, False, "ok"
    except Exception as e:
        return name, True, str(e)


def run_julius_alignment(wav_dir: Path) -> int:
    """Run Julius segmentation-kit on prepared wav directory."""
    segkit = SEGKIT_DIR.resolve()
    script = segkit / "segment_julius.pl"
    if not script.exists():
        log.error("segment_julius.pl not found at %s", script)
        return 1

    # segmentation-kit expects files in its own wav/ directory
    # Create symlinks from our wav_dir to segkit/wav/
    kit_wav_dir = segkit / "wav"

    # Backup existing sample files
    existing = list(kit_wav_dir.glob("*"))
    backup_dir = segkit / "wav_backup"
    if existing and not backup_dir.exists():
        backup_dir.mkdir()
        for f in existing:
            f.rename(backup_dir / f.name)

    # Clear and symlink our files
    for f in kit_wav_dir.glob("*"):
        f.unlink()

    for f in wav_dir.glob("*.wav"):
        (kit_wav_dir / f.name).symlink_to(f.resolve())
    for f in wav_dir.glob("*.txt"):
        (kit_wav_dir / f.name).symlink_to(f.resolve())

    n_files = len(list(kit_wav_dir.glob("*.wav")))
    log.info("Running Julius alignment on %d files...", n_files)

    # Run segment_julius.pl
    result = subprocess.run(
        ["perl", str(script)],
        cwd=str(segkit),
        capture_output=True,
        text=True,
        timeout=3600 * 12,  # 12 hour timeout
    )

    if result.returncode != 0:
        log.error("Julius alignment failed: %s", result.stderr[:500])
        return 1

    # Collect .lab files back to our directory
    lab_count = 0
    for lab in kit_wav_dir.glob("*.lab"):
        target = wav_dir / lab.name
        if lab.is_symlink():
            continue
        lab.rename(target)
        lab_count += 1

    log.info("Julius alignment complete: %d .lab files generated", lab_count)

    # Restore backup
    if backup_dir.exists():
        for f in kit_wav_dir.glob("*"):
            if not f.is_symlink():
                continue
            f.unlink()
        for f in backup_dir.glob("*"):
            f.rename(kit_wav_dir / f.name)
        backup_dir.rmdir()

    return 0


def lab_to_duration_npy(
    lab_path: Path,
    text: str,
    mel_frames: int,
    output_path: Path,
) -> tuple[bool, str]:
    """Convert .lab to duration .npy for one utterance.

    Delegates to the robust alignment logic in convert_julius_to_durations.py,
    which provides:
    - Proper HTK 100ns time-unit handling via parse_lab_file / time_to_frames
    - map_julius_sequence for phoneme mapping (handles q->cl, silB->sil, etc.)
    - Sequential alignment with look-ahead + automatic DTW fallback (>5% mismatch)
    - Correct handling of prosody symbols, devoiced vowels, and sil/pau
    - Frame-total adjustment to match mel_frames exactly
    """
    from convert_julius_to_durations import process_single_utterance

    output_path.parent.mkdir(parents=True, exist_ok=True)
    return process_single_utterance(lab_path, text, mel_frames, output_path)


def main():
    parser = argparse.ArgumentParser(description="Full Julius alignment pipeline")
    parser.add_argument("--filelist", type=str, nargs="+", required=True)
    parser.add_argument("--output-dir", type=str, default="data/julius_work")
    parser.add_argument("--pt-output-dir", type=str, default="data/jvs_precomputed_aligned")
    parser.add_argument("--mel-mean", type=float, default=-6.550095)
    parser.add_argument("--mel-std", type=float, default=2.383771)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-julius", action="store_true")
    parser.add_argument("--skip-convert", action="store_true")
    parser.add_argument("--skip-embed", action="store_true")
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

    # ===== Step 1: Prepare Julius input =====
    if not args.skip_prepare:
        log.info("=== Step 1: Preparing Julius input (16kHz + hiragana) ===")
        errors = []
        skipped = 0
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {}
            for entry in all_entries:
                wav_path, spk_str, text = entry
                future = executor.submit(prepare_single, wav_path, text, wav_dir)
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

    # ===== Step 2: Run Julius alignment =====
    if not args.skip_julius:
        log.info("=== Step 2: Running Julius forced alignment ===")
        ret = run_julius_alignment(wav_dir)
        if ret != 0:
            log.error("Julius alignment failed")
            return 1

    # ===== Step 3: Convert .lab to duration .npy =====
    if not args.skip_convert:
        log.info("=== Step 3: Converting .lab to duration .npy ===")
        success = 0
        fail = 0
        # Build lookup: name -> (text, mel_frames)
        entry_map = {}
        for entry in all_entries:
            wav_path, spk_str, text = entry
            wav_p = Path(wav_path)
            spk_name = wav_p.parent.name
            name = f"{spk_name}_{wav_p.stem}"
            entry_map[name] = text

        # Find precomputed .pt files for mel_frames
        pt_dirs = [Path("/dev/shm/jvs_precomputed/train"), Path("/dev/shm/jvs_precomputed/val"),
                    Path("data/jvs_precomputed/train"), Path("data/jvs_precomputed/val")]

        pt_map = {}
        for pt_dir in pt_dirs:
            if pt_dir.exists():
                for pt_path in pt_dir.glob("*.pt"):
                    pt_map[pt_path.stem] = pt_path

        for lab_path in tqdm(sorted(wav_dir.glob("*.lab")), desc="Converting"):
            name = lab_path.stem
            if name not in entry_map:
                fail += 1
                continue
            text = entry_map[name]

            # Get mel frames from .pt file
            if name in pt_map:
                pt_data = torch.load(pt_map[name], weights_only=True)
                mel_frames = pt_data["mel"].shape[-1]
            else:
                fail += 1
                continue

            npy_path = dur_dir / f"{name}.npy"
            ok, msg = lab_to_duration_npy(lab_path, text, mel_frames, npy_path)
            if ok:
                success += 1
            else:
                fail += 1

        log.info("Converted: %d success, %d failed", success, fail)

    # ===== Step 4: Re-generate .pt with durations =====
    if not args.skip_embed:
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

    log.info("=== Pipeline complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
