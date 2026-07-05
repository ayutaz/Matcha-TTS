"""Run Julius segmentation-kit to produce forced alignment .lab files.

Takes the output of prepare_julius_input.py (wav/ and txt/ directories) and
runs the Julius segmentation-kit on each utterance, producing HTK-format .lab
files with phoneme-level time boundaries.

Usage:
    uv run python scripts/run_julius_alignment.py \
        --input-dir data/julius_input \
        --segkit-dir tools/segmentation-kit \
        --output-dir data/julius_alignment \
        --num-workers 4

Output:
    data/julius_alignment/
        jvs001_BASIC5000_0025.lab
        ...

.lab format (HTK, 100ns units):
    0 2100000 silB
    2100000 3200000 k
    3200000 4500000 o
    ...
    14100000 16000000 silE
"""

import argparse
import os
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm


def check_prerequisites(segkit_dir):
    """Verify that Julius and segmentation-kit are available.

    Args:
        segkit_dir: Path to the segmentation-kit directory.

    Raises:
        FileNotFoundError: If segmentation-kit or julius is not found.
    """
    segkit_path = Path(segkit_dir)
    if not segkit_path.is_dir():
        raise FileNotFoundError(f"segmentation-kit not found at {segkit_dir}. Run: bash scripts/setup_julius.sh")

    segment_script = segkit_path / "segment_julius.pl"
    if not segment_script.exists():
        raise FileNotFoundError(f"segment_julius.pl not found in {segkit_dir}. The segmentation-kit may be incomplete.")

    if not shutil.which("julius"):
        raise FileNotFoundError(
            "julius command not found in PATH. Install via:\n"
            "  Ubuntu/Debian: sudo apt-get install julius\n"
            "  From source:   https://github.com/julius-speech/julius"
        )

    if not shutil.which("perl"):
        raise FileNotFoundError(
            "perl command not found in PATH. Install via:\n  Ubuntu/Debian: sudo apt-get install perl"
        )


def run_segkit_batch(wav_files, txt_files, segkit_dir, output_dir, timeout=300):
    """Run segmentation-kit on a batch of files.

    segment_julius.pl expects .wav and .txt files in the same directory
    (default ``./wav``), plus ``bin/`` and ``models/`` from the segkit.
    We create a temporary directory with symlinks to satisfy this layout.

    Args:
        wav_files: List of (name, wav_path) tuples.
        txt_files: List of (name, txt_path) tuples.
        segkit_dir: Path to segmentation-kit.
        output_dir: Where to write .lab files.
        timeout: Timeout in seconds for the subprocess.

    Returns:
        (success_list, error_list) where each is a list of (name, message).
    """
    segkit_path = Path(segkit_dir).resolve()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    successes = []
    errors = []

    with tempfile.TemporaryDirectory(prefix="julius_align_") as tmpdir:
        tmp = Path(tmpdir)
        wav_dir = tmp / "wav"
        wav_dir.mkdir()

        # Symlink bin/ and models/ from segmentation-kit so that
        # segment_julius.pl can find julius binary and acoustic models
        (tmp / "bin").symlink_to(segkit_path / "bin")
        (tmp / "models").symlink_to(segkit_path / "models")

        # Symlink .wav and .txt into the same wav/ directory
        # (segment_julius.pl reads "$datadir/$basename.txt" alongside .wav)
        name_set = set()
        for name, wav_path in wav_files:
            (wav_dir / f"{name}.wav").symlink_to(Path(wav_path).resolve())
            name_set.add(name)
        for name, txt_path in txt_files:
            (wav_dir / f"{name}.txt").symlink_to(Path(txt_path).resolve())

        # Run segment_julius.pl from the temp directory.
        # PATH: segkit bin/ first, then julius's directory (only when resolvable),
        # then the standard Linux locations.
        julius_exe = shutil.which("julius")
        path_dirs = [str(segkit_path / "bin")]
        if julius_exe:
            path_dirs.append(str(Path(julius_exe).parent))
        path_dirs += ["/usr/bin", "/bin", "/usr/local/bin"]
        try:
            result = subprocess.run(
                ["perl", str(segkit_path / "segment_julius.pl")],
                cwd=str(tmp),
                capture_output=True,
                text=True,
                timeout=timeout,
                env={
                    "PATH": os.pathsep.join(path_dirs),
                    "HOME": str(Path.home()),
                },
                check=False,
            )
        except subprocess.TimeoutExpired:
            for name in name_set:
                errors.append((name, "Timeout expired"))
            return successes, errors
        except Exception as e:
            for name in name_set:
                errors.append((name, f"subprocess error: {e}"))
            return successes, errors

        # Collect output .lab files
        # segmentation-kit writes .lab files alongside the wav files or in a
        # lab/ directory depending on version
        lab_dirs = [wav_dir, tmp / "lab", tmp]
        for name in name_set:
            found = False
            for lab_dir in lab_dirs:
                lab_file = lab_dir / f"{name}.lab"
                if lab_file.exists():
                    # Copy to output directory
                    dest = output_path / f"{name}.lab"
                    shutil.copy2(str(lab_file), str(dest))
                    successes.append((name, str(dest)))
                    found = True
                    break
            if not found:
                stderr_snippet = result.stderr[-500:] if result.stderr else "(no stderr)"
                errors.append((name, f"No .lab file produced. stderr: {stderr_snippet}"))

    return successes, errors


def _align_single(args_tuple):
    """Worker for single-file alignment.

    Falls back to running segmentation-kit per file when batch mode is not
    suitable (e.g., to isolate failures).

    Args:
        args_tuple: (name, wav_path, txt_path, segkit_dir, output_dir, timeout)

    Returns:
        (name, error_or_None)
    """
    name, wav_path, txt_path, segkit_dir, output_dir, timeout = args_tuple
    try:
        successes, errors = run_segkit_batch(
            [(name, wav_path)],
            [(name, txt_path)],
            segkit_dir,
            output_dir,
            timeout=timeout,
        )
        if errors:
            return name, errors[0][1]
        return name, None
    except Exception as e:
        return name, str(e)


def main():
    parser = argparse.ArgumentParser(description="Run Julius segmentation-kit for forced alignment.")
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory with wav/ and txt/ subdirectories (from prepare_julius_input.py)",
    )
    parser.add_argument(
        "--segkit-dir",
        type=str,
        default="tools/segmentation-kit",
        help="Path to segmentation-kit (default: tools/segmentation-kit)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write .lab files",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of files per segmentation-kit invocation (default: 1)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout per batch in seconds (default: 300)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    wav_dir = input_dir / "wav"
    txt_dir = input_dir / "txt"

    if not wav_dir.is_dir():
        raise FileNotFoundError(f"wav/ directory not found in {input_dir}")
    if not txt_dir.is_dir():
        raise FileNotFoundError(f"txt/ directory not found in {input_dir}")

    # Check prerequisites
    check_prerequisites(args.segkit_dir)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover files (matched pairs of .wav and .txt)
    wav_files = {p.stem: p for p in sorted(wav_dir.glob("*.wav"))}
    txt_files = {p.stem: p for p in sorted(txt_dir.glob("*.txt"))}

    matched_names = sorted(set(wav_files.keys()) & set(txt_files.keys()))
    wav_only = set(wav_files.keys()) - set(txt_files.keys())
    txt_only = set(txt_files.keys()) - set(wav_files.keys())

    print(f"Input directory: {input_dir}")
    print(f"Matched pairs: {len(matched_names)}")
    if wav_only:
        print(f"WARNING: {len(wav_only)} wav files without matching txt")
    if txt_only:
        print(f"WARNING: {len(txt_only)} txt files without matching wav")

    # Skip already-aligned files
    already_done = {p.stem for p in output_dir.glob("*.lab")}
    todo_names = [n for n in matched_names if n not in already_done]
    if already_done:
        print(f"Already aligned: {len(already_done)}, remaining: {len(todo_names)}")

    if not todo_names:
        print("Nothing to do. All files already aligned.")
        return

    # Process files
    start_time = time.time()
    all_errors = []
    all_successes = []

    if args.batch_size == 1:
        # Single-file mode with parallel workers
        tasks = [
            (name, str(wav_files[name]), str(txt_files[name]), args.segkit_dir, str(output_dir), args.timeout)
            for name in todo_names
        ]

        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {executor.submit(_align_single, task): task[0] for task in tasks}
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Aligning",
                unit="files",
            ):
                name = futures[future]
                try:
                    result_name, error = future.result()
                    if error:
                        all_errors.append((result_name, error))
                        tqdm.write(f"ERROR [{result_name}]: {error}")
                    else:
                        all_successes.append(result_name)
                except Exception as e:
                    all_errors.append((name, str(e)))
                    tqdm.write(f"ERROR [{name}]: {e}")
    else:
        # Batch mode: group files and process sequentially
        for i in tqdm(range(0, len(todo_names), args.batch_size), desc="Aligning batches", unit="batch"):
            batch_names = todo_names[i : i + args.batch_size]
            batch_wav = [(n, str(wav_files[n])) for n in batch_names]
            batch_txt = [(n, str(txt_files[n])) for n in batch_names]
            successes, errors = run_segkit_batch(
                batch_wav,
                batch_txt,
                args.segkit_dir,
                str(output_dir),
                timeout=args.timeout,
            )
            all_successes.extend([s[0] for s in successes])
            all_errors.extend(errors)
            for name, msg in errors:
                tqdm.write(f"ERROR [{name}]: {msg}")

    elapsed = time.time() - start_time

    # Summary
    print("\nAlignment complete:")
    print(f"  Success: {len(all_successes)}")
    print(f"  Errors:  {len(all_errors)}")
    print(f"  Skipped: {len(already_done)} (already done)")
    print(f"  Total:   {len(matched_names)}")
    print(f"  Time:    {elapsed:.1f}s")

    if all_errors:
        print(f"\nError details ({min(len(all_errors), 20)} shown):")
        for name, msg in all_errors[:20]:
            print(f"  {name}: {msg}")
        if len(all_errors) > 20:
            print(f"  ... and {len(all_errors) - 20} more")


if __name__ == "__main__":
    main()
