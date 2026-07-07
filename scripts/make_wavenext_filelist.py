"""Build a WaveNeXt training filelist (one absolute wav path per line) from a directory.

Usage:
    uv run python scripts/make_wavenext_filelist.py --wav-dir data/moespeech/wavs --out data/moespeech/wavenext_train.txt
"""

import argparse
from pathlib import Path


def make_filelist(wav_dir, recursive=True):
    root = Path(wav_dir)
    it = root.rglob("*.wav") if recursive else root.glob("*.wav")
    return sorted(str(p.resolve()) for p in it)


def main(argv=None):
    p = argparse.ArgumentParser(description="wav directory -> WaveNeXt training filelist")
    p.add_argument("--wav-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--no-recursive", action="store_true")
    args = p.parse_args(argv)
    lines = make_filelist(args.wav_dir, recursive=not args.no_recursive)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[+] wrote {args.out}: {len(lines)} wav paths")


if __name__ == "__main__":
    main()
