"""Prepare the Tsukuyomi-chan corpus (fine-tune target) into precompute filelists.

Source: ayousanz/tsukuyomi-chan-ljspeech (LJSpeech format, already 22050 Hz mono, clean
human transcripts). metadata.csv is ``VOICEACTRESS100_NNN|<text>`` (pipe-delimited).

Emits ``train.txt`` / ``val.txt`` in precompute_dataset.py's ``wav_path|speaker_id|text``
format, where speaker_id is the Tsukuyomi slot appended by transfer_speaker_embedding.py
(jvs_aligned base has n_spks=100, so the new slot is 100). Feed these to:
    scripts/precompute_dataset.py --filelist ... \
        --mel-mean -6.550095 --mel-std 2.383771
using the jvs_aligned fmax=8000 mel stats (default --fmax 8000), because strict-load keeps the
base model's mel_mean/std buffers — precompute must normalize with the same values.
"""

import argparse
import random
from pathlib import Path


def load_meta(meta_path):
    """Parse ``name|text`` rows (robust to extra pipes in text)."""
    rows = []
    with open(meta_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("|")
            if len(parts) < 2:
                continue
            name, text = parts[0].strip(), "|".join(parts[1:]).strip()
            if name and text:
                rows.append((name, text))
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description="Tsukuyomi-chan corpus -> precompute filelists")
    p.add_argument("--meta", default=None, help="metadata.csv path (default: download from HF)")
    p.add_argument("--wavs-dir", default=None, help="wavs directory (default: download from HF)")
    p.add_argument("--out-dir", default="data/tsukuyomi")
    p.add_argument("--slot-id", type=int, default=100, help="speaker slot id (= base model n_spks; jvs_aligned=100)")
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    meta_path, wavs_dir = args.meta, args.wavs_dir
    if meta_path is None or wavs_dir is None:
        from huggingface_hub import snapshot_download

        root = snapshot_download("ayousanz/tsukuyomi-chan-ljspeech", repo_type="dataset")
        meta_path = meta_path or str(Path(root) / "metadata.csv")
        wavs_dir = wavs_dir or str(Path(root) / "wavs")
        print(f"[tsukuyomi] downloaded -> {root}")

    rows = load_meta(meta_path)
    wavs = Path(wavs_dir)
    lines, missing = [], 0
    for name, text in rows:
        wav = wavs / f"{name}.wav"
        if not wav.exists():
            missing += 1
            continue
        lines.append(f"{wav}|{args.slot_id}|{text}")
    if missing:
        print(f"[tsukuyomi] WARNING {missing} wav(s) referenced by metadata not found in {wavs_dir}")

    rng = random.Random(args.seed)
    rng.shuffle(lines)
    n_val = max(1, int(len(lines) * args.val_ratio))
    splits = {"val": lines[:n_val], "train": lines[n_val:]}

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for split, split_lines in splits.items():
        path = out / f"{split}.txt"
        path.write_text("\n".join(split_lines) + "\n", encoding="utf-8")
        print(f"[tsukuyomi] {split}: {len(split_lines)} -> {path}")
    print(f"[tsukuyomi] slot_id={args.slot_id}. Next: precompute_dataset.py --fmax 11025 "
          f"--mel-mean <BASE_MOE_MEAN> --mel-std <BASE_MOE_STD>")


if __name__ == "__main__":
    main()
