"""Create a Japanese initial checkpoint from an English Matcha-TTS checkpoint.

Replaces the text encoder embedding layer (178-vocab English → 52-vocab Japanese)
while preserving all other weights for transfer learning.

Usage:
    uv run python scripts/transfer_from_english.py \
        --source matcha_ljspeech.ckpt \
        --target matcha_jsut_init.ckpt \
        --n-vocab-new 52
"""

import argparse
from pathlib import Path

import torch
from torch import nn


def main():
    parser = argparse.ArgumentParser(
        description="Create a Japanese initial checkpoint from an English Matcha-TTS model"
    )
    parser.add_argument("--source", type=str, required=True, help="Path to English checkpoint (.ckpt)")
    parser.add_argument("--target", type=str, required=True, help="Output path for Japanese initial checkpoint")
    parser.add_argument("--n-vocab-new", type=int, default=52, help="New vocabulary size (default: 52 for Japanese)")
    parser.add_argument(
        "--n-channels",
        type=int,
        default=None,
        help="Embedding dimension. Auto-detected from checkpoint if omitted.",
    )
    args = parser.parse_args()

    source_path = Path(args.source)
    target_path = Path(args.target)

    if not source_path.exists():
        raise FileNotFoundError(f"Source checkpoint not found: {source_path}")

    print(f"[*] Loading source checkpoint: {source_path}")
    checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)

    state_dict = checkpoint["state_dict"]

    # Find and replace the text encoder embedding
    emb_key = "encoder.emb.weight"
    if emb_key not in state_dict:
        raise KeyError(
            f"'{emb_key}' not found in checkpoint. "
            f"Available keys containing 'emb': {[k for k in state_dict if 'emb' in k]}"
        )

    old_emb = state_dict[emb_key]
    n_channels = old_emb.shape[1]
    print(f"[*] Original embedding: {old_emb.shape} (vocab={old_emb.shape[0]}, dim={n_channels})")

    # Validate --n-channels against the checkpoint if the user provided it
    if args.n_channels is not None and args.n_channels != n_channels:
        raise ValueError(
            f"--n-channels ({args.n_channels}) does not match checkpoint "
            f"embedding dimension ({n_channels}). "
            f"Remove --n-channels to auto-detect from checkpoint."
        )

    # Create new embedding with Xavier uniform initialization
    new_emb = nn.Embedding(args.n_vocab_new, n_channels)
    nn.init.xavier_uniform_(new_emb.weight)

    state_dict[emb_key] = new_emb.weight.data
    print(f"[+] New embedding: {new_emb.weight.shape} (vocab={args.n_vocab_new}, dim={n_channels})")

    # Update hyperparameters if present
    if "hyper_parameters" in checkpoint and "n_vocab" in checkpoint["hyper_parameters"]:
        old_vocab = checkpoint["hyper_parameters"]["n_vocab"]
        checkpoint["hyper_parameters"]["n_vocab"] = args.n_vocab_new
        print(f"[+] Updated hyper_parameters.n_vocab: {old_vocab} -> {args.n_vocab_new}")

    # Save
    target_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, target_path)
    print(f"[+] Saved Japanese initial checkpoint: {target_path}")
    print(f"[*] Removed keys: (none — only replaced {emb_key})")
    print("[*] To train: uv run python matcha/train.py experiment=jsut ckpt_path=<path>")


if __name__ == "__main__":
    main()
