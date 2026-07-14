"""Transfer a multi-speaker MoeSpeech base checkpoint to a Tsukuyomi single-speaker init.

Design (adversarially verified, D1/D2):
- We do NOT shrink to n_spks=1 (that changes spk_emb/FiLM/proj shapes -> strict-load fails).
  Instead we KEEP the multi-speaker architecture and RESIZE ``spk_emb.weight`` from
  ``(n_spks, 64)`` to ``(n_spks+1, 64)``; the new last row is the Tsukuyomi slot (id=n_spks).
  Every other tensor stays byte-identical, so ``load_state_dict(strict=True)`` succeeds.
- We save WEIGHTS-ONLY (``{state_dict[, hyper_parameters]}``): stripping ``optimizer_states``
  makes matcha/train.py take the weights-only branch (strict load -> fresh optimizer/epoch 0),
  i.e. a real fine-tune rather than a full-resume of the 2500-epoch base.

At fine-tune / inference time set the speaker id to the new slot (``spks=n_spks``) and the
data config ``n_spks`` to ``n_spks+1``.
"""

import argparse
from pathlib import Path

import torch


def main(argv=None):
    p = argparse.ArgumentParser(description="MoeSpeech multi-speaker ckpt -> Tsukuyomi single-speaker init")
    p.add_argument("--source", required=True, help="MoeSpeech base checkpoint (last.ckpt)")
    p.add_argument("--target", required=True, help="output weights-only init checkpoint")
    p.add_argument(
        "--init-from-id",
        type=int,
        default=None,
        help="initialize the new slot from an existing speaker id (default: mean of all speakers)",
    )
    p.add_argument(
        "--use-current-model-state",
        action="store_true",
        help="transfer raw weights (current_model_state) instead of the default EMA weights (state_dict)",
    )
    args = p.parse_args(argv)

    ckpt = torch.load(args.source, map_location="cpu", weights_only=False)
    key = "current_model_state" if args.use_current_model_state else "state_dict"
    if key not in ckpt:
        raise KeyError(f"'{key}' not in checkpoint; available top-level keys: {list(ckpt)[:20]}")
    sd = dict(ckpt[key])

    emb_key = "spk_emb.weight"
    if emb_key not in sd:
        raise KeyError(f"'{emb_key}' missing -> base is single-speaker (n_spks<=1). A multi-speaker base is required.")
    old = sd[emb_key]  # (n_spks, spk_emb_dim)
    n_old, dim = old.shape

    # New slot row. Avoid torch.empty (CLAUDE.md: uninitialized memory -> NaN); build via torch.cat.
    if args.init_from_id is not None:
        if not (0 <= args.init_from_id < n_old):
            raise ValueError(f"--init-from-id {args.init_from_id} out of range [0, {n_old})")
        new_row = old[args.init_from_id : args.init_from_id + 1].clone()
        how = f"copied from speaker id {args.init_from_id}"
    else:
        new_row = old.mean(dim=0, keepdim=True)
        how = "mean of all existing speakers"
    sd[emb_key] = torch.cat([old, new_row], dim=0)  # (n_old+1, dim)
    print(f"[+] {emb_key}: {tuple(old.shape)} -> {tuple(sd[emb_key].shape)}; new slot id={n_old} ({how})")

    # D4 guard: surface mel stats so they get transcribed into precompute + data config,
    # and refuse a base whose stats are still the placeholder (fmax=11025 stats not computed).
    for stat in ("mel_mean", "mel_std"):
        if stat in sd:
            v = float(sd[stat])
            print(f"[i] base {stat} = {v:.6f}  (put this in tsukuyomi precompute --{stat.replace('_', '-')} + config)")
            if stat == "mel_std" and abs(v - 1.0) < 1e-9:
                raise ValueError("mel_std==1.0 looks like the placeholder — base must be trained with fmax=11025 stats")

    hp = ckpt.get("hyper_parameters", {})
    if "n_spks" in hp:
        hp["n_spks"] = n_old + 1

    out = {"state_dict": sd}  # drop optimizer_states / averaging_state / current_model_state
    if hp:
        out["hyper_parameters"] = hp
    Path(args.target).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.target)
    print(f"[+] wrote {args.target}: n_spks={n_old + 1}; set data.n_spks={n_old + 1}, infer spks={n_old}")


if __name__ == "__main__":
    main()
