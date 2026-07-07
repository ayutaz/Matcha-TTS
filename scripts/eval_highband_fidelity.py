"""High-band spectral fidelity: does the fmax=11025 vocoder reproduce 8-11 kHz better
than the fmax=8000 one? UTMOS cannot judge this (it downsamples to 16 kHz), so we compare
the resynthesized wavs against the GROUND-TRUTH wav directly in the 8-11 kHz band.

For each held-out GT wav we already have two resyntheses (fmax11025_trained / fmax8000_bsc,
same file index). We compute, per resynthesis vs GT:
  - highband log-STFT L1 (8000-11025 Hz): lower = closer to the real high band
  - full-band log-STFT L1 (sanity)
  - highband energy ratio resynth/GT (closer to 1 = better; fmax=8000 should sit well below 1)
and report the paired mean so the fmax raise can be judged on the band it actually affects.

Usage:
    uv run python scripts/eval_highband_fidelity.py \
        --gt-filelist data/moespeech_w/val.txt \
        --resynth-dir-a eval_mid/fmax11025_trained --label-a fmax11025 \
        --resynth-dir-b eval_mid/fmax8000_bsc      --label-b fmax8000 \
        --output eval_mid/highband.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

SR = 22050
N_FFT = 1024
HOP = 256
HB_LO_HZ = 8000
HB_HI_HZ = 11025


def _log_stft(wav):
    w = torch.from_numpy(np.asarray(wav, dtype=np.float32))
    spec = torch.stft(w, N_FFT, HOP, window=torch.hann_window(N_FFT), center=True, return_complex=True)
    return torch.log(spec.abs().clamp(min=1e-5))  # (freq_bins, frames)


def _band_bins():
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / SR)  # (n_fft//2+1,)
    lo = int(np.searchsorted(freqs, HB_LO_HZ))
    hi = int(np.searchsorted(freqs, HB_HI_HZ, side="right"))
    return lo, hi


def _load(path):
    a, sr = sf.read(str(path), dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    assert sr == SR, f"{path}: {sr}"
    return a


def _metrics(gt, re, lo, hi):
    g, r = _log_stft(gt), _log_stft(re)
    t = min(g.shape[-1], r.shape[-1])
    g, r = g[:, :t], r[:, :t]
    hb_l1 = (g[lo:hi] - r[lo:hi]).abs().mean().item()
    full_l1 = (g - r).abs().mean().item()
    # linear-domain high-band energy ratio (resynth / GT)
    g_e = torch.exp(g[lo:hi]).pow(2).mean().item()
    r_e = torch.exp(r[lo:hi]).pow(2).mean().item()
    ratio = r_e / g_e if g_e > 0 else 0.0
    return hb_l1, full_l1, ratio


def main(argv=None):
    p = argparse.ArgumentParser(description="8-11 kHz high-band spectral fidelity A/B vs ground truth")
    p.add_argument("--gt-filelist", required=True, help="held-out GT wav paths, same order as resynth indices")
    p.add_argument("--resynth-dir-a", required=True)
    p.add_argument("--label-a", default="A")
    p.add_argument("--resynth-dir-b", required=True)
    p.add_argument("--label-b", default="B")
    p.add_argument("--output", default=None)
    args = p.parse_args(argv)

    gts = [w for w in Path(args.gt_filelist).read_text(encoding="utf-8").splitlines() if w.strip()]
    lo, hi = _band_bins()
    da, db = Path(args.resynth_dir_a), Path(args.resynth_dir_b)
    rows = {args.label_a: [], args.label_b: []}
    for i, gtp in enumerate(gts):
        wa, wb = da / f"{i:04d}.wav", db / f"{i:04d}.wav"
        if not (wa.exists() and wb.exists()):
            continue
        gt = _load(gtp)
        rows[args.label_a].append(_metrics(gt, _load(wa), lo, hi))
        rows[args.label_b].append(_metrics(gt, _load(wb), lo, hi))

    report = {"band_hz": [HB_LO_HZ, HB_HI_HZ], "n": len(rows[args.label_a])}
    for label, m in rows.items():
        arr = np.array(m)  # (n, 3): hb_l1, full_l1, ratio
        report[label] = {
            "highband_logstft_l1_mean": float(arr[:, 0].mean()),
            "fullband_logstft_l1_mean": float(arr[:, 1].mean()),
            "highband_energy_ratio_mean": float(arr[:, 2].mean()),
        }
        print(f"[{label}] highband L1={arr[:, 0].mean():.4f} (lower=better)  "
              f"fullband L1={arr[:, 1].mean():.4f}  highband energy ratio={arr[:, 2].mean():.4f} (→1=better)")
    a, b = np.array(rows[args.label_a]), np.array(rows[args.label_b])
    d = a[:, 0] - b[:, 0]  # highband L1 delta (A - B); negative => A closer to GT high band
    report["highband_l1_delta_A_minus_B"] = {"mean": float(d.mean()), "std": float(d.std()),
                                             "A_better_rate": float((d < 0).mean()), "n": int(len(d))}
    print(f"[paired] highband L1 {args.label_a}-{args.label_b}: {d.mean():+.4f} "
          f"({args.label_a} closer {(d < 0).mean() * 100:.0f}%, n={len(d)})  [negative => {args.label_a} better]")
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[done] {args.output}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
