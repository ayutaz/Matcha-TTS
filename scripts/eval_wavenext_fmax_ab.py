"""Phase W validation: does fmax=11025 reduce muddiness vs the current fmax=8000 vocoder?

GT-mel resynthesis A/B (no acoustic model needed): for each held-out real wav,
  A: wav -> mel(fmax=11025) -> trained WaveNeXt(11025) -> wav_A
  B: wav -> mel(fmax=8000)  -> BSC WaveNeXt(8000)     -> wav_B
then UTMOS + listening. If A clearly beats B in high-frequency clarity, the fmax=11025
raise is worth the full acoustic-model pretrain; if not, it saves that spend.

Usage:
    uv run python scripts/eval_wavenext_fmax_ab.py \
        --val-filelist data/moespeech_w/val.txt \
        --wavenext-11025 checkpoints/wavenext_ja_11025.bin \
        --n 40 --output-dir eval/wavenext_fmax_ab
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from matcha.cli import VOCODER_URLS, load_vocoder
from matcha.utils.audio import mel_spectrogram
from matcha.utils.utils import assert_model_downloaded, get_user_data_dir

UTMOS_HUB_REPO = "tarepan/SpeechMOS:v1.2.0"
UTMOS_HUB_MODEL = "utmos22_strong"


def _mel(wav, fmax):
    return mel_spectrogram(wav, 1024, 80, 22050, 256, 1024, 0.0, fmax, center=False)


def _bsc_wavenext_path():
    from huggingface_hub import hf_hub_download

    return hf_hub_download("BSC-LT/wavenext-mel", "pytorch_model.bin")


def _hifigan_path():
    save_dir = get_user_data_dir()
    path = save_dir / "hifigan_univ_v1"
    assert_model_downloaded(path, VOCODER_URLS["hifigan_univ_v1"])
    return str(path)


def main(argv=None):
    p = argparse.ArgumentParser(description="fmax=11025 (trained) vs fmax=8000 (BSC) GT-mel resynthesis A/B")
    p.add_argument("--val-filelist", required=True, help="held-out real 22050 Hz wav paths (one per line)")
    p.add_argument("--wavenext-11025", required=True, help="trained fmax=11025 WaveNeXt generator bin")
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--output-dir", default="eval/wavenext_fmax_ab")
    p.add_argument("--device", default="cpu")
    p.add_argument("--no-utmos", action="store_true")
    args = p.parse_args(argv)

    device = torch.device(args.device)
    wavs = [w for w in Path(args.val_filelist).read_text(encoding="utf-8").splitlines() if w.strip()][: args.n]
    out_root = Path(args.output_dir)
    print(f"[cfg] {len(wavs)} held-out wavs")

    # fmax=11025 (trained) and fmax=8000 (BSC) both loaded via the matcha inference loader.
    wn_11025, _ = load_vocoder("wavenext", args.wavenext_11025, device)
    wn_8000, _ = load_vocoder("wavenext", _bsc_wavenext_path(), device)

    def resynth(wav_path, vocoder, fmax):
        audio, sr = sf.read(wav_path, dtype="float32")
        assert sr == 22050, f"{wav_path}: {sr}"
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        y = torch.from_numpy(audio).unsqueeze(0).to(device)
        with torch.inference_mode():
            wav = vocoder(_mel(y, fmax)).clamp(-1, 1)
        return wav.squeeze().cpu().numpy()

    paths = {"fmax11025_trained": {}, "fmax8000_bsc": {}}
    for i, wp in enumerate(wavs):
        rel = f"{i:04d}"
        for tag, voc, fmax in [("fmax11025_trained", wn_11025, 11025.0), ("fmax8000_bsc", wn_8000, 8000.0)]:
            w = resynth(wp, voc, fmax)
            dst = out_root / tag / f"{rel}.wav"
            dst.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(dst), w, 22050)
            paths[tag][rel] = dst
    print(f"[resynth] wrote {len(wavs)} x 2 wavs -> {out_root}")

    report = {"config": vars(args), "n": len(wavs)}
    if not args.no_utmos:
        predictor = torch.hub.load(UTMOS_HUB_REPO, UTMOS_HUB_MODEL, trust_repo=True)
        per = {}
        for tag, d in paths.items():
            scores = {}
            for rel, path in d.items():
                a, s = sf.read(str(path), dtype="float32")
                with torch.no_grad():
                    scores[rel] = float(predictor(torch.from_numpy(a).unsqueeze(0), s))
            per[tag] = scores
            report.setdefault("utmos", {})[tag] = {
                "mean": float(np.mean(list(scores.values()))),
                "std": float(np.std(list(scores.values()))),
            }
            print(f"[UTMOS] {tag}: {report['utmos'][tag]['mean']:.3f} +/- {report['utmos'][tag]['std']:.3f}")
        common = sorted(set(per["fmax11025_trained"]) & set(per["fmax8000_bsc"]))
        d = np.array([per["fmax11025_trained"][k] - per["fmax8000_bsc"][k] for k in common])
        report["paired_delta_11025_minus_8000"] = {
            "mean": float(d.mean()), "std": float(d.std()), "n": int(len(d)), "win_rate": float((d > 0).mean())
        }
        print(f"[paired] fmax11025 - fmax8000: {d.mean():+.3f} (win {(d > 0).mean() * 100:.0f}%, n={len(d)})")
        report["per_file_utmos"] = per

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] {out_root / 'report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
