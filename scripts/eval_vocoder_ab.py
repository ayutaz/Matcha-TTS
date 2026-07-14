"""Phase A: zero-shot WaveNeXt vs HiFi-GAN A/B for the JVS Japanese model.

Synthesises predicted mels from the jvs_aligned model for a set of texts x speakers,
vocodes each mel with BOTH vocoders (paired on the exact same mel), and scores UTMOS.
Because both vocoders see the identical predicted mel, the paired delta isolates the
vocoder's contribution (the acoustic model is held constant).

Usage (CPU, local):
    uv run --no-sync python scripts/eval_vocoder_ab.py \
        --checkpoint <path/to/jvs_aligned/last.ckpt> \
        --text-file eval/texts_ja.txt \
        --speakers 0-9 --n-timesteps 10 --output-dir eval/vocoder_ab

Downloads (cached): WaveNeXt (BSC-LT/wavenext-mel, 55MB), HiFi-GAN univ (g_02500000),
UTMOS predictor (tarepan/SpeechMOS, ~360MB via torch.hub).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from matcha.cli import load_matcha, load_vocoder, process_text, to_waveform
from matcha.utils.utils import assert_model_downloaded, get_user_data_dir

UTMOS_HUB_REPO = "tarepan/SpeechMOS:v1.2.0"
UTMOS_HUB_MODEL = "utmos22_strong"


def parse_speakers(spec):
    """'0-9' -> [0..9]; '0,3,7' -> [0,3,7]."""
    if "-" in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in spec.split(",")]


# Aliases let the same "wavenext" loader run different weight files in one A/B run.
# Maps an eval label -> (loader_name, checkpoint_path_or_None). Populated in main().
VOCODER_ALIASES = {}


def get_vocoder_path(vocoder_name):
    """Resolve a local checkpoint path for a vocoder, downloading if needed."""
    if vocoder_name in VOCODER_ALIASES:
        _loader, path = VOCODER_ALIASES[vocoder_name]
        if path is not None:
            return path
    if vocoder_name.startswith("wavenext"):
        from huggingface_hub import hf_hub_download

        return hf_hub_download("BSC-LT/wavenext-mel", "pytorch_model.bin")
    # hifigan_* : reuse Matcha's user-data-dir + release-download flow
    from matcha.cli import VOCODER_URLS

    save_dir = get_user_data_dir()
    path = save_dir / vocoder_name
    assert_model_downloaded(path, VOCODER_URLS[vocoder_name])
    return str(path)


def loader_name(vocoder_name):
    """The matcha.cli.load_vocoder name to use for a (possibly aliased) eval label."""
    if vocoder_name in VOCODER_ALIASES:
        return VOCODER_ALIASES[vocoder_name][0]
    return vocoder_name


def synthesise_mels(model, texts, speakers, n_timesteps, temperature, device):
    """Return list of {spk, idx, text, mel(1,80,T)} — denormalized vocoder-domain mels."""
    items = []
    for spk in speakers:
        spks = torch.tensor([spk], dtype=torch.long, device=device)
        for i, text in enumerate(texts):
            ti = process_text(i, text, device, language="ja")
            with torch.inference_mode():
                out = model.synthesise(
                    ti["x"], ti["x_lengths"], n_timesteps=n_timesteps,
                    temperature=temperature, spks=spks, length_scale=1.0,
                )
            items.append({"spk": spk, "idx": i, "text": text, "mel": out["mel"]})
    return items


def score_utmos(wav_paths, predictor):
    scores = {}
    for rel, path in wav_paths.items():
        audio, sr = sf.read(str(path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        with torch.no_grad():
            scores[rel] = float(predictor(torch.from_numpy(audio).unsqueeze(0), sr))
    return scores


def main(argv=None):
    p = argparse.ArgumentParser(description="Phase A WaveNeXt vs HiFi-GAN A/B")
    p.add_argument("--checkpoint", required=True, help="jvs_aligned last.ckpt path")
    p.add_argument("--text-file", default="eval/texts_ja.txt")
    p.add_argument("--speakers", default="0-4", help="'0-9' or '0,3,7'")
    p.add_argument("--n-timesteps", type=int, default=10)
    p.add_argument("--temperature", type=float, default=0.667)
    p.add_argument("--vocoders", nargs="+", default=["wavenext", "hifigan_univ_v1"])
    p.add_argument("--output-dir", default="eval/vocoder_ab")
    p.add_argument("--device", default="cpu")
    p.add_argument("--no-utmos", action="store_true", help="only synthesise/vocode, skip scoring")
    p.add_argument("--wavenext-ja-bin", default=None,
                   help="local WaveNeXt bin to run under the label 'wavenext_ja' (loaded via the wavenext loader)")
    args = p.parse_args(argv)

    if args.wavenext_ja_bin:
        VOCODER_ALIASES["wavenext_ja"] = ("wavenext", args.wavenext_ja_bin)

    device = torch.device(args.device)
    texts = [t for t in Path(args.text_file).read_text(encoding="utf-8").splitlines() if t.strip()]
    speakers = parse_speakers(args.speakers)
    out_root = Path(args.output_dir)
    print(f"[cfg] {len(texts)} texts x {len(speakers)} speakers = {len(texts) * len(speakers)} samples, "
          f"n_timesteps={args.n_timesteps}, vocoders={args.vocoders}")

    model = load_matcha("jvs_aligned", args.checkpoint, device)
    items = synthesise_mels(model, texts, speakers, args.n_timesteps, args.temperature, device)
    print(f"[synth] generated {len(items)} predicted mels")

    # Vocode the SAME mels with each vocoder -> paired wavs under <out>/<vocoder>/spk_XX/text_YY.wav
    wav_paths = {v: {} for v in args.vocoders}
    for vname in args.vocoders:
        vpath = get_vocoder_path(vname)
        vocoder, denoiser = load_vocoder(loader_name(vname), vpath, device)
        for it in items:
            rel = f"spk_{it['spk']:03d}/text_{it['idx']:02d}"
            # synthesise() runs under inference_mode, so its mel is an inference tensor;
            # the vocoder call must also be in inference_mode to avoid autograd tracking.
            with torch.inference_mode():
                wav = to_waveform(it["mel"], vocoder, denoiser)
            dst = out_root / vname / f"{rel}.wav"
            dst.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(dst), wav.numpy(), 22050)
            wav_paths[vname][rel] = dst
        print(f"[vocode] {vname}: wrote {len(wav_paths[vname])} wavs")

    report = {"config": vars(args), "n_samples": len(items)}
    if not args.no_utmos:
        predictor = torch.hub.load(UTMOS_HUB_REPO, UTMOS_HUB_MODEL, trust_repo=True)
        per_voc = {v: score_utmos(wav_paths[v], predictor) for v in args.vocoders}
        report["utmos"] = {
            v: {"mean": float(np.mean(list(s.values()))), "std": float(np.std(list(s.values())))}
            for v, s in per_voc.items()
        }
        for v, st in report["utmos"].items():
            print(f"[UTMOS] {v}: {st['mean']:.3f} +/- {st['std']:.3f}")

        # Paired delta vs the first hifigan_* baseline (same mel -> same rel key)
        baseline = next((v for v in args.vocoders if v.startswith("hifigan")), None)
        if baseline:
            for v in args.vocoders:
                if v == baseline:
                    continue
                common = sorted(set(per_voc[v]) & set(per_voc[baseline]))
                deltas = [per_voc[v][k] - per_voc[baseline][k] for k in common]
                report.setdefault("paired_delta", {})[f"{v}_minus_{baseline}"] = {
                    "mean": float(np.mean(deltas)), "std": float(np.std(deltas)),
                    "n": len(deltas), "win_rate": float(np.mean([d > 0 for d in deltas])),
                }
                d = report["paired_delta"][f"{v}_minus_{baseline}"]
                print(f"[paired] {v} - {baseline}: {d['mean']:+.3f} +/- {d['std']:.3f} "
                      f"(win {d['win_rate'] * 100:.0f}% over n={d['n']})")
        report["per_file_utmos"] = per_voc

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] report -> {out_root / 'report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
