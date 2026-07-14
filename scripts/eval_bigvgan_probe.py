"""Diagnostic probe: is the JVS muddiness a vocoder-quality ceiling or a structural
(fmax=8000 / mel) ceiling?

Vocodes the SAME jvs_aligned predicted mels with BigVGAN v2 (fmax8k, mel-compatible,
strongest available, anti-aliased) alongside WaveNeXt and the current HiFi-GAN, then
scores UTMOS and writes wavs for listening.

Interpretation:
  - BigVGAN clearly cleaner  -> vocoder quality IS the lever -> Phase B (fine-tune) is worth it.
  - BigVGAN equally muddy     -> structural (fmax=8000 / mel) ceiling -> fine-tune won't help.

BigVGAN is a quality reference only (112M, GPU-oriented, iSTFT-free but heavy) — NOT a
deployment candidate. use_cuda_kernel=False runs it in pure torch on CPU (slow but fine
for a few samples).

Usage:
    uv run --no-sync python scripts/eval_bigvgan_probe.py --checkpoint <jvs_aligned/last.ckpt> \
        --speakers 0-1 --n-timesteps 32 --output-dir eval/bigvgan_probe
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

# Import matcha modules FIRST so BigVGAN's absolute imports (utils/env/activations)
# added to sys.path below cannot shadow them.
from matcha.cli import load_matcha, load_vocoder, process_text, to_waveform  # noqa: E402
from matcha.utils.utils import assert_model_downloaded, get_user_data_dir  # noqa: E402

UTMOS_HUB_REPO = "tarepan/SpeechMOS:v1.2.0"
UTMOS_HUB_MODEL = "utmos22_strong"
BIGVGAN_REPO = "nvidia/bigvgan_v2_22khz_80band_fmax8k_256x"


def parse_speakers(spec):
    if "-" in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in spec.split(",")]


def load_bigvgan(device):
    """Load BigVGAN v2 in pure-torch (CPU) mode via its HF repo code.

    Constructs from config.json + bigvgan_generator.pt directly; from_pretrained()
    is bypassed because BigVGAN's PyTorchModelHubMixin usage is incompatible with
    huggingface_hub>=1.x (missing proxies/resume_download kwargs).
    """
    import json
    import os

    from huggingface_hub import snapshot_download

    repo_dir = snapshot_download(BIGVGAN_REPO)
    sys.path.insert(0, repo_dir)
    import bigvgan  # provided by the HF repo
    from env import AttrDict  # provided by the HF repo

    with open(os.path.join(repo_dir, "config.json"), encoding="utf-8") as f:
        h = AttrDict(json.load(f))
    model = bigvgan.BigVGAN(h, use_cuda_kernel=False)
    sd = torch.load(os.path.join(repo_dir, "bigvgan_generator.pt"), map_location="cpu")
    model.load_state_dict(sd["generator"])
    model.remove_weight_norm()
    return model.eval().to(device)


def bigvgan_to_waveform(mel, model):
    with torch.inference_mode():
        wav = model(mel).clamp(-1, 1)  # (B, 1, T*256)
    return wav.squeeze().cpu()


def hifi_wavenext_path(vocoder_name):
    if vocoder_name == "wavenext":
        from huggingface_hub import hf_hub_download

        return hf_hub_download("BSC-LT/wavenext-mel", "pytorch_model.bin")
    from matcha.cli import VOCODER_URLS

    save_dir = get_user_data_dir()
    path = save_dir / vocoder_name
    assert_model_downloaded(path, VOCODER_URLS[vocoder_name])
    return str(path)


def main(argv=None):
    p = argparse.ArgumentParser(description="BigVGAN diagnostic probe")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--text-file", default="eval/texts_ja.txt")
    p.add_argument("--speakers", default="0-1")
    p.add_argument("--n-timesteps", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.667)
    p.add_argument("--output-dir", default="eval/bigvgan_probe")
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)

    device = torch.device(args.device)
    texts = [t for t in Path(args.text_file).read_text(encoding="utf-8").splitlines() if t.strip()]
    speakers = parse_speakers(args.speakers)
    out_root = Path(args.output_dir)
    print(f"[cfg] {len(texts)} texts x {len(speakers)} speakers = {len(texts) * len(speakers)} samples")

    model = load_matcha("jvs_aligned", args.checkpoint, device)
    items = []
    for spk in speakers:
        spks = torch.tensor([spk], dtype=torch.long, device=device)
        for i, text in enumerate(texts):
            ti = process_text(i, text, device, language="ja")
            with torch.inference_mode():
                out = model.synthesise(
                    ti["x"], ti["x_lengths"], n_timesteps=args.n_timesteps,
                    temperature=args.temperature, spks=spks, length_scale=1.0,
                )
            items.append({"spk": spk, "idx": i, "mel": out["mel"]})
    print(f"[synth] {len(items)} predicted mels")

    # Vocoders: BigVGAN (reference) + WaveNeXt + HiFi-GAN, all on the SAME mels.
    wav_paths = {"bigvgan": {}, "wavenext": {}, "hifigan_univ_v1": {}}

    bv = load_bigvgan(device)
    for it in items:
        rel = f"spk_{it['spk']:03d}/text_{it['idx']:02d}"
        wav = bigvgan_to_waveform(it["mel"], bv)
        dst = out_root / "bigvgan" / f"{rel}.wav"
        dst.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(dst), wav.numpy(), 22050)
        wav_paths["bigvgan"][rel] = dst
    print(f"[vocode] bigvgan: {len(wav_paths['bigvgan'])} wavs")

    for vname in ("wavenext", "hifigan_univ_v1"):
        vocoder, denoiser = load_vocoder(vname, hifi_wavenext_path(vname), device)
        for it in items:
            rel = f"spk_{it['spk']:03d}/text_{it['idx']:02d}"
            with torch.inference_mode():
                wav = to_waveform(it["mel"], vocoder, denoiser)
            dst = out_root / vname / f"{rel}.wav"
            dst.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(dst), wav.numpy(), 22050)
            wav_paths[vname][rel] = dst
        print(f"[vocode] {vname}: {len(wav_paths[vname])} wavs")

    predictor = torch.hub.load(UTMOS_HUB_REPO, UTMOS_HUB_MODEL, trust_repo=True)
    report = {"config": vars(args), "n_samples": len(items), "utmos": {}}
    per = {}
    for v, paths in wav_paths.items():
        scores = {}
        for rel, path in paths.items():
            audio, sr = sf.read(str(path), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            with torch.no_grad():
                scores[rel] = float(predictor(torch.from_numpy(audio).unsqueeze(0), sr))
        per[v] = scores
        report["utmos"][v] = {"mean": float(np.mean(list(scores.values()))), "std": float(np.std(list(scores.values())))}
        print(f"[UTMOS] {v}: {report['utmos'][v]['mean']:.3f} +/- {report['utmos'][v]['std']:.3f}")

    # Paired deltas vs current HiFi-GAN (same mel key)
    base = "hifigan_univ_v1"
    for v in ("bigvgan", "wavenext"):
        common = sorted(set(per[v]) & set(per[base]))
        d = np.array([per[v][k] - per[base][k] for k in common])
        report.setdefault("paired_delta", {})[f"{v}_minus_{base}"] = {
            "mean": float(d.mean()), "std": float(d.std()), "n": int(len(d)),
            "win_rate": float((d > 0).mean()),
        }
        e = report["paired_delta"][f"{v}_minus_{base}"]
        print(f"[paired] {v} - {base}: {e['mean']:+.3f} (win {e['win_rate'] * 100:.0f}%, n={e['n']})")

    report["per_file_utmos"] = per
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] {out_root / 'report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
