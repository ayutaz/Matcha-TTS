"""Generate evaluation samples from a trained Matcha-TTS model.

Usage:
    uv run python scripts/generate_eval_samples.py \
        --checkpoint logs/train/jvs_aligned/runs/.../checkpoints/last.ckpt \
        --text-file eval/texts_ja.txt \
        --output-dir eval/samples/julius_model \
        --speakers 0-99 \
        --n-timesteps 10 \
        --temperature 0.667
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


def parse_speaker_range(spec: str) -> list[int]:
    """Parse speaker range spec like '0-99', '0,5,10', or a mix like '0-9,12'."""
    speakers = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            start_str, end_str = part.split("-")
            start, end = int(start_str), int(end_str)
            if start > end:
                raise ValueError(f"Reversed speaker range: {part!r}")
            speakers.extend(range(start, end + 1))
        else:
            speakers.append(int(part))
    return speakers


def load_model(checkpoint_path: str, device: str = "cpu"):
    """Load MatchaTTS model from checkpoint."""
    from matcha.cli import load_matcha

    model = load_matcha("eval", checkpoint_path, device)
    return model


def load_vocoder(device: str = "cpu"):
    """Load HiFi-GAN vocoder."""
    from matcha.cli import load_vocoder as _load_vocoder
    from matcha.utils.utils import assert_model_downloaded, get_user_data_dir

    save_dir = get_user_data_dir()
    vocoder_name = "hifigan_univ_v1"
    from matcha.cli import VOCODER_URLS

    vocoder_path = save_dir / vocoder_name
    assert_model_downloaded(vocoder_path, VOCODER_URLS[vocoder_name])
    vocoder, denoiser = _load_vocoder(vocoder_name, vocoder_path, device)
    return vocoder


def process_text(text: str, device: str = "cpu"):
    """Process Japanese text to phoneme sequence."""
    from matcha.cli import process_text as _process_text

    result = _process_text(0, text, device, language="ja")
    return result["x"], result["x_lengths"]


@torch.inference_mode()
def generate_sample(
    model,
    vocoder,
    text: str,
    spk_id: int,
    n_timesteps: int = 10,
    temperature: float = 0.667,
    length_scale: float = 1.0,
    clamp_boundary_blanks: bool = True,
    device: str = "cpu",
) -> dict:
    """Generate a single speech sample.

    Wrapped in ``torch.inference_mode`` so the vocoder call runs in the same
    mode as ``model.synthesise`` (also inference_mode); otherwise passing the
    inference-mode ``mel`` into the grad-tracking vocoder raises
    "Inference tensors cannot be saved for backward".
    """
    x, x_lengths = process_text(text, device)

    spks = torch.tensor([spk_id], dtype=torch.long, device=device) if model.n_spks > 1 else None

    output = model.synthesise(
        x,
        x_lengths,
        n_timesteps,
        temperature=temperature,
        spks=spks,
        length_scale=length_scale,
        clamp_boundary_blanks=clamp_boundary_blanks,
    )

    # Vocoder
    if vocoder is not None:
        from matcha.cli import to_waveform

        waveform = to_waveform(output["mel"], vocoder)
    else:
        waveform = None

    return {
        "mel": output["mel"].cpu(),
        "encoder_outputs": output["encoder_outputs"].cpu(),
        "decoder_outputs": output["decoder_outputs"].cpu(),
        "durations": output["durations"].cpu() if "durations" in output else None,
        "mel_lengths": output["mel_lengths"].cpu(),
        "rtf": output["rtf"],
        "waveform": waveform,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate evaluation samples")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--text-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--speakers", type=str, default="0-99")
    parser.add_argument("--n-timesteps", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.667)
    parser.add_argument("--length-scale", type=float, default=1.0)
    parser.add_argument("--clamp-boundary-blanks", action="store_true", default=False)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--no-vocoder", action="store_true")
    parser.add_argument("--max-speakers", type=int, default=None, help="Limit number of speakers (for testing)")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load texts
    text_file = Path(args.text_file)
    if not text_file.exists():
        print(f"Text file not found: {text_file}", file=sys.stderr)
        return 1

    texts = [line.strip() for line in text_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"Loaded {len(texts)} evaluation texts")

    # Parse speakers
    speakers = parse_speaker_range(args.speakers)
    if args.max_speakers:
        speakers = speakers[: args.max_speakers]
    print(f"Speakers: {len(speakers)} ({speakers[0]}-{speakers[-1]})")

    # Check checkpoint
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}", file=sys.stderr)
        print("Saving metadata only (dry-run mode)")
        # Save metadata even without checkpoint
        metadata = {
            "checkpoint": str(ckpt_path),
            "n_timesteps": args.n_timesteps,
            "temperature": args.temperature,
            "length_scale": args.length_scale,
            "clamp_boundary_blanks": args.clamp_boundary_blanks,
            "texts": texts,
            "speakers": speakers,
            "status": "dry_run",
        }
        (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0

    # Load model
    print(f"Loading model from {ckpt_path}...")
    model = load_model(str(ckpt_path), args.device)

    vocoder = None
    if not args.no_vocoder:
        print("Loading vocoder...")
        vocoder = load_vocoder(args.device)

    # Generate samples
    total = len(speakers) * len(texts)
    generated = 0
    errors = []
    rtfs = []

    for spk_id in tqdm(speakers, desc="Speakers"):
        spk_dir = output_dir / f"spk_{spk_id:03d}"
        spk_dir.mkdir(exist_ok=True)

        for text_idx, text in enumerate(texts):
            try:
                result = generate_sample(
                    model,
                    vocoder,
                    text,
                    spk_id,
                    n_timesteps=args.n_timesteps,
                    temperature=args.temperature,
                    length_scale=args.length_scale,
                    clamp_boundary_blanks=args.clamp_boundary_blanks,
                    device=args.device,
                )

                # Save mel
                mel_path = spk_dir / f"text_{text_idx:02d}.npy"
                np.save(mel_path, result["mel"].numpy())

                # Save waveform
                if result["waveform"] is not None:
                    import soundfile as sf

                    wav_path = spk_dir / f"text_{text_idx:02d}.wav"
                    sf.write(str(wav_path), result["waveform"], 22050)

                # Save duration info
                dur_info = {
                    "text": text,
                    "speaker_id": spk_id,
                    "n_timesteps": args.n_timesteps,
                    "temperature": args.temperature,
                    "length_scale": args.length_scale,
                    "clamp_boundary_blanks": args.clamp_boundary_blanks,
                    "rtf": result["rtf"],
                    "total_frames": result["mel_lengths"][0].item(),
                }
                if result["durations"] is not None:
                    dur_info["predicted_durations"] = result["durations"][0].tolist()

                dur_path = spk_dir / f"text_{text_idx:02d}_dur.json"
                dur_path.write_text(json.dumps(dur_info, ensure_ascii=False, indent=2), encoding="utf-8")

                rtfs.append(result["rtf"])
                generated += 1

            except Exception as e:
                errors.append(f"spk_{spk_id:03d}/text_{text_idx:02d}: {e}")

    # Save metadata
    metadata = {
        "checkpoint": str(ckpt_path),
        "n_timesteps": args.n_timesteps,
        "temperature": args.temperature,
        "length_scale": args.length_scale,
        "clamp_boundary_blanks": args.clamp_boundary_blanks,
        "texts": texts,
        "speakers": speakers,
        "generated": generated,
        "errors": len(errors),
        "mean_rtf": float(np.mean(rtfs)) if rtfs else None,
        "status": "complete",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nGenerated {generated}/{total} samples")
    if errors:
        print(f"Errors ({len(errors)}):")
        for e in errors[:10]:
            print(f"  {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
