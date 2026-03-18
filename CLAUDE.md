# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Matcha-TTS is a fast, non-autoregressive Text-to-Speech system based on conditional flow matching (ICASSP 2024). It generates mel-spectrograms from text using an ODE-based approach, then converts them to waveforms via HiFi-GAN vocoder.

## Common Commands

### Installation
```bash
pip install -e .  # Installs package with Cython extension (monotonic_align)
```

### Training
```bash
python matcha/train.py experiment=ljspeech              # Standard LJ Speech training
python matcha/train.py experiment=ljspeech_min_memory    # Low-memory variant
python matcha/train.py experiment=multispeaker           # Multi-speaker (VCTK)
```
Training uses Hydra for config composition. Override any parameter with `key=value` syntax.

### Inference
```bash
matcha-tts --text "Hello world"                          # CLI synthesis
matcha-tts --file input.txt --batched --batch_size 32    # Batch mode
matcha-tts-app                                           # Gradio web UI
```

### Testing & Linting
```bash
make test           # Run fast tests (skip @pytest.mark.slow)
make test-full      # Run all tests including slow
make format         # Run pre-commit hooks (black, isort, flake8, pylint)
```
- pytest config is in `pyproject.toml` (test dir: `tests/`)
- Black: 120-char line length, Python 3.10 target
- flake8 excludes: `logs/*`, `data/*`, `matcha/hifigan/*`

### Data Utilities
```bash
matcha-data-stats -i ljspeech.yaml                          # Compute mel statistics
matcha-tts-get-durations -i ljspeech.yaml -c model.ckpt     # Extract duration alignments
```

### ONNX
```bash
python3 -m matcha.onnx.export model.ckpt output.onnx --n-timesteps 5
python3 -m matcha.onnx.infer output.onnx --text "hello" --output-dir ./outputs
```

## Architecture

### Synthesis Pipeline
```
Text → cleaners (english_cleaners2) → phoneme sequence → intersperse blanks
  → TextEncoder (Conformer + DurationPredictor)
  → expand to mel-length via predicted durations
  → CFM Decoder (Euler ODE solver, n_timesteps steps)
  → mel-spectrogram
  → HiFi-GAN vocoder → waveform (22050 Hz)
```

### Key Modules

- **`matcha/models/matcha_tts.py`** — Main model (PyTorch Lightning module). Orchestrates encoder, flow matching, and synthesis.
- **`matcha/models/components/text_encoder.py`** — Conformer-based encoder with duration predictor (ConvReluNorm).
- **`matcha/models/components/flow_matching.py`** — Conditional flow matching (BASECFM). Euler ODE solver interpolates from noise to mel.
- **`matcha/models/components/decoder.py`** — U-Net-like decoder with ResNet blocks, downsampling, and transformer attention (uses diffusers library).
- **`matcha/text/`** — Text-to-phoneme pipeline. 178-symbol vocabulary. Cleaners handle normalization/number expansion.
- **`matcha/data/text_mel_datamodule.py`** — Lightning DataModule for loading text+audio filelists, computing mel-spectrograms.
- **`matcha/hifigan/`** — HiFi-GAN vocoder (pre-trained, loaded separately). Includes optional denoiser.
- **`matcha/utils/monotonic_align/`** — Cython-accelerated monotonic alignment search (MAS). Compiled during `pip install`.

### Training Losses
Three losses summed: **duration loss** (MSE on predicted durations), **prior loss** (KL divergence), **flow matching loss** (denoising objective on mel).

### Configuration System
Hydra configs in `configs/`. Main composition: `train.yaml` pulls from `data/`, `model/`, `trainer/`, `callbacks/`, `logger/`, `optimizer/`, `scheduler/`. Experiment files (e.g., `experiment/ljspeech.yaml`) override defaults.

### Key Parameters
- `n_feats: 80` (mel bins), `sample_rate: 22050`, `hop_length: 256`, `n_fft: 1024`
- `n_vocab: 178` (phoneme vocabulary size)
- `data_statistics`: pre-computed `mel_mean`/`mel_std` for z-score normalization
- Inference controls: `n_timesteps` (ODE steps), `temperature` (noise variance), `length_scale` (speaking rate)
