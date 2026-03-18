import argparse
import os
import warnings
from pathlib import Path
from time import perf_counter

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch

from matcha.cli import plot_spectrogram_to_numpy, process_text


def validate_args(args):
    assert args.text or args.file, (
        "Either text or file must be provided Matcha-T(ea)TTS need sometext to whisk the waveforms."
    )
    assert args.temperature >= 0, "Sampling temperature cannot be negative"
    assert args.speaking_rate >= 0, "Speaking rate must be greater than 0"
    if args.cleaners is None:
        args.cleaners = ["japanese_cleaners"] if args.language == "ja" else None
    return args


def resolve_model_path(model_path, quantized):
    """Resolve model path, handling INT8 quantized model selection.

    When --quantized is passed, looks for a *_int8.onnx file derived from
    the given model path.  These files are produced by
    ``python -m matcha.onnx.export --quantize``.

    Returns the resolved path as a string.
    """
    model_path = Path(model_path)

    if quantized:
        # Explicitly requested: derive the _int8 path from the given model
        int8_path = model_path.with_name(model_path.stem + "_int8.onnx")
        if int8_path.exists():
            print(f"[+] Loading INT8 quantized model: {int8_path}")
            return str(int8_path)
        else:
            raise FileNotFoundError(
                f"Quantized model not found at {int8_path}. "
                f"Generate one with: python -m matcha.onnx.export --quantize <checkpoint> {model_path}"
            )

    return str(model_path)


def create_session_options(num_threads=None):
    """Create optimised ONNX Runtime session options.

    Args:
        num_threads: Number of threads for intra/inter op parallelism.
            Defaults to ``os.cpu_count()`` (capped at a sensible maximum).
    """
    if num_threads is None:
        cpu_count = os.cpu_count() or 4
        # Cap to avoid over-subscription on high-core machines
        num_threads = min(cpu_count, 16)

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = num_threads
    sess_options.inter_op_num_threads = num_threads
    sess_options.enable_mem_pattern = True
    sess_options.enable_cpu_mem_arena = True
    return sess_options


def write_wavs(model, inputs, output_dir, original_indices=None, external_vocoder=None):
    if external_vocoder is None:
        print("The provided model has the vocoder embedded in the graph.\nGenerating waveform directly")
        t0 = perf_counter()
        wavs, wav_lengths = model.run(None, inputs)
        infer_secs = perf_counter() - t0
        mel_infer_secs = vocoder_infer_secs = None
    else:
        print("[🍵] Generating mel using Matcha")
        mel_t0 = perf_counter()
        mels, mel_lengths = model.run(None, inputs)
        mel_infer_secs = perf_counter() - mel_t0
        print("Generating waveform from mel using external vocoder")
        vocoder_inputs = {external_vocoder.get_inputs()[0].name: mels}
        vocoder_t0 = perf_counter()
        wavs = external_vocoder.run(None, vocoder_inputs)[0]
        vocoder_infer_secs = perf_counter() - vocoder_t0
        wavs = wavs.squeeze(1)
        wav_lengths = mel_lengths * 256
        infer_secs = mel_infer_secs + vocoder_infer_secs

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, (wav, wav_length) in enumerate(zip(wavs, wav_lengths)):
        # Use original index for naming when inputs were re-sorted
        out_idx = original_indices[i] if original_indices is not None else i
        output_filename = output_dir.joinpath(f"output_{out_idx + 1}.wav")
        audio = wav[:wav_length]
        print(f"Writing audio to {output_filename}")
        sf.write(output_filename, audio, 22050, "PCM_24")

    wav_secs = wav_lengths.sum() / 22050
    print(f"Inference seconds: {infer_secs}")
    print(f"Generated wav seconds: {wav_secs}")
    rtf = infer_secs / wav_secs
    if mel_infer_secs is not None:
        mel_rtf = mel_infer_secs / wav_secs
        print(f"Matcha RTF: {mel_rtf}")
    if vocoder_infer_secs is not None:
        vocoder_rtf = vocoder_infer_secs / wav_secs
        print(f"Vocoder RTF: {vocoder_rtf}")
    print(f"Overall RTF: {rtf}")


def write_mels(model, inputs, output_dir, original_indices=None):
    t0 = perf_counter()
    mels, mel_lengths = model.run(None, inputs)
    infer_secs = perf_counter() - t0

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, mel in enumerate(mels):
        out_idx = original_indices[i] if original_indices is not None else i
        output_stem = output_dir.joinpath(f"output_{out_idx + 1}")
        plot_spectrogram_to_numpy(mel.squeeze(), output_stem.with_suffix(".png"))
        np.save(output_stem.with_suffix(".numpy"), mel)

    wav_secs = (mel_lengths * 256).sum() / 22050
    print(f"Inference seconds: {infer_secs}")
    print(f"Generated wav seconds: {wav_secs}")
    rtf = infer_secs / wav_secs
    print(f"RTF: {rtf}")


def main():
    parser = argparse.ArgumentParser(
        description=" 🍵 Matcha-TTS: A fast TTS architecture with conditional flow matching"
    )
    parser.add_argument(
        "model",
        type=str,
        help="ONNX model to use",
    )
    parser.add_argument("--vocoder", type=str, default=None, help="Vocoder to use (defaults to None)")
    parser.add_argument("--text", type=str, default=None, help="Text to synthesize")
    parser.add_argument("--file", type=str, default=None, help="Text file to synthesize")
    parser.add_argument("--spk", type=int, default=None, help="Speaker ID")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.667,
        help="Variance of the x0 noise (default: 0.667)",
    )
    parser.add_argument(
        "--speaking-rate",
        type=float,
        default=1.0,
        help="change the speaking rate, a higher value means slower speaking rate (default: 1.0)",
    )
    parser.add_argument("--gpu", action="store_true", help="Use CPU for inference (default: use GPU if available)")
    parser.add_argument("--tensorrt", action="store_true", help="Use TensorRT execution provider (requires --gpu)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.getcwd(),
        help="Output folder to save results (default: current dir)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        choices=["en", "ja"],
        help="Language for text processing (default: en)",
    )
    parser.add_argument(
        "--cleaners",
        type=str,
        nargs="+",
        default=None,
        help="Text cleaners to use (default: auto-selected based on --language)",
    )
    parser.add_argument(
        "--quantized",
        action="store_true",
        help="Load INT8 quantized model (*_int8.onnx) produced by export --quantize",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Number of threads for ONNX Runtime (default: auto based on CPU count)",
    )

    args = parser.parse_args()
    args = validate_args(args)

    model_path = resolve_model_path(args.model, args.quantized)
    sess_options = create_session_options(num_threads=args.threads)

    if args.gpu and args.tensorrt:
        providers = [
            ('TensorrtExecutionProvider', {
                'trt_max_workspace_size': 2147483648,
            }),
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ]
    elif args.gpu:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    model = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)

    model_inputs = model.get_inputs()
    model_outputs = list(model.get_outputs())

    if args.text:
        text_lines = args.text.splitlines()
    else:
        with open(args.file, encoding="utf-8") as file:
            text_lines = file.read().splitlines()

    processed_lines = [process_text(i, line, "cpu", cleaners=args.cleaners, language=args.language) for i, line in enumerate(text_lines)]

    # Sort by sequence length to minimise padding waste (matches cli.py batched_synthesis)
    sorted_indices = sorted(range(len(processed_lines)), key=lambda k: processed_lines[k]["x"].shape[-1])
    sorted_lines = [processed_lines[idx] for idx in sorted_indices]
    # Build reverse mapping: sorted position -> original index
    original_indices = [sorted_indices[i] for i in range(len(sorted_indices))]

    x = [line["x"].squeeze() for line in sorted_lines]
    # Pad (shorter sequences grouped together after sort = less wasted padding)
    x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True)
    x = x.detach().cpu().numpy()
    x_lengths = np.array([line["x_lengths"].item() for line in sorted_lines], dtype=np.int64)

    if len(text_lines) > 1:
        unsorted_max = max(p["x"].shape[-1] for p in processed_lines)
        sorted_max = x.shape[1]
        if sorted_max < unsorted_max:
            print(f"[+] Length sorting reduced max padded length: {unsorted_max} -> {sorted_max}")

    inputs = {
        "x": x,
        "x_lengths": x_lengths,
        "scales": np.array([args.temperature, args.speaking_rate], dtype=np.float32),
    }
    is_multi_speaker = len(model_inputs) == 4
    if is_multi_speaker:
        if args.spk is None:
            args.spk = 0
            warn = "[!] Speaker ID not provided! Using speaker ID 0"
            warnings.warn(warn, UserWarning)
        inputs["spks"] = np.repeat(args.spk, x.shape[0]).astype(np.int64)

    has_vocoder_embedded = model_outputs[0].name == "wav"
    if has_vocoder_embedded:
        write_wavs(model, inputs, args.output_dir, original_indices=original_indices)
    elif args.vocoder:
        external_vocoder = ort.InferenceSession(args.vocoder, sess_options=sess_options, providers=providers)
        write_wavs(model, inputs, args.output_dir, original_indices=original_indices, external_vocoder=external_vocoder)
    else:
        warn = "[!] A vocoder is not embedded in the graph nor an external vocoder is provided. The mel output will be written as numpy arrays to `*.npy` files in the output directory"
        warnings.warn(warn, UserWarning)
        write_mels(model, inputs, args.output_dir, original_indices=original_indices)


if __name__ == "__main__":
    main()
