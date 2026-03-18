import tempfile
from argparse import Namespace
from pathlib import Path

import gradio as gr
import soundfile as sf
import torch

from matcha.cli import (
    MATCHA_URLS,
    VOCODER_URLS,
    assert_model_downloaded,
    get_device,
    load_matcha,
    load_vocoder,
    process_text,
    to_waveform,
)
from matcha.utils.utils import get_user_data_dir, plot_tensor

LOCATION = Path(get_user_data_dir())

args = Namespace(
    cpu=False,
    model="matcha_vctk",
    vocoder="hifigan_univ_v1",
    spk=0,
)

CURRENTLY_LOADED_MODEL = args.model
CURRENT_LANGUAGE = "en"

# Cache for loaded models: key -> (model, vocoder, denoiser)
_model_cache = {}


def MATCHA_TTS_LOC(x):
    return LOCATION / f"{x}.ckpt"


def VOCODER_LOC(x):
    return LOCATION / f"{x}"


def _warmup_model(m, dev, n_timesteps=10):
    """Run a warmup pass to trigger CUDA kernel caching and torch.compile tracing.

    Uses realistic input shapes and the actual n_timesteps value to ensure
    compiled graphs cover the shapes used during real inference.
    """
    if dev.type == "cuda":
        print("[+] Running GPU warmup...")
        # Use a realistic phoneme sequence length (~50 tokens after intersperse)
        # to trigger compilation/caching for shapes close to real inference.
        seq_len = 50
        with torch.inference_mode():
            dummy_x = torch.randint(0, 178, (1, seq_len), dtype=torch.long, device=dev)
            dummy_x_lengths = torch.tensor([seq_len], dtype=torch.long, device=dev)
            output = m.synthesise(dummy_x, dummy_x_lengths, n_timesteps=n_timesteps)
        torch.cuda.synchronize()
        print("[+] Warmup complete.")


def _compile_models(m, v, dev):
    """Apply torch.compile to performance-critical model components on CUDA."""
    if dev.type == "cuda":
        print("[+] Compiling model components with torch.compile...")
        m.encoder = torch.compile(m.encoder, mode="reduce-overhead")
        m.decoder.estimator = torch.compile(m.decoder.estimator, mode="reduce-overhead")
        v = torch.compile(v, mode="reduce-overhead")
        print("[+] Compilation complete.")
    return m, v


def load_model(model_name, vocoder_name):
    cache_key = (model_name, vocoder_name)
    if cache_key in _model_cache:
        print(f"[+] Using cached model: {model_name} + {vocoder_name}")
        return _model_cache[cache_key]

    m = load_matcha(model_name, MATCHA_TTS_LOC(model_name), device)
    v, d = load_vocoder(vocoder_name, VOCODER_LOC(vocoder_name), device)
    m, v = _compile_models(m, v, device)
    _warmup_model(m, device)
    _model_cache[cache_key] = (m, v, d)
    return m, v, d


LOGO_URL = "https://shivammehta25.github.io/Matcha-TTS/images/logo.png"
RADIO_OPTIONS = {
    "マルチスピーカー (VCTK)": {
        "model": "matcha_vctk",
        "vocoder": "hifigan_univ_v1",
        "language": "en",
    },
    "シングルスピーカー (LJ Speech)": {
        "model": "matcha_ljspeech",
        "vocoder": "hifigan_T2_v1",
        "language": "en",
    },
    "日本語 (JSUT)": {
        "model": "matcha_jsut",
        "vocoder": "hifigan_univ_v1",
        "language": "ja",
    },
}

# Ensure all the required models are downloaded
assert_model_downloaded(MATCHA_TTS_LOC("matcha_ljspeech"), MATCHA_URLS["matcha_ljspeech"])
assert_model_downloaded(VOCODER_LOC("hifigan_T2_v1"), VOCODER_URLS["hifigan_T2_v1"])
assert_model_downloaded(MATCHA_TTS_LOC("matcha_vctk"), MATCHA_URLS["matcha_vctk"])
assert_model_downloaded(VOCODER_LOC("hifigan_univ_v1"), VOCODER_URLS["hifigan_univ_v1"])

device = get_device(args)

# Load default model (with warmup and caching)
model, vocoder, denoiser = load_model(args.model, args.vocoder)


def load_model_ui(model_type, textbox):
    model_name, vocoder_name = RADIO_OPTIONS[model_type]["model"], RADIO_OPTIONS[model_type]["vocoder"]
    language = RADIO_OPTIONS[model_type]["language"]

    global model, vocoder, denoiser, CURRENTLY_LOADED_MODEL, CURRENT_LANGUAGE  # pylint: disable=global-statement
    if model_name != CURRENTLY_LOADED_MODEL:
        model_path = MATCHA_TTS_LOC(model_name)
        if not model_path.exists():
            gr.Warning(f"モデル '{model_name}' が見つかりません: {model_path}")
            return (
                textbox,
                gr.Button(interactive=False),
                gr.Slider(visible=False, value=-1),
                gr.Row(visible=False),
                gr.Row(visible=False),
                gr.Row(visible=False),
                gr.Slider(value=1.0),
            )
        model, vocoder, denoiser = load_model(model_name, vocoder_name)
        CURRENTLY_LOADED_MODEL = model_name
    CURRENT_LANGUAGE = language

    if model_name == "matcha_ljspeech":
        spk_slider = gr.Slider(visible=False, value=-1)
        single_speaker_examples = gr.Row(visible=True)
        multi_speaker_examples = gr.Row(visible=False)
        japanese_examples = gr.Row(visible=False)
        length_scale = gr.Slider(value=0.95)
    elif model_name == "matcha_jsut":
        spk_slider = gr.Slider(visible=False, value=-1)
        single_speaker_examples = gr.Row(visible=False)
        multi_speaker_examples = gr.Row(visible=False)
        japanese_examples = gr.Row(visible=True)
        length_scale = gr.Slider(value=1.0)
    else:
        spk_slider = gr.Slider(visible=True, value=0)
        single_speaker_examples = gr.Row(visible=False)
        multi_speaker_examples = gr.Row(visible=True)
        japanese_examples = gr.Row(visible=False)
        length_scale = gr.Slider(value=0.85)

    return (
        textbox,
        gr.Button(interactive=True),
        spk_slider,
        single_speaker_examples,
        multi_speaker_examples,
        japanese_examples,
        length_scale,
    )


@torch.inference_mode()
def process_text_gradio(text):
    cleaners = ["japanese_cleaners"] if CURRENT_LANGUAGE == "ja" else None
    output = process_text(1, text, device, cleaners=cleaners, language=CURRENT_LANGUAGE)
    return output["x_phones"][1::2], output["x"], output["x_lengths"]


@torch.inference_mode()
def synthesise_mel(text, text_length, n_timesteps, temperature, length_scale, spk):
    spk = torch.tensor([spk], device=device, dtype=torch.long) if spk >= 0 else None
    output = model.synthesise(
        text,
        text_length,
        n_timesteps=n_timesteps,
        temperature=temperature,
        spks=spk,
        length_scale=length_scale,
    )
    output["waveform"] = to_waveform(output["mel"], vocoder, denoiser)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
        sf.write(fp.name, output["waveform"], 22050, "PCM_24")

    return fp.name, plot_tensor(output["mel"].squeeze().cpu().numpy())


def multispeaker_example_cacher(text, n_timesteps, mel_temp, length_scale, spk):
    global CURRENTLY_LOADED_MODEL, CURRENT_LANGUAGE  # pylint: disable=global-statement
    if CURRENTLY_LOADED_MODEL != "matcha_vctk":
        global model, vocoder, denoiser  # pylint: disable=global-statement
        model, vocoder, denoiser = load_model("matcha_vctk", "hifigan_univ_v1")
        CURRENTLY_LOADED_MODEL = "matcha_vctk"
    CURRENT_LANGUAGE = "en"

    phones, text, text_lengths = process_text_gradio(text)
    audio, mel_spectrogram = synthesise_mel(text, text_lengths, n_timesteps, mel_temp, length_scale, spk)
    return phones, audio, mel_spectrogram


def ljspeech_example_cacher(text, n_timesteps, mel_temp, length_scale, spk=-1):
    global CURRENTLY_LOADED_MODEL, CURRENT_LANGUAGE  # pylint: disable=global-statement
    if CURRENTLY_LOADED_MODEL != "matcha_ljspeech":
        global model, vocoder, denoiser  # pylint: disable=global-statement
        model, vocoder, denoiser = load_model("matcha_ljspeech", "hifigan_T2_v1")
        CURRENTLY_LOADED_MODEL = "matcha_ljspeech"
    CURRENT_LANGUAGE = "en"

    phones, text, text_lengths = process_text_gradio(text)
    audio, mel_spectrogram = synthesise_mel(text, text_lengths, n_timesteps, mel_temp, length_scale, spk)
    return phones, audio, mel_spectrogram


def jsut_example_cacher(text, n_timesteps, mel_temp, length_scale, spk=-1):
    global CURRENTLY_LOADED_MODEL, CURRENT_LANGUAGE  # pylint: disable=global-statement
    if CURRENTLY_LOADED_MODEL != "matcha_jsut":
        model_path = MATCHA_TTS_LOC("matcha_jsut")
        if not model_path.exists():
            gr.Warning("JSUTモデルが見つかりません。先にモデルを学習してください。")
            return "", None, None
        global model, vocoder, denoiser  # pylint: disable=global-statement
        model, vocoder, denoiser = load_model("matcha_jsut", "hifigan_univ_v1")
        CURRENTLY_LOADED_MODEL = "matcha_jsut"
    CURRENT_LANGUAGE = "ja"

    phones, text, text_lengths = process_text_gradio(text)
    audio, mel_spectrogram = synthesise_mel(text, text_lengths, n_timesteps, mel_temp, length_scale, spk)
    return phones, audio, mel_spectrogram


def main():
    description = """# 🍵 Matcha-TTS: 条件付きフローマッチングによる高速TTSアーキテクチャ
    ### [Shivam Mehta](https://www.kth.se/profile/smehta), [Ruibo Tu](https://www.kth.se/profile/ruibo), [Jonas Beskow](https://www.kth.se/profile/beskow), [Éva Székely](https://www.kth.se/profile/szekely), [Gustav Eje Henter](https://people.kth.se/~ghe/)
    🍵 Matcha-TTS は、条件付きフローマッチング（整流フローと類似）を用いてODEベースの音声合成を高速化する、非自己回帰型ニューラルTTSの新しいアプローチです。本手法の特徴：


    * 確率的モデルである
    * コンパクトなメモリフットプリントを持つ
    * 非常に自然な音声を生成する
    * 合成速度が非常に高速である


    [デモページ](https://shivammehta25.github.io/Matcha-TTS)をご覧いただくか、[arXiv論文](https://arxiv.org/abs/2309.03199)をお読みください。
    コードと学習済みモデルは[GitHubリポジトリ](https://github.com/shivammehta25/Matcha-TTS)で公開しています。

    ページ下部にサンプル例があります。
    """

    with gr.Blocks(title="🍵 Matcha-TTS: 条件付きフローマッチングによる高速TTSアーキテクチャ") as demo:
        processed_text = gr.State(value=None)
        processed_text_len = gr.State(value=None)

        with gr.Group():
            with gr.Row():
                gr.Markdown(description)
                with gr.Column():
                    gr.Image(LOGO_URL, label="Matcha-TTS ロゴ", height=50, width=50, scale=1, show_label=False)
                    html = '<br><iframe width="560" height="315" src="https://www.youtube.com/embed/xmvJkz3bqw0?si=jN7ILyDsbPwJCGoa" title="YouTube動画プレーヤー" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>'
                    gr.HTML(html)

        with gr.Group():
            radio_options = list(RADIO_OPTIONS.keys())
            model_type = gr.Radio(
                radio_options, value=radio_options[0], label="モデルを選択", interactive=True, container=False
            )

            with gr.Row():
                gr.Markdown("# テキスト入力")
            with gr.Row():
                text = gr.Textbox(value="", lines=2, label="合成するテキスト", scale=3)
                spk_slider = gr.Slider(
                    minimum=0, maximum=107, step=1, value=args.spk, label="話者ID", interactive=True, scale=1
                )

            with gr.Row():
                gr.Markdown("### ハイパーパラメータ")
            with gr.Row():
                n_timesteps = gr.Slider(
                    label="ODEステップ数",
                    minimum=1,
                    maximum=100,
                    step=1,
                    value=10,
                    interactive=True,
                )
                length_scale = gr.Slider(
                    label="長さスケール（発話速度）",
                    minimum=0.5,
                    maximum=1.5,
                    step=0.05,
                    value=1.0,
                    interactive=True,
                )
                mel_temp = gr.Slider(
                    label="サンプリング温度",
                    minimum=0.00,
                    maximum=2.001,
                    step=0.16675,
                    value=0.667,
                    interactive=True,
                )

                synth_btn = gr.Button("音声合成")

        with gr.Group(), gr.Row():
            gr.Markdown("### 音素化テキスト")
            phonetised_text = gr.Textbox(interactive=False, scale=10, label="音素化テキスト")

        with gr.Group(), gr.Row():
            mel_spectrogram = gr.Image(interactive=False, label="メルスペクトログラム")

            audio = gr.Audio(interactive=False, label="音声")

        with gr.Row(visible=False) as example_row_lj_speech:
            gr.Examples(
                examples=[
                    [
                        "We propose Matcha-TTS, a new approach to non-autoregressive neural TTS, that uses conditional flow matching (similar to rectified flows) to speed up O D E-based speech synthesis.",
                        50,
                        0.677,
                        0.95,
                    ],
                    [
                        "The Secret Service believed that it was very doubtful that any President would ride regularly in a vehicle with a fixed top, even though transparent.",
                        2,
                        0.677,
                        0.95,
                    ],
                    [
                        "The Secret Service believed that it was very doubtful that any President would ride regularly in a vehicle with a fixed top, even though transparent.",
                        4,
                        0.677,
                        0.95,
                    ],
                    [
                        "The Secret Service believed that it was very doubtful that any President would ride regularly in a vehicle with a fixed top, even though transparent.",
                        10,
                        0.677,
                        0.95,
                    ],
                    [
                        "The Secret Service believed that it was very doubtful that any President would ride regularly in a vehicle with a fixed top, even though transparent.",
                        50,
                        0.677,
                        0.95,
                    ],
                    [
                        "The narrative of these events is based largely on the recollections of the participants.",
                        10,
                        0.677,
                        0.95,
                    ],
                    [
                        "The jury did not believe him, and the verdict was for the defendants.",
                        10,
                        0.677,
                        0.95,
                    ],
                ],
                fn=ljspeech_example_cacher,
                inputs=[text, n_timesteps, mel_temp, length_scale],
                outputs=[phonetised_text, audio, mel_spectrogram],
                cache_examples=False,
            )

        with gr.Row() as example_row_multispeaker:
            gr.Examples(
                examples=[
                    [
                        "Hello everyone! I am speaker 0 and I am here to tell you that Matcha-TTS is amazing!",
                        10,
                        0.677,
                        0.85,
                        0,
                    ],
                    [
                        "Hello everyone! I am speaker 16 and I am here to tell you that Matcha-TTS is amazing!",
                        10,
                        0.677,
                        0.85,
                        16,
                    ],
                    [
                        "Hello everyone! I am speaker 44 and I am here to tell you that Matcha-TTS is amazing!",
                        50,
                        0.677,
                        0.85,
                        44,
                    ],
                    [
                        "Hello everyone! I am speaker 45 and I am here to tell you that Matcha-TTS is amazing!",
                        50,
                        0.677,
                        0.85,
                        45,
                    ],
                    [
                        "Hello everyone! I am speaker 58 and I am here to tell you that Matcha-TTS is amazing!",
                        4,
                        0.677,
                        0.85,
                        58,
                    ],
                ],
                fn=multispeaker_example_cacher,
                inputs=[text, n_timesteps, mel_temp, length_scale, spk_slider],
                outputs=[phonetised_text, audio, mel_spectrogram],
                cache_examples=False,
                label="マルチスピーカー サンプル",
            )

        with gr.Row(visible=False) as example_row_japanese:
            gr.Examples(
                examples=[
                    [
                        "ようこそ、音声合成の世界へ。",
                        10,
                        0.677,
                        1.0,
                    ],
                    [
                        "今日はとても良い天気ですね。",
                        10,
                        0.677,
                        1.0,
                    ],
                    [
                        "条件付きフローマッチングを用いた高速な音声合成システムです。",
                        50,
                        0.677,
                        1.0,
                    ],
                    [
                        "吾輩は猫である。名前はまだ無い。",
                        10,
                        0.677,
                        1.0,
                    ],
                    [
                        "東京は日本の首都であり、世界最大の都市圏の一つです。",
                        10,
                        0.677,
                        1.0,
                    ],
                ],
                fn=jsut_example_cacher,
                inputs=[text, n_timesteps, mel_temp, length_scale],
                outputs=[phonetised_text, audio, mel_spectrogram],
                cache_examples=False,
                label="日本語 サンプル",
            )

        model_type.change(lambda _: gr.Button(interactive=False), inputs=[synth_btn], outputs=[synth_btn]).then(
            load_model_ui,
            inputs=[model_type, text],
            outputs=[text, synth_btn, spk_slider, example_row_lj_speech, example_row_multispeaker, example_row_japanese, length_scale],
        )

        synth_btn.click(
            fn=process_text_gradio,
            inputs=[
                text,
            ],
            outputs=[phonetised_text, processed_text, processed_text_len],
            api_name="matcha_tts",
        ).then(
            fn=synthesise_mel,
            inputs=[processed_text, processed_text_len, n_timesteps, mel_temp, length_scale, spk_slider],
            outputs=[audio, mel_spectrogram],
        )

        demo.launch(share=True)


if __name__ == "__main__":
    main()
