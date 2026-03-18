<div align="center">

# Matcha-TTS: 条件付きフローマッチングによる高速TTSアーキテクチャ

### [Shivam Mehta](https://www.kth.se/profile/smehta), [Ruibo Tu](https://www.kth.se/profile/ruibo), [Jonas Beskow](https://www.kth.se/profile/beskow), [Eva Szekely](https://www.kth.se/profile/szekely), [Gustav Eje Henter](https://people.kth.se/~ghe/)

[![python](https://img.shields.io/badge/-Python_3.10-blue?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3100/)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/matcha-tts?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/matcha-tts)
<p style="text-align: center;">
  <img src="https://shivammehta25.github.io/Matcha-TTS/images/logo.png" height="128"/>
</p>

</div>

> これは Matcha-TTS [ICASSP 2024] の公式コード実装です。

Matcha-TTS は、ODEベースの音声合成を高速化するために[条件付きフローマッチング](https://arxiv.org/abs/2210.02747)（[整流フロー](https://arxiv.org/abs/2209.03003)と類似）を使用する、非自己回帰型ニューラルTTSの新しいアプローチです。本手法の特徴：

- 確率的モデルである
- コンパクトなメモリフットプリントを持つ
- 非常に自然な音声を生成する
- 合成速度が非常に高速である

詳細については、[デモページ](https://shivammehta25.github.io/Matcha-TTS)をご覧いただくか、[ICASSP 2024論文](https://arxiv.org/abs/2309.03199)をお読みください。

[学習済みモデル](https://drive.google.com/drive/folders/17C_gYgEHOxI5ZypcfE_k1piKCtyR0isJ?usp=sharing)は、CLIまたはGradioインターフェースで自動的にダウンロードされます。

[HuggingFace Spacesでブラウザ上からMatcha-TTSを試す](https://huggingface.co/spaces/shivammehta25/Matcha-TTS)こともできます。

## ティーザー動画

[![動画を見る](https://img.youtube.com/vi/xmvJkz3bqw0/hqdefault.jpg)](https://youtu.be/xmvJkz3bqw0)

## インストール

### uvを使用する場合（推奨）

1. [uv](https://docs.astral.sh/uv/)をインストールする

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. リポジトリをクローンして依存関係をインストールする

```bash
git clone https://github.com/shivammehta25/Matcha-TTS.git
cd Matcha-TTS
uv sync
```

3. 特定のオプション依存関係が必要な場合

```bash
uv sync --extra app          # Gradio Web UI
uv sync --extra onnx         # ONNXサポート
uv sync --all-groups         # 全開発依存関係
```

### pipを使用する場合

```bash
pip install matcha-tts
```

ソースからインストールする場合

```bash
pip install git+https://github.com/shivammehta25/Matcha-TTS.git
cd Matcha-TTS
pip install -e .
```

3. CLI / Gradioアプリ / Jupyterノートブックを実行する

```bash
# 必要なモデルが自動的にダウンロードされます
matcha-tts --text "<INPUT TEXT>"
```

または

```bash
matcha-tts-app
```

またはJupyterノートブックで `synthesis.ipynb` を開いてください

### CLI引数

- 指定したテキストから合成するには、以下を実行します：

```bash
matcha-tts --text "<INPUT TEXT>"
```

- ファイルから合成するには、以下を実行します：

```bash
matcha-tts --file <PATH TO FILE>
```

- ファイルからバッチ合成するには、以下を実行します：

```bash
matcha-tts --file <PATH TO FILE> --batched
```

追加の引数

- 発話速度

```bash
matcha-tts --text "<INPUT TEXT>" --speaking_rate 1.0
```

- サンプリング温度

```bash
matcha-tts --text "<INPUT TEXT>" --temperature 0.667
```

- オイラーODEソルバーのステップ数

```bash
matcha-tts --text "<INPUT TEXT>" --steps 10
```

## 独自のデータセットで学習する

ここではLJ Speechを使用して学習する場合を想定します

1. [こちら](https://keithito.com/LJ-Speech-Dataset/)からデータセットをダウンロードし、`data/LJSpeech-1.1` に展開します。そして、[NVIDIA Tacotron 2リポジトリのセットアップ手順5](https://github.com/NVIDIA/tacotron2#setup)のように、展開したデータを指すファイルリストを準備します。

2. Matcha-TTSリポジトリをクローンして移動する

```bash
git clone https://github.com/shivammehta25/Matcha-TTS.git
cd Matcha-TTS
```

3. ソースからパッケージをインストールする

```bash
uv sync  # または pip install -e .
```

4. `configs/data/ljspeech.yaml` を開き、以下を変更する

```yaml
train_filelist_path: data/filelists/ljs_audio_text_train_filelist.txt
valid_filelist_path: data/filelists/ljs_audio_text_val_filelist.txt
```

5. データセット設定のyamlファイルを使用して正規化統計量を生成する

```bash
matcha-data-stats -i ljspeech.yaml
# Output:
#{'mel_mean': -5.53662231756592, 'mel_std': 2.1161014277038574}
```

これらの値を `configs/data/ljspeech.yaml` の `data_statistics` キーに更新します。

```bash
data_statistics:  # Computed for ljspeech dataset
  mel_mean: -5.536622
  mel_std: 2.116101
```

学習用および検証用ファイルリストのパスを指定してください。

6. 学習スクリプトを実行する

```bash
make train-ljspeech
```

または

```bash
python matcha/train.py experiment=ljspeech
```

- 最小メモリで実行する場合

```bash
python matcha/train.py experiment=ljspeech_min_memory
```

- マルチGPUで学習する場合

```bash
python matcha/train.py experiment=ljspeech trainer.devices=[0,1]
```

7. カスタム学習済みモデルから合成する

```bash
matcha-tts --text "<INPUT TEXT>" --checkpoint_path <PATH TO CHECKPOINT>
```

## ONNXサポート

> ONNXエクスポートおよび推論サポートの実装について、[@mush42](https://github.com/mush42) 氏に特別な感謝を申し上げます。

Matchaのチェックポイントを[ONNX](https://onnx.ai/)にエクスポートし、エクスポートされたONNXグラフ上で推論を実行することが可能です。

### ONNXエクスポート

チェックポイントをONNXにエクスポートするには、まずONNXをインストールします

```bash
uv sync --extra onnx  # または pip install onnx
```

次に、以下を実行します：

```bash
python3 -m matcha.onnx.export matcha.ckpt model.onnx --n-timesteps 5
```

オプションとして、ONNXエクスポーターは **vocoder-name** および **vocoder-checkpoint** 引数を受け付けます。これにより、エクスポートされたグラフにボコーダーを埋め込み、エンドツーエンドTTSシステムと同様に、単一の実行で波形を生成できます。

**注意**: `n_timesteps` はモデル入力ではなくハイパーパラメータとして扱われます。つまり、推論時ではなくエクスポート時に指定する必要があります。指定しない場合、`n_timesteps` は **5** に設定されます。

**重要**: 現時点では、`scaled_product_attention` 演算子が古いバージョンではエクスポートできないため、エクスポートにはtorch>=2.1.0が必要です。最終バージョンがリリースされるまで、モデルをエクスポートしたい方はtorch>=2.1.0をプレリリース版として手動でインストールする必要があります。

### ONNX推論

エクスポートされたモデルで推論を実行するには、まず `onnxruntime` をインストールします

```bash
uv sync --extra onnx      # または pip install onnxruntime
uv sync --extra onnx-gpu  # GPU推論用（または pip install onnxruntime-gpu）
```

次に、以下を使用します：

```bash
python3 -m matcha.onnx.infer model.onnx --text "hey" --output-dir ./outputs
```

合成パラメータを制御することもできます：

```bash
python3 -m matcha.onnx.infer model.onnx --text "hey" --output-dir ./outputs --temperature 0.4 --speaking_rate 0.9 --spk 0
```

**GPU**で推論を実行するには、**onnxruntime-gpu** パッケージがインストールされていることを確認し、推論コマンドに `--gpu` を渡します：

```bash
python3 -m matcha.onnx.infer model.onnx --text "hey" --output-dir ./outputs --gpu
```

MatchaのみをONNXにエクスポートした場合、メルスペクトログラムがグラフおよび `numpy` 配列として出力ディレクトリに書き込まれます。
エクスポートされたグラフにボコーダーを埋め込んだ場合、`.wav` 音声ファイルが出力ディレクトリに書き込まれます。

MatchaのみをONNXにエクスポートし、フルTTSパイプラインを実行したい場合は、`ONNX` 形式のボコーダーモデルのパスを渡すことができます：

```bash
python3 -m matcha.onnx.infer model.onnx --text "hey" --output-dir ./outputs --vocoder hifigan.small.onnx
```

これにより、`.wav` 音声ファイルが出力ディレクトリに書き込まれます。

## Matcha-TTSからの音素アラインメント抽出

データセットが以下の構造になっている場合

```bash
data/
└── LJSpeech-1.1
    ├── metadata.csv
    ├── README
    ├── test.txt
    ├── train.txt
    ├── val.txt
    └── wavs
```
学習済みのMatcha-TTSモデルから音素レベルのアラインメントを以下のように抽出できます：
```bash
python  matcha/utils/get_durations_from_trained_model.py -i dataset_yaml -c <checkpoint>
```
例：
```bash
python  matcha/utils/get_durations_from_trained_model.py -i ljspeech.yaml -c matcha_ljspeech.ckpt
```
または簡単に：
```bash
matcha-tts-get-durations -i ljspeech.yaml -c matcha_ljspeech.ckpt
```
---
## 抽出したアラインメントを使用して学習する

データセット設定でload durationを有効にします。
例: `ljspeech.yaml`
```
load_durations: True
```
または configs/experiment/ljspeech_from_durations.yaml の例を参照してください


## 引用情報

本コードを使用した場合、または本研究が有用であった場合は、以下の論文を引用してください：

```text
@inproceedings{mehta2024matcha,
  title={Matcha-{TTS}: A fast {TTS} architecture with conditional flow matching},
  author={Mehta, Shivam and Tu, Ruibo and Beskow, Jonas and Sz{\'e}kely, {\'E}va and Henter, Gustav Eje},
  booktitle={Proc. ICASSP},
  year={2024}
}
```

## 謝辞

本コードは [Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template) を使用しているため、そのすべての機能を活用できます。

その他、感謝を表したいソースコード：

- [Coqui-TTS](https://github.com/coqui-ai/TTS/tree/dev): Cythonバイナリをpipでインストール可能にする方法を理解する上での助力と励まし
- [Hugging Face Diffusers](https://huggingface.co/): 優れたDiffusersライブラリとそのコンポーネント
- [Grad-TTS](https://github.com/huawei-noah/Speech-Backbones/tree/main/Grad-TTS): 単調アラインメント探索のソースコード
- [torchdyn](https://github.com/DiffEqML/torchdyn): 研究開発中に他のODEソルバーを試すのに有用
- [labml.ai](https://nn.labml.ai/transformers/rope/index.html): RoPE実装
