# CLAUDE.md

このファイルは、Claude Code (claude.ai/code) がこのリポジトリのコードを扱う際のガイダンスを提供します。

## プロジェクト概要

Matcha-TTS は、条件付きフローマッチングに基づく高速な非自己回帰型テキスト音声合成システムです（ICASSP 2024）。ODEベースのアプローチによりテキストからメルスペクトログラムを生成し、HiFi-GANボコーダを通じて波形に変換します。

## よく使うコマンド

### セットアップ（uv）
```bash
uv sync                      # 依存関係をインストール（Cython拡張を含む）
uv sync --all-groups         # 全開発依存関係をインストール
uv sync --extra app          # Gradio Web UIの依存関係を追加
uv sync --extra onnx         # ONNXサポートの依存関係を追加
```

### 学習
```bash
uv run python matcha/train.py experiment=ljspeech              # 標準的なLJ Speech学習
uv run python matcha/train.py experiment=ljspeech_min_memory    # 省メモリ版
uv run python matcha/train.py experiment=multispeaker           # マルチスピーカー（VCTK）
```
学習にはHydraによる設定合成を使用します。`key=value` 構文で任意のパラメータを上書きできます。

### 推論
```bash
uv run matcha-tts --text "Hello world"                          # CLIによる音声合成
uv run matcha-tts --file input.txt --batched --batch_size 32    # バッチモード
uv run matcha-tts-app                                           # Gradio Web UI
```

### テストとリンティング
```bash
make test           # 高速テストを実行（@pytest.mark.slowをスキップ）
make test-full      # スローテストを含む全テストを実行
make format         # pre-commitフックを実行（black, isort, flake8, pylint）
```
- pytestの設定は `pyproject.toml` にあります（テストディレクトリ: `tests/`）
- Black: 1行120文字、Python 3.10対象
- flake8の除外対象: `logs/*`, `data/*`, `matcha/hifigan/*`

### データユーティリティ
```bash
uv run matcha-data-stats -i ljspeech.yaml                          # メル統計量の計算
uv run matcha-tts-get-durations -i ljspeech.yaml -c model.ckpt     # 継続時間アライメントの抽出
```

### ONNX
```bash
uv run python3 -m matcha.onnx.export model.ckpt output.onnx --n-timesteps 5
uv run python3 -m matcha.onnx.infer output.onnx --text "hello" --output-dir ./outputs
```

## アーキテクチャ

### 音声合成パイプライン
```
Text → cleaners (english_cleaners2) → 音素列 → ブランクの挿入
  → TextEncoder (Conformer + DurationPredictor)
  → 予測された継続時間によりメル長に展開
  → CFM Decoder (Euler ODEソルバー、n_timestepsステップ)
  → メルスペクトログラム
  → HiFi-GAN vocoder → 波形 (22050 Hz)
```

### 主要モジュール

- **`matcha/models/matcha_tts.py`** — メインモデル（PyTorch Lightningモジュール）。エンコーダ、フローマッチング、音声合成を統括します。
- **`matcha/models/components/text_encoder.py`** — Conformerベースのエンコーダと継続時間予測器（ConvReluNorm）。
- **`matcha/models/components/flow_matching.py`** — 条件付きフローマッチング（BASECFM）。Euler ODEソルバーがノイズからメルへの補間を行います。
- **`matcha/models/components/decoder.py`** — ResNetブロック、ダウンサンプリング、Transformerアテンションを備えたU-Net型デコーダ（diffusersライブラリを使用）。
- **`matcha/text/`** — テキストから音素への変換パイプライン。178シンボルの語彙。クリーナーが正規化や数値展開を処理します。
- **`matcha/data/text_mel_datamodule.py`** — テキスト＋音声ファイルリストの読み込みとメルスペクトログラムの計算を行うLightning DataModule。
- **`matcha/hifigan/`** — HiFi-GANボコーダ（事前学習済み、個別に読み込み）。オプションのデノイザーを含みます。
- **`matcha/utils/monotonic_align/`** — Cython高速化された単調アライメント探索（MAS）。`uv sync` 時にコンパイルされます。

### 学習損失

3つの損失の合計: **継続時間損失**（予測された継続時間に対するMSE）、**事前分布損失**（KLダイバージェンス）、**フローマッチング損失**（メルに対するデノイジング目的関数）。

### 設定システム
Hydraの設定ファイルは `configs/` にあります。主な合成構造: `train.yaml` が `data/`, `model/`, `trainer/`, `callbacks/`, `logger/`, `optimizer/`, `scheduler/` から設定を取得します。実験ファイル（例: `experiment/ljspeech.yaml`）がデフォルト値を上書きします。

### 主要パラメータ
- `n_feats: 80`（メルビン数）、`sample_rate: 22050`、`hop_length: 256`、`n_fft: 1024`
- `n_vocab: 178`（音素語彙サイズ）
- `data_statistics`: zスコア正規化のための事前計算された `mel_mean`/`mel_std`
- 推論制御: `n_timesteps`（ODEステップ数）、`temperature`（ノイズ分散）、`length_scale`（発話速度）
