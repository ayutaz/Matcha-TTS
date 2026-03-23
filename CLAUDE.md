# CLAUDE.md

このファイルは、Claude Code (claude.ai/code) がこのリポジトリのコードを扱う際のガイダンスを提供します。

## プロジェクト概要

Matcha-TTS は、条件付きフローマッチングに基づく高速な非自己回帰型テキスト音声合成システムです（ICASSP 2024）。ODEベースのアプローチによりテキストからメルスペクトログラムを生成し、HiFi-GANボコーダを通じて波形に変換します。

**日本語サポート**: JVSコーパス（100話者）による日本語音声合成に対応。pyopenjtalkによるフルコンテキストラベルからの音素変換、55シンボルの日本語語彙テーブルを実装。

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
# 英語（LJSpeech）
uv run python matcha/train.py experiment=ljspeech              # 標準的なLJ Speech学習
uv run python matcha/train.py experiment=ljspeech_min_memory    # 省メモリ版
uv run python matcha/train.py experiment=multispeaker           # マルチスピーカー（VCTK）

# 日本語（JVS — 事前計算済み特徴量使用）
uv run python matcha/train.py experiment=jvs_fast data.batch_size=64 trainer.max_epochs=200

# チェックポイントからの再開
uv run python matcha/train.py experiment=jvs_fast data.batch_size=64 trainer.max_epochs=200 \
  ckpt_path=logs/train/jvs_fast/runs/<run_dir>/checkpoints/last.ckpt

# 4GPU DDP学習（jvs_fastはddp_optimizedトレーナーを使用）
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python matcha/train.py \
  experiment=jvs_fast compile_model=false data.batch_size=64 data.num_workers=4 \
  trainer.max_epochs=200
```
学習にはHydraによる設定合成を使用します。`key=value` 構文で任意のパラメータを上書きできます。

### 推論
```bash
uv run matcha-tts --text "Hello world"                          # CLIによる音声合成（英語）
uv run matcha-tts --text "こんにちは" --language ja              # CLIによる音声合成（日本語）
uv run matcha-tts --file input.txt --batched --batch_size 32    # バッチモード
uv run matcha-tts-app                                           # Gradio Web UI
```
推論時はCUDA環境で自動的にtorch.compile（encoder/decoder/vocoder）が適用されます。

### テストとリンティング
```bash
make test           # 高速テストを実行（@pytest.mark.slowをスキップ）
make test-full      # スローテストを含む全テストを実行
make format         # pre-commitフックを実行（ruff）
uv run ruff check . # リンターチェック
```
- pytestの設定は `pyproject.toml` にあります（テストディレクトリ: `tests/`、256テスト）
- ruff: リンティングとフォーマット

### データ前処理
```bash
# メル統計量の計算
uv run matcha-data-stats -i ljspeech.yaml

# JVSデータセットの事前計算（GPU使用）
uv run python scripts/precompute_dataset.py --config configs/data/jvs.yaml --gpu

# JVSデータセットの準備
uv run python scripts/prepare_jvs.py --jvs-dir /path/to/jvs --output-dir data/jvs --num-workers 8

# 英語→日本語モデル転移
uv run python scripts/transfer_from_english.py --source model.ckpt --target ja_model.ckpt --n-vocab-new 55
```

### ONNX
```bash
uv run python3 -m matcha.onnx.export model.ckpt output.onnx --n-timesteps 5
uv run python3 -m matcha.onnx.export model.ckpt output.onnx --n-timesteps 5 --quantize  # INT8量子化
uv run python3 -m matcha.onnx.infer output.onnx --text "hello" --output-dir ./outputs
uv run python3 -m matcha.onnx.infer output.onnx --quantized --text "hello"              # INT8モデル使用
```

## アーキテクチャ

### 音声合成パイプライン
```
Text → cleaners (english_cleaners2 / japanese_cleaners) → 音素列 → ブランクの挿入
  → TextEncoder (Conformer + RoPE + DurationPredictor)
  → 予測された継続時間によりメル長に展開
  → CFM Decoder (Euler/Midpoint ODEソルバー、n_timestepsステップ)
  → メルスペクトログラム
  → HiFi-GAN vocoder → 波形 (22050 Hz)
```

### 主要モジュール

- **`matcha/models/matcha_tts.py`** — メインモデル（PyTorch Lightningモジュール）。エンコーダ、フローマッチング、音声合成を統括。LOG_2PI定数キャッシュ、torch.empty最適化済み。
- **`matcha/models/components/text_encoder.py`** — Conformerベースのエンコーダ。RoPE（静的キャッシュ、max_seq_len=2048）、SDPA attention、speaker embedding expand最適化。
- **`matcha/models/components/flow_matching.py`** — 条件付きフローマッチング（BASECFM）。Euler/Midpoint ODEソルバー。one_minus_sigma_minキャッシュ、solリスト除去済み。
- **`matcha/models/components/decoder.py`** — U-Net型デコーダ。SinusoidalPosEmbキャッシュ（register_buffer）、einops完全除去（torch native ops使用）、gradient checkpointing対応。
- **`matcha/text/`** — テキストから音素への変換パイプライン。英語178シンボル / 日本語55シンボル。LRUキャッシュ（16,384エントリ）。
- **`matcha/text/cleaners.py`** — `english_cleaners2`（espeak-ng）と `japanese_cleaners`（pyopenjtalk）。
- **`matcha/data/text_mel_datamodule.py`** — テキスト＋音声ファイルリストのLightning DataModule。drop_last=True（DDP対応）。
- **`matcha/data/precomputed_datamodule.py`** — 事前計算済み.ptファイル用DataModule。os.scandirによる高速列挙、ファイルサイズキャッシュ、BucketBatchSampler（単一GPU時）。
- **`matcha/hifigan/`** — HiFi-GANボコーダ（事前学習済み）。推論時はweight_norm除去済み。
- **`matcha/utils/monotonic_align/`** — MAS。CUDA入力時はPyTorch GPU実装（torch.jit.script）、CPU時はCythonフォールバック。

### 学習損失

3つの損失の合計: **継続時間損失**（予測された継続時間に対するMSE）、**事前分布損失**（KLダイバージェンス）、**フローマッチング損失**（メルに対するデノイジング目的関数）。

### 設定システム
Hydraの設定ファイルは `configs/` にあります。主な合成構造: `train.yaml` が `data/`, `model/`, `trainer/`, `callbacks/`, `logger/`, `optimizer/`, `scheduler/` から設定を取得します。実験ファイル（例: `experiment/ljspeech.yaml`）がデフォルト値を上書きします。

### 主要パラメータ
- `n_feats: 80`（メルビン数）、`sample_rate: 22050`、`hop_length: 256`、`n_fft: 1024`
- `n_vocab: 178`（英語音素語彙）/ `n_vocab: 55`（日本語音素語彙）
- `data_statistics`: zスコア正規化のための事前計算された `mel_mean`/`mel_std`
- 推論制御: `n_timesteps`（ODEステップ数、デフォルト5）、`temperature`（ノイズ分散）、`length_scale`（発話速度）

## パフォーマンス最適化

### 学習最適化
- **Fused AdamW**: `fused=True`でオプティマイザステップ高速化
- **FP32精度**: V100/T4ではFP16が逆効果のため、FP32をデフォルト使用
- **Gradient Checkpointing**: デコーダのメモリ使用量を30-50%削減
- **DDP最適化**: `gradient_as_bucket_view=true`、`bucket_cap_mb=100`、`broadcast_buffers=false`、NCCLタイムアウト7200秒
- **ログ最適化**: `sync_dist=False`（ステップレベル）、`log_dict()`統合でDDPオーバーヘッド削減
- **データ読込**: os.scandir（NFS 12倍高速）、ファイルサイズキャッシュ、drop_last=True

### 推論最適化
- **torch.compile**: CUDA時にencoder/decoder/vocoder自動コンパイル（`reduce-overhead`モード）
- **テキスト並列処理**: ThreadPoolExecutorによるバッチテキスト処理
- **ONNX INT8量子化**: `--quantize`フラグで動的量子化サポート
- **GPU warmup**: 推論前のCUDAカーネル事前ロード

### デコーダ最適化
- einops完全除去 → `transpose`/`squeeze`/`cat`/`expand`（不要コピー12+回削減）
- SinusoidalPosEmbの`register_buffer`キャッシュ（ODE各ステップの再計算回避）
- `torch.cat`によるテンソル結合（einops pack除去）

### 4GPU DDP学習の注意事項
- `compile_model=false`が必要（gradient checkpointingとの非互換）
- `static_graph=false`が必要（同上）
- `data.batch_size=64`が安定上限（可変長データのピークVRAMに注意）
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`推奨
- NCCLタイムアウトは7200秒に設定済み（NFS I/O遅延対策）
