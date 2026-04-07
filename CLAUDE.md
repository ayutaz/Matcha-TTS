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

### JVSデータ準備（新環境セットアップ）
```bash
# 1. JVSデータセットの準備（無音トリミング付きリサンプリング）
uv run python scripts/prepare_jvs.py --jvs-dir /path/to/jvs_ver1 --output-dir data/jvs --num-workers 8

# 2. メルスペクトログラム事前計算
uv run python scripts/precompute_dataset.py \
  --filelist data/jvs/train.txt --output-dir data/jvs_precomputed/train \
  --mel-mean -6.550095 --mel-std 2.383771 --num-workers 8
uv run python scripts/precompute_dataset.py \
  --filelist data/jvs/val.txt --output-dir data/jvs_precomputed/val \
  --mel-mean -6.550095 --mel-std 2.383771 --num-workers 8

# 3. /dev/shmにコピー（高速I/O、任意）
cp -r data/jvs_precomputed /dev/shm/jvs_precomputed
```

### 学習
```bash
# 英語（LJSpeech）
uv run python matcha/train.py experiment=ljspeech              # 標準的なLJ Speech学習
uv run python matcha/train.py experiment=ljspeech_min_memory    # 省メモリ版
uv run python matcha/train.py experiment=multispeaker           # マルチスピーカー（VCTK）

# 日本語（JVS — 事前計算済み特徴量使用、推奨設定）
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python matcha/train.py \
  experiment=jvs_fast compile_model=false \
  data.batch_size=32 data.num_workers=0 +data.preload_to_memory=true \
  test=false

# チェックポイントからの再開
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python matcha/train.py \
  experiment=jvs_fast compile_model=false \
  data.batch_size=32 data.num_workers=0 +data.preload_to_memory=true \
  test=false ckpt_path=logs/train/jvs_fast/runs/<run_dir>/checkpoints/last.ckpt
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

# JVSデータセットの準備（無音トリミング付き）
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

- **`matcha/models/matcha_tts.py`** — メインモデル（PyTorch Lightningモジュール）。エンコーダ、フローマッチング、音声合成を統括。LOG_2PI定数キャッシュ。out_size切り出しにはtorch.zeros使用（torch.emptyはNaN発散の原因）。
- **`matcha/models/components/text_encoder.py`** — Conformerベースのエンコーダ。RoPE（静的キャッシュ、max_seq_len=2048）、SDPA attention、speaker embedding expand最適化。
- **`matcha/models/components/flow_matching.py`** — 条件付きフローマッチング（BASECFM）。Euler/Midpoint ODEソルバー。compute_lossでfloat()キャスト（FP16 overflow防止）。timestep samplingは一様分布（logit-normalは品質劣化の原因）。
- **`matcha/models/components/decoder.py`** — U-Net型デコーダ。SinusoidalPosEmbキャッシュ（register_buffer）、einops完全除去（torch native ops使用）、gradient checkpointing対応（Transformerブロックのみ）。
- **`matcha/text/`** — テキストから音素への変換パイプライン。英語178シンボル / 日本語55シンボル。LRUキャッシュ（16,384エントリ）。
- **`matcha/text/cleaners.py`** — `english_cleaners2`（espeak-ng）と `japanese_cleaners`（pyopenjtalk）。
- **`matcha/data/text_mel_datamodule.py`** — テキスト＋音声ファイルリストのLightning DataModule。drop_last=True（DDP対応）。
- **`matcha/data/precomputed_datamodule.py`** — 事前計算済み.ptファイル用DataModule。os.scandirによる高速列挙、ファイルサイズキャッシュ、BucketBatchSampler（単一GPU時）、DistributedBucketBatchSampler（DDP時）、preload_to_memoryオプション。
- **`matcha/hifigan/`** — HiFi-GANボコーダ（事前学習済み）。推論時はweight_norm除去済み。
- **`matcha/utils/monotonic_align/`** — MAS。CUDA入力時はPyTorch GPU実装（torch.jit.script）、CPU時はCythonフォールバック。

### 学習損失

3つの損失の合計: **継続時間損失**（予測された継続時間に対するMSE）、**事前分布損失**（KLダイバージェンス + LOG_2PI）、**フローマッチング損失**（メルに対するデノイジング目的関数）。全て重み1.0で均等加算（原論文準拠）。

### 設定システム
Hydraの設定ファイルは `configs/` にあります。主な合成構造: `train.yaml` が `data/`, `model/`, `trainer/`, `callbacks/`, `logger/`, `optimizer/`, `scheduler/` から設定を取得します。実験ファイル（例: `experiment/ljspeech.yaml`）がデフォルト値を上書きします。

### 主要パラメータ
- `n_feats: 80`（メルビン数）、`sample_rate: 22050`、`hop_length: 256`、`n_fft: 1024`
- `n_vocab: 178`（英語音素語彙）/ `n_vocab: 55`（日本語音素語彙）
- `data_statistics`: zスコア正規化のための事前計算された `mel_mean`/`mel_std`
  - JVS（トリミング済み）: `mel_mean: -6.550095`, `mel_std: 2.383771`
- 推論制御: `n_timesteps`（ODEステップ数、デフォルト5）、`temperature`（ノイズ分散、CLIデフォルト0.667）、`length_scale`（発話速度）

## JVS日本語学習の重要な知見

### 学習設定（確認済みの安定構成）
- **FP32精度**: T4/V100ではFP16が学習不安定を引き起こす（diff_lossスパイク、NaN発散）
- **out_size=null**: 全長メル学習が必須。`out_size=172`はDecoder/Duration Predictorの品質不足を招く
- **原論文準拠のLoss**: prior重み1.0、LOG_2PI復元、MSE duration loss。変更すると品質劣化
- **原論文準拠のLR**: `lr=1e-4`、scheduler=なし、`weight_decay=0.0`。高LRは不安定
- **一様分布timestep sampling**: logit-normalはTTSでは検証不足で品質劣化の原因
- **EMA**: `decay=0.9995`、`update_starting_at_epoch=10`

### 過去に失敗した最適化（適用しないこと）
- **FP16 Mixed Precision**: Loss計算にFP32キャスト追加してもDuration Predictor品質が劣化（音素あたり2.4フレーム vs 正解7.2フレーム）。encoder/decoder内部表現のFP16精度不足が原因。NaN防止だけでは不十分（2026-04-06検証済み）
- **LRスケジューラ（warmup + cosine decay）**: 原論文はlr=1e-4固定。cosine decayはDuration Predictorの収束を妨げる可能性
- **Logit-Normal timestep sampling**: t≈0,1の学習不足 → decoder出力に正バイアス
- **out_size=172**: Duration Predictorが内部音素に1-1.5フレームしか割り当てず発音不明瞭
- **prior_loss重み0.5**: encoder mu_y品質低下 → decoder条件付け劣化の連鎖
- **torch.empty（out_size切り出し）**: 未初期化メモリでNaN混入 → torch.zeros必須
- **LR 5e-4 + weight_decay=0.01**: 原論文より攻撃的すぎて不安定

### JVSデータの注意点
- **無音トリミング必須**: JVSコーパスは各発話の先頭/末尾に~500msの無音を含む。`prepare_jvs.py`で自動トリミング（`top_db=30`、50msマージン）
- **mel統計量**: トリミング後のデータで再計算が必要（`mel_mean: -6.550095`, `mel_std: 2.383771`）
- **blank[0] Duration爆発**: Duration Predictorが先頭blankトークンに異常に大きなdurationを予測する場合がある。推論時のclampで対症的に対応可能だが、根本的にはデータのトリミングと十分な学習が必要
- **torchaudio非互換**: PyTorch 2.10+ではtorchcodec依存でtorchaudio.loadが失敗する場合あり。`soundfile`をフォールバックとして使用

## パフォーマンス最適化

### 学習最適化
- **Fused AdamW**: `fused=True`でオプティマイザステップ高速化（ただしFP16+gradient clippingとは非互換）
- **FP32精度**: V100/T4ではFP16が学習不安定のため、FP32をデフォルト使用
- **Gradient Checkpointing**: デコーダのTransformerブロックのみ（ResNetブロックは除外して計算効率化）
- **DDP最適化**: `gradient_as_bucket_view=true`、`bucket_cap_mb=25`（通信overlap有効化）、`broadcast_buffers=false`、NCCLタイムアウト7200秒
- **ログ最適化**: `sync_dist=False`（ステップレベル）、`log_dict()`統合でDDPオーバーヘッド削減
- **データ読込**: os.scandir（NFS 12倍高速）、ファイルサイズキャッシュ、drop_last=True、preload_to_memoryオプション
- **チェックポイント**: `save_on_train_epoch_end=true`で確実に保存（validation非依存）

### 推論最適化
- **torch.compile**: CUDA時にencoder/decoder/vocoder自動コンパイル（`reduce-overhead`モード）
- **テキスト並列処理**: ThreadPoolExecutorによるバッチテキスト処理
- **ONNX INT8量子化**: `--quantize`フラグで動的量子化サポート
- **GPU warmup**: 推論前のCUDAカーネル事前ロード（マルチスピーカー対応済み）

### デコーダ最適化
- einops完全除去 → `transpose`/`squeeze`/`cat`/`expand`（不要コピー12+回削減）
- SinusoidalPosEmbの`register_buffer`キャッシュ（ODE各ステップの再計算回避）
- `torch.cat`によるテンソル結合（einops pack除去）

### 4GPU DDP学習の注意事項
- `compile_model=false`が必要（gradient checkpointingとの非互換）
- `static_graph=false`が必要（同上）
- `out_size=null`の場合 `data.batch_size=32`が安定上限
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`推奨
- NCCLタイムアウトは7200秒に設定済み（NFS I/O遅延対策）
- `use_distributed_sampler=false`でカスタムDistributedBucketBatchSamplerを使用
- `sync_batchnorm=false`（モデルにBatchNormなし）
