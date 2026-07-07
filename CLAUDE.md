# CLAUDE.md

このファイルは、Claude Code (claude.ai/code) がこのリポジトリのコードを扱う際のガイダンスを提供します。

## プロジェクト概要

Matcha-TTS は、条件付きフローマッチングに基づく高速な非自己回帰型テキスト音声合成システムです（ICASSP 2024）。ODEベースのアプローチによりテキストからメルスペクトログラムを生成し、HiFi-GANボコーダを通じて波形に変換します。

**日本語サポート**: JVSコーパス（100話者）による日本語音声合成に対応。pyopenjtalk-plus（本体依存）によるフルコンテキストラベルからの音素変換、55シンボルの日本語語彙テーブルを実装。onnxruntimeも本体依存に含み、「何」の読み分け（Nani prediction）が有効。

**MARINE（DNNアクセント推定、`run_marine=True`）は調査の上不採用**（2026-07決定）: 公開モデルはJSUTのみ学習で論文精度（80.4%）に届かず、pyopenjtalk-plusメンテナー自身が「ルールベースの方が精度が高い傾向」と明言。JVS実テキストでAssertionErrorクラッシュを確認、モデルロード~24秒、numpy 1.x固定の制約もある。Style-BERT-VITS2 / ESPnet / VOICEVOX等の主要日本語TTSも全て非採用。アクセント誤りが問題になった場合はユーザー辞書で対処する（推論時の入力補正であり再学習不要）。

## よく使うコマンド

### セットアップ（uv）
```bash
uv sync                      # 依存関係をインストール（Cython拡張・pyopenjtalk-plusを含む）
uv sync --all-groups         # 全開発依存関係をインストール
uv sync --extra app          # Gradio Web UIの依存関係を追加
uv sync --extra onnx         # ONNXエクスポートの依存関係を追加
```
- Pythonは**3.12固定**（`requires-python = ">=3.12,<3.13"`、`.python-version`）。uvが自動で3.12を取得する
- 日本語g2p（pyopenjtalk-plus + onnxruntime）は本体依存のため `uv sync` だけで入る（旧 `--extra japanese` は廃止）

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

### JVSデータ準備（最適化版パイプライン）
```bash
# 1. JVSデータセットの準備 + Julius用16kHz同時出力
uv run python scripts/prepare_jvs.py --jvs-dir /path/to/jvs_ver1 --output-dir data/jvs \
  --julius-output-dir data/julius_work/wav --num-workers 8

# 2. /dev/shmキャッシュセットアップ（高速I/O）
bash scripts/setup_shm_cache.sh --full

# 3. 最適化パイプライン実行（Julius並列化 + fast CPU precompute、~3分）
uv run python scripts/run_optimized_pipeline.py \
  --filelist data/jvs/train.txt data/jvs/val.txt \
  --output-dir /dev/shm/julius_work \
  --pt-output-dir /dev/shm/jvs_precomputed_aligned \
  --mel-mean -6.550095 --mel-std 2.383771 \
  --num-workers 16 --use-shm

# 4. NFSにバックアップ
cp -r /dev/shm/jvs_precomputed_aligned data/jvs_precomputed_aligned
```

デフォルトで fast CPU path（Tier 3）を使用。precompute がさらに速くなり、Step 3+4 が 25.2分 → 1.2分 (20.6倍) に短縮されます。

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

# 日本語（JVS — 外部アライナーduration使用、MASバイパス）
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python matcha/train.py \
  experiment=jvs_aligned compile_model=false \
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
uv run python scripts/precompute_dataset.py --filelist data/jvs/train.txt --output-dir data/jvs_precomputed/train --gpu

# JVSデータセットの準備（無音トリミング付き）
uv run python scripts/prepare_jvs.py --jvs-dir /path/to/jvs --output-dir data/jvs --num-workers 8

# 英語→日本語モデル転移
uv run python scripts/transfer_from_english.py --source model.ckpt --target ja_model.ckpt --n-vocab-new 55

# JVS最適化パイプライン（Julius並列化 + fast CPU precompute、~3分）
uv run python scripts/run_optimized_pipeline.py \
  --filelist data/jvs/train.txt data/jvs/val.txt \
  --output-dir /dev/shm/julius_work \
  --pt-output-dir /dev/shm/jvs_precomputed_aligned \
  --mel-mean -6.550095 --mel-std 2.383771 \
  --num-workers 16 --use-shm

# precompute fast path のチューニング（必要時のみ）
# デフォルト: --precompute-device cpu --precompute-batch-size 32 --precompute-io-workers 16
uv run python scripts/run_optimized_pipeline.py \
  --filelist data/jvs/train.txt data/jvs/val.txt \
  --output-dir /dev/shm/julius_work \
  --pt-output-dir /dev/shm/jvs_precomputed_aligned \
  --mel-mean -6.550095 --mel-std 2.383771 \
  --num-workers 16 --use-shm \
  --precompute-device cpu --precompute-io-workers 32  # I/O 強化

# 旧 ProcessPoolExecutor 経路（デバッグ/互換性用途）
uv run python scripts/run_optimized_pipeline.py \
  --filelist data/jvs/train.txt data/jvs/val.txt \
  --output-dir /dev/shm/julius_work \
  --pt-output-dir /dev/shm/jvs_precomputed_aligned \
  --mel-mean -6.550095 --mel-std 2.383771 \
  --num-workers 16 --use-shm --precompute-legacy-path
```

### ONNX
```bash
uv run python3 -m matcha.onnx.export model.ckpt output.onnx --n-timesteps 5
uv run python3 -m matcha.onnx.export model.ckpt output.onnx --n-timesteps 5 --quantize  # INT8量子化
uv run python3 -m matcha.onnx.infer output.onnx --text "hello" --output-dir ./outputs
uv run python3 -m matcha.onnx.infer output.onnx --quantized --text "hello"              # INT8モデル使用

# 日本語モデル + WaveNeXtボコーダを単一グラフに埋め込み（CPU/モバイル配布向け、iSTFT無し）
uv run python3 -m matcha.onnx.export jvs_aligned.ckpt jvs_wavenext.onnx --n-timesteps 5 \
  --vocoder-name wavenext --vocoder-checkpoint-path <BSC-LT/wavenext-mel pytorch_model.bin>
```
- **依存**: `uv sync --extra onnx`（`onnx` + `onnxruntime` + `onnxscript`）。torch≥2.9は`torch.onnx.export`を
  dynamo経路に流すが、Matchaのsynthesise() SymInt indexingで失敗するため`export.py`は`dynamo=False`で
  旧TorchScript exporter（opset17）に固定している
- **WaveNeXtボコーダ**（`--vocoder-name wavenext`、`matcha/wavenext/`）: iSTFT無しのConvNeXt+線形ヘッドで
  ONNX単一グラフ埋め込み可（非対応op無し・onnx.checker PASS・ONNX-CPU RTF 0.093）。VocosはiSTFTがONNX非対応で
  除外。音質は現行HiFi-GAN同等以上。調査は `docs/vocoder-improvement-survey.md`

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

### Duration Predictor
- **アーキテクチャ**: 2層Conv1d（256ch, k=3）+ Linear projection。~395Kパラメータ
- **受容野**: 5トークンのみ（2層×k=3、dilation無し）。日本語のプロソディ文脈には狭い
- **話者条件付け**: 間接的のみ（encoderのdetach出力経由）。DP内に直接のFiLM/話者projection無し
- **勾配**: `x_dp = torch.detach(x)` — encoder→DP方向の勾配なし（Glow-TTS/Grad-TTS準拠）
- **損失**: log-domain MSE、blank含む全トークンで均等重み。blankが~50%を占め短duration側にバイアス

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
- **精度**: **RTX 5090 DDPでは bf16-mixed が実証済みデフォルト**（`jvs_aligned`/`jvs_fast` にconfig化。出荷済み2500epモデルはbf16学習で全品質ゲート通過＝退化率0%/UTMOS 3.00、FP32比+11% steps/sec）。bf16はFP16と別物 — 8bit指数部を持ちoverflowしないため、下記FP16のdiff_lossスパイク/NaN発散は該当しない。**T4/V100等でFP16が不安定な環境ではCLIで `trainer.precision=32-true` に戻す**。bf16-mixed時は `model.optimizer.fused=false` 必須（fused AdamW + mixed + gradient clipping はクラッシュ）
- **out_size=null**: 全長メル学習が必須。`out_size=172`はDecoder/Duration Predictorの品質不足を招く
- **原論文準拠のLoss**: prior重み1.0、LOG_2PI復元、MSE duration loss。変更すると品質劣化
- **原論文準拠のLR**: `lr=1e-4`、scheduler=なし、`weight_decay=0.0`。高LRは不安定
- **一様分布timestep sampling**: logit-normalはTTSでは検証不足で品質劣化の原因
- **EMA**: `decay=0.9995`、`update_starting_at_epoch=10`
- **max_epochs=2500**: 原論文500Kステップに匹敵する~240Kステップ。500epでは不十分（ただしMAS退化問題は学習量では解決しない）

### 学習ステップ数の比較（原論文 vs JVS）
| 指標 | 原論文（LJSpeech） | JVS 500ep | JVS 2500ep |
|------|-------------------|-----------|------------|
| 総ステップ | 500,000 | 48,000 | 240,000 |
| 有効バッチサイズ | 64 | 128 | 128 |
| サンプル露出量 | 32M | 6.1M | 30.7M |
| 話者数 | 1 | 100 | 100 |
| 話者あたり露出 | 32M | 61K | 307K |

### 過去に失敗した最適化（適用しないこと）
- **FP16 Mixed Precision**: Loss計算にFP32キャスト追加してもDuration Predictor品質が劣化（音素あたり2.4フレーム vs 正解7.2フレーム）。encoder/decoder内部表現のFP16精度不足が原因。NaN防止だけでは不十分（2026-04-06検証済み）
- **LRスケジューラ（warmup + cosine decay）**: 原論文はlr=1e-4固定。cosine decayはDuration Predictorの収束を妨げる可能性
- **Logit-Normal timestep sampling**: t≈0,1の学習不足 → decoder出力に正バイアス
- **out_size=172**: Duration Predictorが内部音素に1-1.5フレームしか割り当てず発音不明瞭
- **prior_loss重み0.5**: encoder mu_y品質低下 → decoder条件付け劣化の連鎖
- **torch.empty（out_size切り出し）**: 未初期化メモリでNaN混入 → torch.zeros必須
- **LR 5e-4 + weight_decay=0.01**: 原論文より攻撃的すぎて不安定

### MASアライメント退化問題（2026-04-10確認、対策決定済み）

100話者JVS学習において、MAS（Monotonic Alignment Search）が構造的に退化アライメントを生成する問題が確認された。**学習量を増やしても改善しない**（500ep→2500epで悪化: 39.2%→43.0%）。

#### 根本原因: MASは多話者TTSに不向き

MASはlog_prior（ガウシアン距離 `-½||y - μ_x||²`）に基づく貪欲DPアルゴリズムであり、正しく機能するにはencoder出力μ_xがphonemeごとに十分異なる表現を持つ必要がある。多話者設定ではencoderが100人分の音声特性を同一空間に圧縮するため、μ_xが平均化されblank/phonemeの区別が消失する。

**日本語55音素がMAS退化を加速**: 英語VCTKでは178音素（細粒度）のためμ_xの区別が容易でMAS退化は報告されていない。日本語55音素は粗粒度でモーラ寄りのため、音素間の音響差が小さくMAS退化を誘発しやすい。

#### 症状
- 学習サンプルの**39-43%でMASアライメントが退化**（phonemeの80%以上が1フレーム）
- blank[0]にフレームが集中（退化時: 平均101フレーム、正常時: 平均4フレーム）
- 全phonemeの62.9%が≤1フレーム、73.8%が≤2フレーム
- Duration Predictorは退化したMASターゲットを忠実に学習 → 推論時に各phoneme ~2フレーム

#### 原因チェーン
```
① Encoderのmu_xがblankとphonemeで類似（多話者で表現が平均化、日本語55音素で粒度不足）
  → ② MASのlog_prior計算でblank位置がphonemeと同等のスコアを得る
    → ③ MASの貪欲DPがblankにフレームを大量割当（monotonicity制約下で合理的）
      → ④ Duration Predictorが退化ターゲットを学習（detach + 話者条件付けなし）
```

#### 検証済みの数値
| 指標 | 退化サンプル(43%) | 正常サンプル(57%) |
|------|-----------------|-----------------|
| phoneme median duration | 1.0フレーム | 2.0フレーム |
| blank[0] duration | mean=101 | mean=4 |
| メル中のblank占有率 | 63% | 49% |
| encoder mu_x norm (phoneme) | 2.06-2.26 | 6.14-7.81 |

#### 学習量は原因ではない
- 原論文: 500Kステップ（LJSpeech、単一話者）
- JVS 500ep: 48Kステップ（原論文の9.6%）→ MAS退化率39.2%
- JVS 2500ep: 240Kステップ（原論文の48%）→ MAS退化率**43.0%（悪化）**
- 5倍の学習でも改善なし → アルゴリズムレベルの問題

#### 外部裏付け情報（15エージェント調査、2026-04-10）

| 情報源 | 内容 |
|--------|------|
| Alphacephei分析 (2025/01) | 「MASは多様なデータでは失敗する。現代TTSにはASRアライナーが必要」「DPがspeaker embeddingを使わないことが大きな問題」 |
| Grad-TTS著者 (GitHub #37) | 多話者チェックポイントは「proof-of-concept、品質保証なし」 |
| Matcha-TTS著者Mehta (GitHub #125) | `stoc_dur`ブランチで確率的DP実装済み |
| Glow-TTS著者 (GitHub #43) | blank挿入に「理論的根拠はない」 |
| Lajszczak et al. (2024) | MAS+決定論的DPの平均回帰バイアスを実証 |
| JATTS toolkit (名古屋大/戸田研) | Julius forced aligner + Matcha-TTSの日本語パイプライン実証済み |
| Style-BERT-VITS2 | 確率的DP（MAS退化の影響を受けにくい）でMOS 4.37達成 |
| Matcha-TTS VCTK設定 | MAS退化対策は一切なし、論文での多話者評価もなし |
| NVIDIA RAD-TTS | HMM forward-sum + beta-binomial priorで247話者LibriTTS安定動作 |

#### 対策方針: 外部アライナーによるduration事前計算（決定済み）

MASをバイパスし、外部forced alignerで正確なphoneme durationを事前計算して`use_precomputed_durations=true`で学習する。

**選定理由**:
- MAS退化の根本原因（アルゴリズム自体の多話者不適合）を完全に回避
- JATTSがJulius + Matcha-TTSの日本語パイプラインを実証済み
- ESPnet JVSレシピもMFA/teacher forcingでduration抽出（MAS非使用）
- コードベースに`use_precomputed_durations=True`のパスが既存（`matcha_tts.py` L193）
- アーキテクチャ変更不要で最もlow-risk

**検討した代替案と不採用理由**:

| 対策 | 不採用理由 |
|------|-----------|
| MAS blankペナルティ | 対症療法。先行実装なし。ペナルティ値のチューニングが必要で効果不確実 |
| 確率的DP（VITS式/stoc_dur） | MAS退化ターゲット自体は変わらない。外部アライナーとの併用が前提 |
| F5-TTS式Duration-free | モデルの70-80%書き直しが必要。コスト過大 |
| NVIDIA RAD-TTS式Forward-Sum | アーキテクチャ変更大。CFMデコーダとの組合せ未検証 |
| VITS2式MASノイズ注入 | 効果は+0.15 MOS程度。根本解決にならない |

#### 追加改善（外部アライナーと併せて実施予定）

1. **DPにFiLM話者条件付け追加**（~20行）: Alphacephei指摘の構造的欠陥を修正
2. **Blank embedding zero-init**（1行）: blank/phonemeの初期分離を促進

### 外部アライナー実装計画

#### アライナー選択: Julius forced aligner

| 候補 | 長所 | 短所 | 採用 |
|------|------|------|------|
| **Julius** | 日本語ネイティブ、JATTS実証済み、10ms精度 | 音素セットがpyopenjtalkと異なる | **○** |
| MFA | pretrained日本語モデルv2.0.1a | JVS issue #541、IPA→ローマ字マッピング必要 | △ |
| pyJuliusAlign | Julius wrapper、Python API | TTS用パイプライン自作必要 | △ |

#### 実装手順

1. **Julius segmentation-kitでJVSをアライメント**
   - julius-speech/segmentation-kitを使用
   - JVS各発話の.wavとひらがな転記を入力 → `.lab`ファイル（phoneme開始/終了時刻）を出力
   - 10ms精度（hop_length=256/22050Hz=11.6msとほぼ一致）

2. **音素セットのマッピング**
   - Julius音素 → pyopenjtalk 55シンボルへの変換テーブル作成
   - 主な差異: 促音(cl)、撥音(N)、ポーズ(pau/sil)、韻律記号(^,$,?,_,#,[,])
   - blank挿入（intersperse）後のシーケンス長との整合性確認

3. **Duration配列の生成**
   - `.lab`の時刻情報をフレーム数に変換: `frames = (end_time - start_time) * sr / hop_length`
   - blank位置のduration設定（0 or 1フレーム）
   - intersperse後の音素列長と一致するようduration配列を構成

4. **PrecomputedDataModuleの修正**
   - 現在`"durations": None`をハードコード → `.pt`ファイルに`"durations"`キーを追加
   - `precompute_dataset.py`にduration埋め込み機能を追加

5. **学習設定**
   - `model.use_precomputed_durations=true`
   - MASブロック（matcha_tts.py L196-208）がスキップされ、`generate_path(durations)`で直接アライメント生成
   - prior_loss、dur_lossは維持（DPの学習に正確なターゲットが供給される）

#### 参考実装
- **JATTS** (unilight/jatts): `matchatts.py`（TTS1）がJulius duration + Matcha-TTSを実装
- **ESPnet** (egs2/jvs/tts1): `scripts/mfa.sh`でMFAアライメント、FastSpeech2に供給
- **Matcha-TTS upstream**: `matcha/utils/get_durations_from_trained_model.py`でMAS durationを.npy保存する既存機能あり

#### 既存コードパスの確認
- **TextMelDataModule**: `load_durations=True` → `data_dir/durations/{name}.npy`を読み込み
- **PrecomputedDataModule**: `"durations": None`をハードコード（**要修正**）
- **モデル側**: `use_precomputed_durations=True` → `generate_path(durations)`でMASバイパス（matcha_tts.py L193-194）
- **設定**: `configs/model/matcha.yaml`に`use_precomputed_durations: ${data.load_durations}`

### 日本語TTS先行実装の参考情報（2026-04-10調査）

#### 日本語Matcha-TTS実装
| プロジェクト | 概要 | MAS対策 |
|-------------|------|---------|
| **JATTS** (unilight/jatts, 名古屋大/戸田研) | JVS 100話者対応。TTS1=Julius forced alignment、TTS2=MAS。Matcha-TTS/VITS実装 | TTS1でJuliusによるMASバイパス |
| **akjava/Matcha-TTS-Japanese** | 単話者（合成音声~100発話）。英語178シンボルテーブル流用。ONNX推論特化 | なし |
| **Fusic Zenn記事** | 英語モデルからITA 381文でfine-tune | なし |

#### 日本語多話者TTSの成功手法
| プロジェクト | 手法 | アライメント | 品質 |
|-------------|------|-------------|------|
| **Style-BERT-VITS2** (JP-Extra) | VITS + 確率的DP + WavLM識別器 | MAS（確率的DPで退化影響軽減） | MOS 4.37 |
| **ESPnet JVS** (egs2/jvs/tts1) | FastSpeech2 / VITS | MFA or teacher forcing | 研究ベンチマーク |
| **VOICEVOX** | 独自アーキテクチャ（分離型） | 別モデルでduration予測（MASなし） | 商用品質 |
| **Matxa-TTS** (カタルーニャ語47話者) | Matcha-TTS（VCTKからfine-tune） | MAS（そのまま） | 実用レベル |

#### Matcha-TTS公式多話者の状況
- **VCTKモデル（108話者）**: 公開済みだが論文での評価なし（Future Workに「多話者対応」と記載）
- **VCTK設定**: LJSpeechと完全同一。MAS退化対策は一切なし
- **英語178音素 vs 日本語55音素**: 英語は音素粒度が細かくμ_xの区別が容易なためMAS退化が起きにくい

### 日本語プロソディ表記の変更（BREAKING、2026-07）

`japanese_cleaners`（`matcha/text/cleaners.py` の `_fullcontext_to_prosody`）を ttslearn `pp_symbols` / ESPnet `pyopenjtalk_g2p_prosody` 準拠に修正した。音素トークン列がほぼ全発話で変化する:

- **マーカー位置**: `[` / `]` / `#` を対応する音素の**直後**に出力（旧実装は直前）。例: 旧 `^ # [ k a $` → 新 `^ k a ? `
- **マーカー数**: elifチェーンにより1音素につき最大1マーカー（旧実装は `#` と `[` を同時に出力することがあり、シーケンス長も変化）
- **疑問文**: 文末silはE3フラグにより `$` ではなく `?` を出力（例: 「元気ですか？」）

**移行が必要な資産（旧convention と新cleanerの混在は不可）**:
- 旧conventionで学習した日本語checkpoint → 新cleanerでの推論は品質が劣化する（エラーは出ない）。再学習が必要
- 事前計算済み `.pt` データセット（`data/jvs_precomputed*`、`jvs_precomputed_aligned`。`x`列が焼き込み済み）と `durations/*.npy` → precompute/alignmentパイプライン（`run_optimized_pipeline.py`等）の再実行が必要
- Julius alignment側の `?` 対応は `scripts/convert_julius_to_durations.py` / `matcha/alignment/base.py` で実装済み（`?` は疑問文の文末silとして `$` と同様にdurationを持つ。duration=0の韻律記号は `#` `[` `]` のみ）

### JVSデータの注意点
- **Juliusアライメントの既知の欠損（~24件、0.18%）**: 全角英字（Ａ/Ｈ）、LOANWORD128の特殊モーラ（てゃ/うょ/るぁ/ぐぉ等）、全角マイナス「−」、探索失敗（1件）はyomi2voca/Julius側で整列不能のため.pt生成から除外される（12,997中12,973件が学習に使用される）。ヴ行は`normalize_vu_kana`でバ行に正規化済み（pyopenjtalk自身もヴをb音素で出力するため整合する）
- **segmentation-kitはLinux用Juliusバイナリを同梱しない**: `bin/julius-4.3.1.exe`（Windows用）のみ。`run_segkit_batch`がシステムの`julius`（apt版）を`bin/julius-4.3.1`としてシンボリックリンクする。juliusが起動できない場合もsegment_julius.plは空.labを作ってexit 0するため、0バイト.labはエラーとして扱う（実装済み）
- **無音トリミング必須**: JVSコーパスは各発話の先頭/末尾に~500msの無音を含む。`prepare_jvs.py`で自動トリミング（`top_db=30`、50msマージン）
- **mel統計量**: トリミング後のデータで再計算が必要（`mel_mean: -6.550095`, `mel_std: 2.383771`）
- **blank[0] Duration爆発**: MASが先頭blankに大量フレームを割り当て、Duration Predictorがこれを学習する。推論時のclamp（max=3.0）で対症対応済みだが、根本原因はMASアライメント退化問題（上記参照）
- **torchaudio非互換**: PyTorch 2.10+ではtorchcodec依存でtorchaudio.loadが失敗する場合あり。`soundfile`をフォールバックとして使用

## パフォーマンス最適化

### 前処理最適化
`run_optimized_pipeline.py` によるフルパイプラインのend-to-end実測値（2026-04-15、JVS 12,997発話、16 workers）:

#### Step 3+4 の3パターン比較 (12,348件のtrain set)

| パターン | 実装 | 時間 | samples/sec | Speedup |
|:---|:---|:---:|:---:|:---:|
| Legacy | ProcessPoolExecutor 16 workers | **1511.9s (25.2分)** | ~8 | 1.0x (baseline) |
| Fast CPU | single-process + ThreadPool + shared text cache | **73.5s (1.2分)** | **168.1** | **20.6倍** |
| Fast GPU | 同上 + GPU batched mel (batch=64) | 94.2s (1.6分) | 131.1 | 16.0倍 |

**GPUがCPUより遅い理由**: H2D転送とPython-levelのreflect paddingがオーバーヘッドの大半を占め、mel計算の実コスト (~2ms/sample) より大きい。推奨設定は `--device cpu`。

#### フルパイプライン timing (fast CPUデフォルト)

| ステップ | 実測時間 | 備考 |
|:---|:---:|:---|
| Step 0: Text cache (T1-3) | **5.3s** | 3,099 unique texts並列処理 |
| Step 1: Prepare (T2-1) | **21.7s** | 16kHzリサンプル+ひらがな生成 |
| Step 2: Julius alignment (T1-1) | **68.3s** | 12,997件 / 0エラー / 16 workers |
| Step 3+4: Unified precompute (fast CPU) | **93.8s (1.6分)** | mel+duration+.pt生成 (train+val) |
| **合計 (fresh run)** | **~189s (3.1分)** | |

**前回との比較**: 1581.3s (26.4分) → 189s (3.1分) = **~8.5倍高速化**

**binary互換性**: legacy vs fast CPU で100件中100件完全一致 (mel diff = 0.0)、fast CPU vs fast GPU は atol=1e-4 以内で一致

#### Tier 1（即効性の高い最適化）
- **T1-1 Julius並列化**: ProcessPoolExecutor 16ワーカーで並列実行。実測 **68秒**（sequential推定14分の13倍高速化）。`run_julius_alignment.py`のインフラを再利用
- **T1-2 Duration変換並列化**: `sf.info()`でmel_frames直接計算（torch.load比2.2倍高速）+ ProcessPoolExecutor並列化
- **T1-3 テキストキャッシュ**: 3,099ユニークテキストを1回だけpyopenjtalk処理（5秒）。非キャッシュ時の~5分から**60倍高速化**

#### Tier 2（パイプライン統合）
- **T2-1 デュアルリサンプル**: `prepare_jvs.py --julius-output-dir`で22kHzと16kHzを同時出力（1回の音声読み込み）
- **T2-2 統合precompute**: `precompute_with_alignment.py`が.lab→duration変換とmel計算を1パスで実行（中間.npy廃止、7%高速化）
- **T2-3 /dev/shmキャッシュ**: `setup_shm_cache.sh --full`で中間ファイルをtmpfsに配置（NFS→SHM 1.4倍高速化、学習時I/O加速）

#### Tier 3（precompute fast path、2026-04-15、Phase 1-5）
ProcessPoolExecutor版からsingle-process版への書き直しで **25.2分 → 1.2分 (20.6倍高速化)** を達成。

- **T3-1 共有テキストキャッシュ**: `build_text_sequence_cache()` が 3,099 unique textsの `text_to_sequence(..., language="ja")` 結果 `(seq, cleaned_text)` を事前計算。ProcessPool で並列化、結果をmain processのdictに集約。multiprocessingで各ワーカーが独立にpyopenjtalkを叩く無駄を排除
- **T3-2 single-process + ThreadPool producer/consumer**: I/O (`sf.read` + `.lab` parse) は `ThreadPoolExecutor(io_workers)` で先読み並列化、`torch.save` も別poolで並列化。GILが sf.read/torch.save 内でリリースされるためThreadで十分。`ProcessPoolExecutor` の spawn/IPC/pyopenjtalk再ロードのオーバーヘッド (~18分分) を完全に排除
- **T3-3 GPU batched mel (ただし非推奨)**: `_mel_batch_gpu()` が可変長audioをzero-pad → `torch.stft(center=False)` batched → per-sample slice (`T_i = 1 + (L_i - 256)//256`) で個別に normalize → CPU転送。実装はあるがbenchmarkでCPU版より遅いため、`--device cpu` をデフォルトに

**新CLI引数** (`scripts/precompute_with_alignment.py`): `--legacy` (旧経路強制), `--device cpu|cuda|auto`, `--batch-size N`, `--io-workers N`, `--text-cache-workers N`, `--bench-only N`

**`run_optimized_pipeline.py` からの透過**: `--precompute-legacy-path`, `--precompute-device` (default: cpu), `--precompute-batch-size` (default: 32), `--precompute-io-workers` (default: 16)

#### run_segkit_batch のバグ修正（2026-04-15）
`scripts/run_julius_alignment.py` に以下のバグがあり、修正済み:
- **旧**: `.txt`ファイルを`tmp/txt/`にsymlinkしていたが、`segment_julius.pl`は`.wav`と同じディレクトリから`.txt`を読む仕様 → 全Julius処理が失敗し、以前の12,575件は手動Perl実行で生成されていた
- **新**: `.wav`と`.txt`の両方を`tmp/wav/`にsymlink + `bin/`と`models/`を segkit から temp dirへsymlink（Julius実行ファイル解決）
- **効果**: バグ修正後は12,997件全てが0エラーで完了（修正前は422件欠落）

### 学習最適化
- **Fused AdamW**: `fused=True`でオプティマイザステップ高速化（ただしmixed precision[bf16/FP16]+gradient clippingとは非互換 → bf16学習時は `model.optimizer.fused=false`）
- **bf16-mixed精度**: RTX 5090 DDPで実証済みデフォルト（`jvs_aligned`/`jvs_fast`、FP32比+11% steps/sec、品質同等）。FP16はDuration Predictor品質劣化のため不可。T4/V100でbf16非対応/FP16不安定な環境はCLIで `trainer.precision=32-true` に戻す
- **Gradient Checkpointing**: デコーダのTransformerブロックのみ（ResNetブロックは除外して計算効率化）
- **DDP最適化**: `gradient_as_bucket_view=true`、`bucket_cap_mb=25`（通信overlap有効化）、`broadcast_buffers=false`、NCCLタイムアウト7200秒
- **ログ最適化**: `sync_dist=False`（ステップレベル）、`log_dict()`統合でDDPオーバーヘッド削減
- **データ読込**: os.scandir（NFS 12倍高速）、ファイルサイズキャッシュ、drop_last=True、preload_to_memoryオプション
- **チェックポイント**: `save_on_train_epoch_end=true`で確実に保存（validation非依存）

#### 高速化オプトインscaffolding（2026-07、全てデフォルトOFF・byte-identical）
実装状況・採否根拠は `docs/training-speedup-implementation-plan.md`、調査は `docs/training-speed-optimization-survey.md`。**速度のために実証済み品質を賭ける変更は無し**。
- **A-1 プロファイリング（律速確定=他投資の前提）**: `bash scripts/profile_training.sh`（または `experiment=jvs_aligned_profile`）。`matcha/callbacks/torch_profiler_callback.py` がrank0限定でカーネル内訳+chrome traceを出力。GEMM律速/カーネル起動律速/通信律速/データ律速を判定
- **A-2 Regional torch.compile（`compile_regional_blocks=true` で有効化、既定false）**: `Decoder.compile_regions()` がdecoderのtransformerブロックを **`nn.Module.compile()`（in-place）で個別compile → state_dictキー不変**（ckpt/EMA/resume が壊れない）。trainer.fit前=DDPラップ前に適用しpytorch#140229を回避。`gradient_checkpointing=true` とは併用不可（自動スキップ）。期待1.0–1.10x。A-1で「カーネル起動律速」確認＋3段品質ゲート通過時のみ本採用
- **C-1/C-2 ランタイム**: 本番launcher `scripts/train_jvs_aligned.sh`（`NCCL_P2P_DISABLE=1`/`NCCL_IB_DISABLE=1` — 5090はP2P物理不可で副作用ゼロ）。`setup_vastai.sh` に persistence mode（安全）+ 電力制限（`MATCHA_POWER_LIMIT` 明示時のみ、既定575W不変）
- **見送り/後回し**: A-3 cuDNN SDPA（explicit maskでengageせず期待≈0・Blackwell silent-bugリスク → 見送り）、B-1 frame batching（A-1で充填律速が実証された場合のみ着手）

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
- `compile_model=false`が必要（`torch.compile`はDDP + dynamic shapesで不安定）
- `static_graph=false`が必要 — **`gradient_checkpointing: true`の場合のみ**。`gradient_checkpointing: false`（jvs_fast/jvs_aligned）では`static_graph: true`が使用可能で、DDP通信最適化の恩恵を受けられる
- `out_size=null`の場合 `data.batch_size=32`が安定上限
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`推奨
- NCCLタイムアウトは7200秒に設定済み（NFS I/O遅延対策）
- `use_distributed_sampler=false`でカスタムDistributedBucketBatchSamplerを使用
- `sync_batchnorm=false`（モデルにBatchNormなし）
