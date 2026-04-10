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
- **FP32精度**: T4/V100ではFP16が学習不安定を引き起こす（diff_lossスパイク、NaN発散）
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

### JVSデータの注意点
- **無音トリミング必須**: JVSコーパスは各発話の先頭/末尾に~500msの無音を含む。`prepare_jvs.py`で自動トリミング（`top_db=30`、50msマージン）
- **mel統計量**: トリミング後のデータで再計算が必要（`mel_mean: -6.550095`, `mel_std: 2.383771`）
- **blank[0] Duration爆発**: MASが先頭blankに大量フレームを割り当て、Duration Predictorがこれを学習する。推論時のclamp（max=3.0）で対症対応済みだが、根本原因はMASアライメント退化問題（上記参照）
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
- `compile_model=false`が必要（`torch.compile`はDDP + dynamic shapesで不安定）
- `static_graph=false`が必要 — **`gradient_checkpointing: true`の場合のみ**。`gradient_checkpointing: false`（jvs_fast/jvs_aligned）では`static_graph: true`が使用可能で、DDP通信最適化の恩恵を受けられる
- `out_size=null`の場合 `data.batch_size=32`が安定上限
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`推奨
- NCCLタイムアウトは7200秒に設定済み（NFS I/O遅延対策）
- `use_distributed_sampler=false`でカスタムDistributedBucketBatchSamplerを使用
- `sync_batchnorm=false`（モデルにBatchNormなし）
