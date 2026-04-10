# M4: 学習設定変更・段階的学習実行

## マイルストーン概要

外部アライナー（Julius）による事前計算durationとFiLM話者条件付けDuration Predictorを用いて、
JVS 100話者モデルを再学習する。MASアライメント退化問題（39-43%）を完全にバイパスし、
全音素の退化率0%を達成することが本マイルストーンの目標である。

### 背景

従来のMASベース学習では以下の問題が確認されている:

| 学習設定 | MAS退化率 | dur_loss | 備考 |
|---------|----------|----------|------|
| FP16 500ep | - | - | NaN発散で学習失敗 |
| FP32 500ep | 39.2% | - | `logs/train/jvs_fast/runs/2026-04-07_01-21-26/` |
| FP32 2500ep | 43.0% | - | `logs/train/jvs_fast/runs/2026-04-08_14-33-58/` |

学習量を5倍に増やしても退化率が悪化（39.2% → 43.0%）しており、アルゴリズムレベルの問題であることが確認済み。
本マイルストーンでは、M1（Julius alignment）、M2（duration付き.ptファイル）、M3（FiLM DP + blank zero-init）の
成果物を統合し、MASを完全にバイパスした学習を実行する。

### 依存関係

```
M1: Julius Alignment ──→ M2: DataModule対応 ──→ M4: 学習実行（本マイルストーン）
                                                    ↑
M3: FiLM DP + Blank init ─────────────────────────┘
```

- **M2が完了していること**: duration付き.ptファイルが `/dev/shm/jvs_precomputed_aligned/` に配置済み
- **M3が完了していること**: DurationPredictorにFiLM話者条件付け、blank embedding zero-initが実装済み

### 完了条件

1. Hydra設定ファイルが正しく解決され、`use_precomputed_durations=true`で学習が開始できること
2. Epoch 10時点でdur_loss/prior_loss/diff_lossにNaNが発生しないこと
3. Epoch 100時点でMAS退化率が0%（事前計算durationのため構造的に保証）
4. Epoch 500時点でMCDがMASベースライン（FP32 500ep）と同等以上であること
5. Epoch 2500まで学習が完了し、チェックポイントが保存されていること

### 想定期間

- T-M4-01（設定ファイル変更）: 0.5日
- T-M4-02（学習実行・モニタリング）: 5-7日（4x T4での2500ep学習）
- 合計: 6-8日

### 一から作り直すとしたら

もしMatcha-TTSの日本語多話者学習を一から設計するなら、以下の構成を選択する:

1. **MASを最初から使わない**: 外部アライナー（Julius or MFA）によるduration事前計算を前提とし、
   学習コードからMASパスを除去した専用設定を用意する。MASはLJSpeech（単話者・英語178音素）では
   有効だが、多話者・日本語55音素の組み合わせでは構造的に退化するため、最初から外部アライナーを前提とすべき。

2. **実験設定の分離を徹底する**: `jvs_fast.yaml`を改変するのではなく、最初から
   `jvs_aligned.yaml`として独立した実験設定を作成する。MASベースとアライナーベースの
   設定が混在すると、パラメータの相互依存が見えにくくなる。

3. **段階的検証をCI/CDに組み込む**: Epoch 10/50/100/500の各チェックポイントで自動評価スクリプトを
   実行するcallbackを実装し、退化率やMCDの閾値を超えた場合にSlack通知を送る仕組みを最初から用意する。
   現状は手動モニタリングに依存しており、5-7日の学習中に問題を見落とすリスクがある。

4. **WandBを最初からデフォルトロガーにする**: TensorBoardは手動でのサーバ起動が必要であり、
   リモートGPUサーバでの長期学習モニタリングにはWandBのほうが適している。

---

## T-M4-01: 学習設定ファイル変更・Hydra設定統合

### タスク目的とゴール

事前計算durationを使用した学習のためのHydra設定ファイルを作成し、
`data.load_durations=true` → `model.use_precomputed_durations=true`の設定連鎖が
正しく解決されることを検証する。

**ゴール**: `uv run python matcha/train.py experiment=jvs_aligned`で学習が正常に開始できる状態にする。

### 実装する内容の詳細

#### 1. 新規実験設定ファイル `configs/experiment/jvs_aligned.yaml` の作成

`jvs_fast.yaml`をベースに、duration対応の変更を加えた独立した実験設定を作成する。
既存の`jvs_fast.yaml`は変更せず、MASベースライン再現用に保持する。

```yaml
# @package _global_

# Japanese multi-speaker TTS with pre-computed Julius durations
# Bypasses MAS entirely via use_precomputed_durations=true
# Depends on: M2 (duration-enabled .pt files), M3 (FiLM DP + blank zero-init)
# To execute: python matcha/train.py experiment=jvs_aligned

defaults:
  - override /data: jvs_precomputed_aligned.yaml
  - override /trainer: ddp_optimized.yaml
  - override /callbacks: default.yaml

tags: ["jvs", "japanese", "multispeaker", "aligned", "julius"]

run_name: jvs_aligned

model:
  n_vocab: 55

compile_model: true
compile_mode: "default"
gradient_checkpointing: false

callbacks:
  early_stopping:
    _target_: lightning.pytorch.callbacks.EarlyStopping
    monitor: "loss/val"
    patience: 30
    min_delta: 0.001
    mode: "min"
    check_finite: true
    verbose: true
  ema:
    _target_: lightning.pytorch.callbacks.EMAWeightAveraging
    decay: 0.9995
    update_every_n_steps: 1
    update_starting_at_epoch: 10

trainer:
  max_epochs: 2500
  check_val_every_n_epoch: 10
  precision: "32-true"
```

**`jvs_fast.yaml`との差分**:
- `defaults`で`jvs_precomputed_aligned.yaml`を参照（`load_durations: true`を含む）
- `run_name: jvs_aligned`（ログ出力ディレクトリの分離）
- `tags`に`"aligned"`, `"julius"`を追加（実験管理用）
- その他パラメータ（lr, EMA, max_epochs, precision）はMASベースラインと同一に保つ

#### 2. 新規データ設定ファイル `configs/data/jvs_precomputed_aligned.yaml` の作成

```yaml
_target_: matcha.data.precomputed_datamodule.PrecomputedTextMelDataModule
name: jvs_precomputed_aligned
train_pt_dir: /dev/shm/jvs_precomputed_aligned/train
val_pt_dir: /dev/shm/jvs_precomputed_aligned/val
batch_size: 32
num_workers: 8
pin_memory: True
n_spks: 100
n_feats: 80
data_statistics:
  mel_mean: -6.550095
  mel_std: 2.383771
seed: ${seed}
load_durations: true
num_buckets: 20
```

**`jvs_precomputed.yaml`との差分**:
- `load_durations: true`（既存は`false`）
- `train_pt_dir` / `val_pt_dir`: duration付き.ptファイルのディレクトリを指定
- `name: jvs_precomputed_aligned`

#### 3. Hydra設定解決チェーンの検証

以下の設定伝播が正しく動作することを確認する:

```
jvs_precomputed_aligned.yaml: load_durations: true
    ↓ (Hydra interpolation)
configs/model/matcha.yaml: use_precomputed_durations: ${data.load_durations}
    ↓ (解決後)
MatchaTTS.__init__: self.use_precomputed_durations = True
    ↓ (forward時)
matcha_tts.py L193: if self.use_precomputed_durations → generate_path(durations)
```

検証方法:
```bash
# Hydra設定の解決結果を表示（学習は実行しない）
uv run python matcha/train.py experiment=jvs_aligned \
  --cfg job 2>&1 | grep -E "(load_durations|use_precomputed_durations)"
```

#### 4. DDP起動コマンドの確定

本番学習で使用するコマンドを確定する:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python matcha/train.py \
  experiment=jvs_aligned compile_model=false \
  data.batch_size=32 data.num_workers=0 +data.preload_to_memory=true \
  test=false
```

コマンドライン上書きパラメータの理由:
- `compile_model=false`: DDP + gradient checkpointingとの非互換（ddp_optimized.yamlの`static_graph: true`と衝突するため）
- `data.batch_size=32`: 4x T4 (16GB) でout_size=null時の安定上限
- `data.num_workers=0`: `preload_to_memory=true`使用時はI/Oワーカー不要
- `+data.preload_to_memory=true`: 全.ptファイルをメモリにプリロード（NFS I/O排除）
- `test=false`: テストステップをスキップ

#### 5. チェックポイント再開コマンドの確定

学習中断時の再開コマンド:
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python matcha/train.py \
  experiment=jvs_aligned compile_model=false \
  data.batch_size=32 data.num_workers=0 +data.preload_to_memory=true \
  test=false ckpt_path=logs/train/jvs_aligned/runs/<run_dir>/checkpoints/last.ckpt
```

#### 6. ddp_optimized.yamlとの整合性確認

現在の`configs/trainer/ddp_optimized.yaml`の設定を確認する:

```yaml
# 確認すべき項目:
strategy.static_graph: true   # compile_model=falseでOK
strategy.find_unused_parameters: false  # use_precomputed_durations=trueでもMAS関連パラメータは存在するため問題なし
devices: [0,1,2,3]           # 4x T4
gradient_clip_val: 1.0        # FP32では通常不要だが安全策として維持
check_val_every_n_epoch: 20   # jvs_aligned.yamlのoverride(10)が優先される
```

`find_unused_parameters`について: `use_precomputed_durations=true`の場合、MAS計算ブロック（L196-208）はスキップされるが、
MASに関連するパラメータ（monotonic_align等）はモデルパラメータではなく外部関数のため、
`find_unused_parameters: false`のままで問題ない。

### エージェントチームの役割と人数

| 役割 | 人数 | 担当内容 |
|------|------|---------|
| 設定エンジニア | 1名 | Hydra設定ファイル作成、設定解決チェーンの検証 |
| レビュアー | 1名 | 設定値の妥当性確認、MASベースラインとの差分レビュー |

**合計: 2名**

### 提供範囲とテスト項目

#### 成果物
- `configs/experiment/jvs_aligned.yaml`（新規作成）
- `configs/data/jvs_precomputed_aligned.yaml`（新規作成）
- 学習起動コマンドのドキュメント（CLAUDE.mdへの追記）

#### テスト項目

| # | テスト内容 | 合格条件 | コマンド |
|---|----------|---------|---------|
| 1 | Hydra設定解決 | `use_precomputed_durations: true`が出力される | `uv run python matcha/train.py experiment=jvs_aligned --cfg job` |
| 2 | データパス解決 | `train_pt_dir`が正しいパスに解決される | 同上 |
| 3 | DDP起動（dry-run） | エラーなく最初のバッチが処理される | 学習コマンド + `trainer.max_epochs=1 trainer.limit_train_batches=2` |
| 4 | duration読み込み確認 | `batch["durations"]`がNoneでないこと | dry-run中のログ or デバッグprint |
| 5 | MASバイパス確認 | `generate_path(durations)`が呼ばれること | matcha_tts.py L193のログ出力 or デバッグ |
| 6 | チェックポイント保存 | `checkpoints/last.ckpt`が生成されること | dry-run後のファイル確認 |
| 7 | 既存設定の非破壊確認 | `experiment=jvs_fast`が従来通り動作すること | `uv run python matcha/train.py experiment=jvs_fast --cfg job` |

### 懸念事項とレビュー項目

#### 懸念事項

1. **`preload_to_memory=true`のメモリ消費増大**: duration付き.ptファイルはduration配列分だけサイズが増加する。
   duration配列は`int64`で音素数分（平均~50要素 = 400バイト）であり、メル（80x数百フレーム = 数十KB）に比べて
   無視できるサイズのため問題にならない見込みだが、全サンプルプリロード後のメモリ使用量を確認すること。

2. **`static_graph: true`とuse_precomputed_durationsの組み合わせ**: DDPのstatic_graphはforward時の計算グラフが
   毎回同一であることを前提とする。`use_precomputed_durations=true`ではMASブロックが常にスキップされるため
   グラフは安定するが、`compile_model=false`を指定している場合は`static_graph`の恩恵は限定的。
   問題が発生した場合は`static_graph: false`に変更する。

3. **EMAのupdate_starting_at_epoch=10**: M3でDurationPredictor構造が変更されているため、
   EMAの開始エポックを調整する必要がある可能性。ただし、FiLM層の追加は小規模（~20行）であり、
   学習初期のウォームアップとして10エポックは妥当と判断。変更が必要な場合はT-M4-02で対応する。

4. **mel統計量の変更有無**: M2のduration付き.pt再生成でメル自体は変更されないため、
   `mel_mean: -6.550095`, `mel_std: 2.383771`はそのまま使用可能。
   ただしM2でメル再計算が行われた場合は統計量の再確認が必要。

#### レビュー項目

- [ ] `jvs_aligned.yaml`と`jvs_fast.yaml`の差分が意図通りか（load_durations, run_name, tagsのみ）
- [ ] `jvs_precomputed_aligned.yaml`のパスがM2の出力先と一致しているか
- [ ] Hydra `--cfg job`で全設定値が期待通りに解決されるか
- [ ] `model.use_precomputed_durations`が`true`に解決されるか
- [ ] `data.load_durations`が`model.use_precomputed_durations`に正しく伝播するか
- [ ] 既存の`jvs_fast.yaml`が変更されていないか

### 一から作り直すとしたら

Hydra設定の管理方針として、以下を改善する:

1. **`load_durations`と`use_precomputed_durations`の二重管理を解消**: 現在はdata側の`load_durations`を
   model側が`${data.load_durations}`で参照する間接参照になっている。一から設計するなら、
   `model.alignment_source: "mas" | "precomputed"`のような単一パラメータで制御し、
   data/modelの両方がこれを参照する構成にする。

2. **データパスの外部化**: `/dev/shm/jvs_precomputed_aligned/`のようなホスト依存パスを
   設定ファイルにハードコードせず、環境変数`${oc.env:JVS_DATA_DIR}`で外部注入する。
   これにより同じ設定ファイルを異なるマシンで再利用できる。

3. **実験設定のバリアント管理**: `jvs_fast.yaml`と`jvs_aligned.yaml`の重複を、
   Hydraのdefaults listで共通設定を`jvs_base.yaml`に抽出して解消する:
   ```yaml
   # jvs_aligned.yaml
   defaults:
     - jvs_base
     - override /data: jvs_precomputed_aligned.yaml
   ```

### 後続タスクへの連絡事項

- **T-M4-02（段階的学習実行）へ**: 本チケットで確定した起動コマンドをそのまま使用すること。
  `compile_model=false`は必須（`true`にするとDDP環境でクラッシュする）。
- **M2チームへの確認依頼**: duration付き.ptファイルの出力ディレクトリが
  `jvs_precomputed_aligned.yaml`の`train_pt_dir`/`val_pt_dir`と一致していることを確認すること。
  パスの不一致はHydra設定解決時にはエラーにならず、DataModuleのsetup()時に初めて失敗する。
- **M3チームへの確認依頼**: FiLM DP追加後もモデルの`__init__`シグネチャに変更がないこと
  （Hydra設定からのインスタンス化に影響しないこと）を確認すること。

---

## T-M4-02: 段階的学習実行・モニタリング

### タスク目的とゴール

T-M4-01で確定した設定を用いて4x T4環境で2500エポックの学習を実行し、
各チェックポイントで品質指標をモニタリングする。MASベースライン（FP32 500ep/2500ep）との
定量比較により、外部アライナー + FiLM DPの効果を検証する。

**ゴール**:
- 2500エポックの学習を完了し、最終チェックポイントを保存すること
- Epoch 10/50/100/500の各段階でMASベースラインとの比較データを記録すること
- MAS退化率0%を確認すること（事前計算durationにより構造的に保証）

### 実装する内容の詳細

#### 1. 学習環境の事前準備

##### 1.1 データ配置の確認

```bash
# M2出力の.ptファイルがdurationを含んでいることを確認
python -c "
import torch
sample = torch.load('/dev/shm/jvs_precomputed_aligned/train/sample_000000.pt', weights_only=True)
print('Keys:', list(sample.keys()))
print('Duration shape:', sample['durations'].shape if sample['durations'] is not None else 'None')
print('Text length:', sample['text'].shape[0])
# durationの合計フレーム数がメル長と一致することを確認
dur_sum = sample['durations'].sum().item()
mel_len = sample['mel'].shape[1]
print(f'Duration sum: {dur_sum}, Mel length: {mel_len}')
assert abs(dur_sum - mel_len) <= 1, f'Duration/mel mismatch: {dur_sum} vs {mel_len}'
"
```

##### 1.2 GPU環境の確認

```bash
# GPU状態の確認
nvidia-smi
# NCCL通信テスト（4GPU間）
python -c "
import torch
import torch.distributed as dist
# 4GPUが全て利用可能であることを確認
print(f'CUDA devices: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {props.name}, {props.total_mem / 1024**3:.1f} GB')
"
```

##### 1.3 ディスク空間の確認

チェックポイントは`every_n_epochs=10`、`save_top_k=10`で保存される。
1チェックポイントあたり約500MB（FP32モデル + オプティマイザ状態 + EMA）と見積もり:
- 常時保持: last.ckpt + top_k=10 = 最大11ファイル = 約5.5GB
- ログ（TensorBoard）: 約500MB

```bash
# ディスク空間の確認
df -h /data/Matcha-TTS/logs/
# 最低10GBの空きが必要
```

#### 2. 学習実行

##### 2.1 学習起動コマンド

```bash
# screenまたはtmuxセッション内で実行
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python matcha/train.py \
  experiment=jvs_aligned compile_model=false \
  data.batch_size=32 data.num_workers=0 +data.preload_to_memory=true \
  test=false
```

##### 2.2 学習再開コマンド（中断時）

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python matcha/train.py \
  experiment=jvs_aligned compile_model=false \
  data.batch_size=32 data.num_workers=0 +data.preload_to_memory=true \
  test=false ckpt_path=logs/train/jvs_aligned/runs/<run_dir>/checkpoints/last.ckpt
```

##### 2.3 学習時間の見積もり

| 項目 | 値 |
|------|-----|
| 総サンプル数 | ~24,000（JVS 100話者 x ~240発話） |
| バッチサイズ | 32 x 4GPU = 128（有効バッチサイズ） |
| 1エポックのステップ数 | ~24,000 / 128 = ~188ステップ/GPU |
| 1ステップの所要時間（推定） | ~0.5秒（FP32、out_size=null） |
| 1エポックの所要時間 | ~94秒 = ~1.6分 |
| 2500エポックの総時間 | ~4,000分 = ~2.8日 |
| マージン込み見積もり | **5-7日**（バリデーション、チェックポイント保存、I/O待ちを含む） |

注意: duration使用時はMAS計算（L196-208）がスキップされるため、
MASベースラインよりステップ速度がやや向上する可能性がある。

#### 3. モニタリング計画

##### 3.1 TensorBoardモニタリング

デフォルトロガーはTensorBoard（`configs/train.yaml`の`logger: tensorboard`）。

```bash
# TensorBoardの起動
uv run tensorboard --logdir logs/train/jvs_aligned/ --port 6006 --bind_all
```

監視すべきメトリクス:

| メトリクス名 | TensorBoardキー | 正常範囲 | 異常兆候 |
|-------------|----------------|---------|---------|
| Duration損失 | `sub_loss/train_dur_loss` | 0.1-0.5（収束時） | 1.0以上で停滞、NaN |
| Prior損失 | `sub_loss/train_prior_loss` | 0.5-2.0 | 5.0以上で停滞 |
| Flow matching損失 | `sub_loss/train_diff_loss` | 0.1-0.5 | 1.0以上で停滞 |
| 合計損失 | `loss/train` | 上記3つの合計 | 発散傾向 |
| 検証損失 | `loss/val` | 学習損失と同程度 | 学習損失との乖離拡大（過学習） |
| 勾配ノルム | `grad_norm/*` | 0.1-10.0 | 100以上（勾配爆発） |

##### 3.2 WandBモニタリング（任意追加）

TensorBoardに加えてWandBを使用する場合:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python matcha/train.py \
  experiment=jvs_aligned compile_model=false \
  data.batch_size=32 data.num_workers=0 +data.preload_to_memory=true \
  test=false \
  logger=wandb logger.wandb.project=matcha-tts-jvs logger.wandb.tags='[jvs,aligned,julius]'
```

WandBの利点:
- リモートからのモニタリング（ブラウザアクセス可能）
- MASベースラインのrunとのoverlay比較が容易
- アラート設定（dur_lossがNaNになった場合の通知等）

##### 3.3 生成サンプルの目視確認

`on_validation_end()`で10エポックごとにTensorBoardへ画像が記録される:
- `generated_enc/{i}`: エンコーダ出力メルスペクトログラム
- `generated_dec/{i}`: デコーダ出力メルスペクトログラム
- `alignment/{i}`: アテンションマップ

確認ポイント:
- アテンションマップが明確な対角線構造を示していること（MASベースラインでは退化サンプルで崩れていた）
- デコーダ出力に不自然なアーティファクト（繰り返し、空白領域）がないこと

#### 4. 段階的検証計画

##### Epoch 10: 初期収束確認

| 検証項目 | 方法 | 合格条件 |
|---------|------|---------|
| NaN発生なし | TensorBoardログ確認 | 全損失値が有限 |
| dur_loss収束開始 | `sub_loss/train_dur_loss`の推移 | 単調減少傾向 |
| EMA開始 | ログメッセージ確認 | `EMAWeightAveraging`が有効化 |
| GPU使用率 | `nvidia-smi` | 4GPU全てが80%以上 |

**MASベースラインとの比較**:
- MASベースライン（FP32 500ep）のEpoch 10時点のdur_lossと比較
- 事前計算durationはMASより正確なターゲットを提供するため、dur_lossの初期値が低いことが期待される

##### Epoch 50: Duration精度の検証

```bash
# チェックポイントからduration予測精度を評価
uv run python scripts/evaluate_durations.py \
  --checkpoint logs/train/jvs_aligned/runs/<run_dir>/checkpoints/checkpoint_050.ckpt \
  --data-dir /dev/shm/jvs_precomputed_aligned/val \
  --output-dir logs/train/jvs_aligned/eval_epoch050
```

| 検証項目 | 方法 | 合格条件 |
|---------|------|---------|
| Duration予測精度 | 予測duration vs 正解durationのMAE | MAE < 2.0フレーム |
| FiLM条件付けの効果 | 話者別のduration精度分散 | 話者間の精度分散が小さいこと |
| Blank duration | blank[0]の予測duration | 平均 < 5フレーム（MASベースライン: 退化時101フレーム） |

注意: `scripts/evaluate_durations.py`はこの時点では存在しない可能性がある。
必要に応じてT-M4-02の一部として簡易評価スクリプトを作成する（M5の本格評価とは別）。

##### Epoch 100: 退化率の確認

| 検証項目 | 方法 | 合格条件 |
|---------|------|---------|
| 退化率 | 予測durationで80%以上のphonemeが1フレームのサンプル比率 | 0%（構造的に保証） |
| Prior lossの安定性 | TensorBoardログ | 単調減少または収束 |
| 過学習兆候 | train loss vs val lossの乖離 | 乖離率 < 20% |

**重要**: 事前計算durationを使用しているため、MAS退化は構造的に発生しない。
ただし、Duration Predictorの予測精度が低い場合、推論時に疑似的な退化が起こる可能性がある。
Epoch 100時点でDP予測の退化率を確認することで、FiLM条件付けの効果を検証する。

##### Epoch 500: MCD評価

```bash
# 推論サンプル生成
uv run python scripts/synthesize_eval.py \
  --checkpoint logs/train/jvs_aligned/runs/<run_dir>/checkpoints/checkpoint_500.ckpt \
  --eval-list data/jvs/eval.txt \
  --output-dir logs/train/jvs_aligned/eval_epoch500 \
  --n-timesteps 10
```

| 検証項目 | 方法 | 合格条件 |
|---------|------|---------|
| MCD (Mel Cepstral Distortion) | 生成メル vs 正解メルのMCD計算 | MASベースライン（FP32 500ep）以下 |
| 話者類似度 | 話者埋め込みのコサイン類似度 | 平均 > 0.85 |
| 音声サンプルの聴取 | 5話者 x 3発話の人間による聴取 | 明瞭な発音、自然なリズム |
| Duration予測精度 | MAE（フレーム単位） | MAE < 1.5フレーム |

**MASベースライン比較表（Epoch 500時点）**:

| 指標 | MASベースライン 500ep | 本学習 500ep | 改善目標 |
|------|---------------------|-------------|---------|
| MAS退化率 | 39.2% | 0% | 完全解消 |
| dur_loss | (要測定) | (要測定) | 低下 |
| prior_loss | (要測定) | (要測定) | 低下 |
| MCD | (要測定) | (要測定) | 同等以上 |

##### Epoch 2500: 最終評価（M5への引き渡し）

2500エポック完了時のチェックポイントをM5（推論・評価・品質検証）に引き渡す。

保存すべきアーティファクト:
- `checkpoints/last.ckpt`: 最終チェックポイント
- `checkpoints/checkpoint_2500.ckpt`: 名前付きチェックポイント
- TensorBoardログ一式
- 各段階の評価結果（duration精度、MCD等）

#### 5. 異常時の対応手順

##### 5.1 NaN発生時

```bash
# 最後の正常チェックポイントを確認
ls -lt logs/train/jvs_aligned/runs/<run_dir>/checkpoints/

# batch_sizeを下げて再開
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python matcha/train.py \
  experiment=jvs_aligned compile_model=false \
  data.batch_size=16 data.num_workers=0 +data.preload_to_memory=true \
  test=false ckpt_path=<last_valid_checkpoint>
```

NaN発生の主な原因候補:
1. duration配列にゼロまたは負の値が含まれている（M2のバグ） → データ検証スクリプトで確認
2. FiLM層の初期化が不適切（M3のバグ） → FiLM層の出力を確認
3. 勾配爆発 → `gradient_clip_val: 1.0`が設定済みだが、値の引き下げを検討

##### 5.2 dur_lossが収束しない場合

dur_lossがEpoch 50以降も1.0以上で停滞する場合:
1. 事前計算durationの品質を再確認（M1/M2の出力検証）
2. FiLM条件付けが正しく機能しているかデバッグ（M3の検証）
3. encoderのmu_xとdurationターゲットの整合性を確認

##### 5.3 OOM（メモリ不足）発生時

```bash
# batch_sizeを段階的に下げる
data.batch_size=24  # 第1段階
data.batch_size=16  # 第2段階

# gradient_checkpointingを有効化
gradient_checkpointing=true  # ただしDDP static_graph=trueとの非互換に注意
```

##### 5.4 学習が不安定（損失の振動）な場合

1. `gradient_clip_val`を`0.5`に引き下げ
2. `batch_size`を増加（メモリ許容範囲内で）して勾配分散を低減
3. EMAの`update_starting_at_epoch`を20に延長

#### 6. ベースライン比較用のログ整理

学習開始前に、MASベースラインのログから比較用データを抽出する:

```bash
# MASベースラインのTensorBoardログを参照用にコピー
cp -r logs/train/jvs_fast/runs/2026-04-07_01-21-26/tensorboard \
      logs/baselines/jvs_mas_fp32_500ep/

cp -r logs/train/jvs_fast/runs/2026-04-08_14-33-58/tensorboard \
      logs/baselines/jvs_mas_fp32_2500ep/

# TensorBoardで並列表示
uv run tensorboard --logdir_spec \
  aligned:logs/train/jvs_aligned/,\
  mas_500ep:logs/baselines/jvs_mas_fp32_500ep/,\
  mas_2500ep:logs/baselines/jvs_mas_fp32_2500ep/ \
  --port 6006 --bind_all
```

### エージェントチームの役割と人数

| 役割 | 人数 | 担当内容 |
|------|------|---------|
| 学習オペレータ | 1名 | 学習起動、中断時の再開、GPU/ディスクの監視 |
| 品質モニタリング | 1名 | TensorBoardの定期確認、各段階の評価実行、MASベースラインとの比較 |
| トラブルシュート | 1名（兼任可） | NaN/OOM/損失停滞時の原因調査と対応 |

**合計: 2-3名**（学習オペレータとトラブルシュートは兼任可能）

### 提供範囲とテスト項目

#### 成果物
- 2500エポック学習済みチェックポイント（`last.ckpt` + top_k=10）
- 各段階（Epoch 10/50/100/500/2500）の評価レポート
- TensorBoardログ一式
- MASベースラインとの比較表

#### テスト項目

| # | テスト内容 | 合格条件 | 実施タイミング |
|---|----------|---------|-------------|
| 1 | 学習開始確認 | エラーなく最初の10ステップが完了 | 学習開始直後 |
| 2 | NaN検出 | 全損失値が有限 | 各エポック（自動: `check_finite: true`） |
| 3 | dur_loss収束 | Epoch 10でdur_loss < 2.0 | Epoch 10 |
| 4 | Duration予測精度 | MAE < 2.0フレーム | Epoch 50 |
| 5 | 退化率 | 0% | Epoch 100 |
| 6 | MCD | MASベースライン以下 | Epoch 500 |
| 7 | 話者類似度 | 平均 > 0.85 | Epoch 500 |
| 8 | 全損失収束 | loss/valが安定 | Epoch 2500 |
| 9 | チェックポイント完全性 | last.ckptからの推論が成功 | Epoch 2500 |
| 10 | Early Stopping非発火確認 | 2500エポックに到達（早期終了していない） | 学習完了時 |

### 懸念事項とレビュー項目

#### 懸念事項

1. **Early Stoppingの予期せぬ発火**: `patience=30`（= 300エポック分の検証間隔）が設定されている。
   外部アライナーにより学習が安定し、損失が早期に収束する可能性がある。
   Epoch 500-1000で損失が十分収束し、2500エポック到達前にEarly Stoppingが発火する可能性がある。
   - 対策: `patience`を50に引き上げるか、Early Stoppingを無効化することを検討。
   - 判断基準: Epoch 500時点のMCDがMASベースラインを十分に下回っていれば、
     Early Stoppingで打ち切られても問題ない。

2. **EMAモデルの品質**: EMAは`decay=0.9995`でEpoch 10から更新される。
   M3でDuration Predictorの構造が変更されているため、EMAの追従が遅れる可能性がある。
   - 検証方法: Epoch 500時点でEMAモデルと通常モデルの両方で推論し、品質を比較する。

3. **学習時間の見積もり精度**: 1ステップ~0.5秒はMASベースラインでの実測値に基づく推定。
   duration使用時はMAS計算がスキップされるためやや高速化するが、
   FiLM DP（M3）の計算コスト増加と相殺される可能性がある。
   実際の学習速度はEpoch 1完了後に再見積もりする。

4. **4GPU間のNCCL通信安定性**: 7200秒のNCCLタイムアウトが設定済みだが、
   NFS I/Oが重い場合にタイムアウトする可能性がある。`preload_to_memory=true`により
   学習中のNFS I/Oは最小化されているが、チェックポイント保存時に一時的なI/O集中が発生する。
   - 対策: チェックポイント保存中にNCCLタイムアウトが発生した場合は、
     `every_n_epochs`を20に変更して保存頻度を下げる。

5. **Epoch間のduration付きデータの一貫性**: `preload_to_memory=true`で全データをメモリにロードするため、
   学習中にデータが変更される心配はない。ただし、学習中にM2のデータを再生成してしまうと、
   メモリ上の古いデータと/dev/shm上の新しいデータが不整合を起こす。
   - 対策: 学習中は`/dev/shm/jvs_precomputed_aligned/`を変更しないこと。

#### レビュー項目

- [ ] 学習起動コマンドがT-M4-01で確定したものと完全一致しているか
- [ ] ベースライン比較用のログが正しく保存されているか
- [ ] 各段階の合格条件が現実的か（特にMCDの目標値）
- [ ] 異常時対応手順が網羅的か
- [ ] Early Stoppingのpatience設定が2500エポック学習に適切か
- [ ] ディスク空間が学習完了まで十分か

### 一から作り直すとしたら

学習実行と品質モニタリングの観点で、以下を改善する:

1. **自動評価callbackの実装**: Epoch 50/100/500で手動評価スクリプトを実行する現在の計画は、
   5-7日の学習中に人間の介入を必要とする。PyTorch Lightningのcallbackとして
   `on_validation_epoch_end`に評価ロジック（duration精度、疑似退化率）を組み込み、
   TensorBoardに自動記録する仕組みを最初から用意すべき。

2. **A/Bテスト用の推論callbackの追加**: 学習中の特定エポックで自動的に音声サンプルを生成し、
   TensorBoard Audioに記録する。現在は画像（メルスペクトログラム）のみが記録されており、
   音質の判断に別途推論スクリプトの実行が必要。

3. **学習再現性の保証**: seed固定（現在`seed: 1234`）に加え、Hydra設定のスナップショットを
   チェックポイントに含める。Hydraはデフォルトで`.hydra/`ディレクトリに設定を保存するが、
   コマンドライン上書きパラメータとの組み合わせで再現が困難になる場合がある。

4. **GPU死活監視の自動化**: 4GPU学習で1台のGPUが無応答になった場合、NCCLタイムアウト（7200秒 = 2時間）まで
   学習が停止する。GPU監視スクリプト（nvidia-smi polling + Slack通知）を学習と並行して実行し、
   異常を早期検知する仕組みを導入すべき。

### 後続タスクへの連絡事項

- **M5（推論・評価・品質検証）へ**: 最終チェックポイントのパスは
  `logs/train/jvs_aligned/runs/<run_dir>/checkpoints/last.ckpt`。
  EMAモデルのチェックポイントも`last.ckpt`内に含まれている（Lightning EMACallbackによる自動保存）。
  M5での推論時は、EMAモデルの重みを使用すること（通常モデルより高品質の可能性が高い）。
- **M5へ（比較データ）**: 各段階の評価結果を`logs/train/jvs_aligned/eval_epoch{N}/`に保存している。
  MASベースラインとの比較表も同ディレクトリに配置する。
- **CLAUDE.mdの更新**: 学習完了後、以下をCLAUDE.mdに追記する:
  - 新しい学習コマンド（`experiment=jvs_aligned`）
  - 最終的なdur_loss/prior_loss/diff_lossの値
  - MASベースラインとの比較結果
  - Early Stoppingが発火した場合はその旨と到達エポック数
