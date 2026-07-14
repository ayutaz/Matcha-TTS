# 学習高速化 実装計画・状況（2026-07-07）

`docs/training-speed-optimization-survey.md` の調査候補を、9エージェントのワークフロー
（分析→敵対的検証→統合）で実コードに落とし込んだ結果と実装状況。方針は一貫して
**「実証済み品質（退化率0% / UTMOS 3.00）を1ミリも損なわず、全てオプトインのfeature flag、
デフォルトはbyte-identical」**。速度のために品質を賭ける変更は一切含まない。

## 実装状況サマリー

| 項目 | 判定 | 状況 | デフォルト |
|------|:---:|------|:---:|
| **bf16-mixed 恒久化** | 採用 | ✅ 実装・適用済み（config default化） | **ON**（実証済み） |
| **A-1 プロファイリング** | 採用 | ✅ 実装・**実行済み**（2026-07-07、結果は下記） | 明示選択時のみ |
| **C-1 NCCL / C-2 電力** | 採用 | ✅ 実装済み（インスタンスで適用待ち） | opt-in |
| **A-2 Regional compile** | **見送り確定** | ✅ scaffolding実装済み（**A-1でtransformerは律速でないと実測 → 本採用せず**） | **OFF** |
| **A-3 cuDNN SDPA** | 見送り | ❌ 未実装（ROIゼロ・品質リスク） | — |
| **B-1 frame batching** | 後回し | ⏸️ 未実装（A-1で充填律速確認時のみ） | — |

## 実装済みの変更（このコミット）

### bf16-mixed 恒久化（ON by default）
- `configs/experiment/jvs_aligned.yaml` / `jvs_fast.yaml`: `trainer.precision: bf16-mixed` +
  `model.optimizer.fused: false`（fused AdamW + mixed + grad clip はクラッシュ）
- 根拠: 出荷済み2500epモデルは既にbf16-mixed学習で全品質ゲート通過（退化率0% / UTMOS 3.00 /
  FP32比+11% steps/sec）。bf16はFP16と別物（8bit指数部でoverflowせず、FP16の劣化は非該当）
- 効果: CLI手打ち（`trainer.precision=bf16-mixed model.optimizer.fused=false`）が不要化、
  fused誤設定によるクラッシュを構造的に防止
- `CLAUDE.md` の precision記述を修正（bf16はFP16と別物・5090で実証済みと明記）

### A-1 プロファイリング（律速の確定 = 他投資の前提）
- `matcha/callbacks/torch_profiler_callback.py`（新規）: torch.profilerでカーネル内訳を取得。
  rank0限定・`should_stop`を立てない（DDP lockstep維持）・`key_averages(group_by_input_shape=False)`
  で可変長メルでも読める1枚テーブル + chrome trace出力
- `configs/callbacks/torch_profiler.yaml`（新規）: `callbacks=torch_profiler` で選択
- `configs/experiment/jvs_aligned_profile.yaml`（新規）: jvs_aligned継承、EMA/early_stopping除去、
  1epoch/35バッチ/bf16-mixed固定
- `scripts/profile_training.sh`（新規）: 単GPU/4GPU切替（`MATCHA_PROFILE_DEVICES`）、
  `--cfg job` dry-run内蔵
- **律速判定**: GEMM(addmm/mm/bmm)高→compute律速 / 微小カーネル多+CPU≫CUDA→**カーネル起動律速（A-2の出番）** /
  ncclDevKernel_*高→通信律速 / copy_・step境界アイドル→データ律速（B-1）

### C-1 NCCL / C-2 電力（リスクゼロの安定化）
- `scripts/train_jvs_aligned.sh`（新規・本番launcher）: `NCCL_P2P_DISABLE=1` / `NCCL_IB_DISABLE=1`
  （5090はP2P物理不可なので副作用ゼロ）/ `PYTORCH_CUDA_ALLOC_CONF` をexport。bf16はconfig default
  なので precision override不要
- `scripts/setup_vastai.sh`（追記）: persistence mode（安全・可逆）、電力制限は `MATCHA_POWER_LIMIT`
  明示時のみ（既定575Wを1バイトも変えない）、nccl-testsビルドは `MATCHA_BUILD_NCCL_TESTS=1` のみ
  （通常はnvcc欠如で失敗するため一次手段は `NCCL_DEBUG=INFO`）

### A-2 Regional compile（scaffolding、OFF by default）
- `matcha/models/components/decoder.py`: `Decoder.compile_regions(mode)` を追加。
  **`nn.Module.compile()`（in-place）を使用 → state_dictキー不変**（ckpt/EMA strict-load/resume が壊れない。
  `torch.compile(block)` 再代入は禁止）。down/mid/up の全transformerブロック（計6個・全dim=256）を個別compile。
  forward 3ループに `_regional_compiled` 時のみ `mark_dynamic(x/mask, 1)`（時間次元Tをsymbolic化）
- `matcha/train.py`: `compile_regional_blocks` 分岐を追加（**trainer.fit前 = DDPラップ前**に適用 →
  compiled領域がDDP境界の内側 → pytorch#140229の再コンパイル暴走を回避）。
  `gradient_checkpointing=true` との併用は自動スキップ。既存の `compile_model`（フルcompile・DDP時skip）は
  `elif` に降格し二重コンパイル防止
- `configs/experiment/jvs_aligned.yaml`: `compile_regional_blocks: false`（デフォルトOFF=byte-identical）
- `tests/test_regional_compile.py`（新規）: **state_dictキー不変**（核心保証）・flag/count・
  デフォルトforward finite・compiled state_dict の strict再ロード・（CUDA限定）compiled vs eager 数値一致
- **期待効果は控えめ**: decoderのcompute_lossは1step1回・256ch launch-bound・SnakeBetaが
  `@torch.compiler.disable` でeager → end-to-end **1.0–1.10x（ノイズ内なら不採用）**

## A-1 実測結果（2026-07-07、単GPU RTX 5090、bf16-mixed、20 profiled steps）

`bash scripts/profile_training.sh` を出荷モデル同構成で実行した実測。

### 律速判定: カーネル起動/dispatch律速で確定（GPU idle ~83%）
| 指標 | 値 |
|------|------|
| GPU実カーネル時間（Self CUDA total） | **66ms/step**（1.321s / 20step） |
| wall-clock | ~390ms/step（2.56 it/s） |
| **GPU稼働率** | **~17%（=83%アイドル）** |
| 1stepあたりカーネル起動数 | copy_ 1339 / layout変換 1265 / mul 821 / add_ 846 / conv_bwd 68 = **数千個/step** |

Self CPU 17.0s ≫ Self CUDA 1.32s + 数千個/stepの微小カーネル = 典型的なカーネル起動律速。

### GPU計算の内訳（self CUDA time）
| 演算 | 占有 | 呼び出し | 経路 |
|------|:---:|:---:|------|
| `convolution_backward` | **39.1%** | 1,360 | decoder ResNet/sampling conv |
| `cudnn_convolution` | 14.3% | 1,340 | 同上 |
| `copy_` | 12.6% | 26,784 | 内部コピー |
| layout変換 nchwToNhwc/nhwcToNchw | 12.0% | 38,458 | Conv1d + cudnn NHWC algo |
| elementwise (mul等) | ~15% | 数万 | mask乗算/GroupNorm等 |
| **GEMM (mm/xmma) ← A-2の対象** | **~2%** | — | transformer attention/FFN |

### 結論（A-2見送りの根拠）
- decoderの**conv経路が~65%**を占め、**A-2が対象とするtransformerブロックのGEMMは~2%と極小**
- → A-2（transformerのみcompile）は律速に当たっておらず、効果はほぼ確実にノイズ内。3段ゲート検証に進む価値なし。**A-1が本来の役割（無駄投資回避）を果たした**
- 本当に効かせるにはdecoder forward全体のcompile（DDP下フルcompile=pytorch#140229の高リスク領域）・layout変換38k回削減（可変長Conv1dで難）・fused AdamW（crash制約）が必要で、いずれも品質を賭けない方針では割に合わない
- **bf16の+11%が事実上の到達点**。これ以上は現状受容（24h/$45）が合理的
- scaffolding（`compile_regional_blocks` フラグ）はデフォルトOFFで無害なため保持（将来decoder全体compileを試す土台）

chrome trace: instance上 `logs/train/jvs_aligned/runs/2026-07-07_05-50-35/profiler/trace_rank0.json`

## インスタンス起動後の手順（stopped → start で$1.76/hr課金再開）

1. **A-1 profiling**: `bash scripts/profile_training.sh`（単GPU）→ 律速を確定 ★投資判断の分岐点
2. **C-1/C-2**: `setup_vastai.sh` 再実行 or 手動で persistence/power 適用、`NCCL_DEBUG=INFO` で transport確認
3. **A-1結果で分岐**:
   - カーネル起動律速 → A-2の3段ゲートへ（本命）
   - GEMM律速 → A-2/A-3薄い。何もしないが合理的
   - 通信/データ律速 → C-1 / B-1検討
4. **A-2採否（3段ゲート）**:
   - G1（ローカル済）: state_dictキー不変 ✅ / 数値等価（要TF32off） / allow_in_graph実効（Inductorグラフ生成確認）
   - G2（単GPU）: bf16 loss曲線がbaselineとノイズ内一致
   - G3（4x5090 DDP）: 全rank step一致・`static_graph=true`でreducerエラー無し・resume/EMA成功
   - G4: steps/sec改善が有意（>数%）でなければ「無害だが無益」→ 本番投入せず
   - G5: 退化率0% / UTMOS ≥ 3.00 維持
   - 有効化: `+compile_regional_blocks=true`。即revert: `+compile_regional_blocks=false`

## コスト/効果の総括

現状ベースライン: **bf16 ~2.8 steps/s → 2500ep ≈ 24h / ~$45**。

- **実施済み**: bf16（採用）+ A-1（実行し律速確定、~$0.6）+ C-1/C-2（実装、適用は次回起動時）
- **A-2は見送り確定**: A-1実測でtransformer（A-2の対象）は律速でないと判明（GEMM~2%）。
  カーネル起動律速の本体はconv経路 + 微小kernelで、A-2のスコープ外。3段ゲートに進まない
- **原則やらない**: A-3（ROIゼロ）。B-1は「充填律速の実証 + 複数run予定」の二条件が揃うまで着手しない
- **最も合理的な既定**: 現行bf16構成（24h/$45）で何もしないのも正当。opt-in scaffoldingはlandしておき
  （default経路はbyte-identical）、次にインスタンスを起動する用事のついでにA-1で律速を確認する低コミット運用

詳細な各論・verifier corrections は本ワークフローの合成結果に準拠。
根拠調査は `docs/training-speed-optimization-survey.md`。
