# プロジェクト現状まとめ（2026-07-07 時点）

日本語Matcha-TTS（JVS 100話者）の開発状況スナップショット。詳細は各 `docs/*.md` を参照。
このファイルは**現状の索引**であり、詳細な根拠は個別ドキュメントに委ねる。

---

## 1. 全体像

- **リポジトリ**: `feature/japanese-support` ブランチ、**PR #3**（feat: 日本語音声合成サポート）OPEN
- **状態**: origin と完全同期・作業ツリークリーン（全push済み）
- **出荷済みモデル**: `jvs_aligned` 2500エポック完走モデル → HF private `ayousanz/matcha-tts-jvs-ja`
  （`jvs_aligned/last.ckpt` + hydra設定 + tensorboard）

### 出荷モデルの品質（`docs/jvs-aligned-eval-report.md`）
| 指標 | 実測 | 判定 |
|------|------|:---:|
| MASアライメント退化率 | **0.0%**（旧MAS手法: 39〜43%） | ✅ PRの核心主張を実証 |
| UTMOS（学習モデル合成） | 3.00 ± 0.38 | ボコーダ天井(2.90)に到達 |
| blank[0] duration | 1.0フレーム（退化時: 101） | ✅ |

**核心**: Julius外部アライナーによるMASバイパスが、日本語多話者TTSのMAS退化をアルゴリズムレベルで解消した。

---

## 2. このセッションの成果（2026-07-07、2タスク完了）

### タスク① 学習高速化（`docs/training-speedup-implementation-plan.md`）
過去の調査（`docs/training-speed-optimization-survey.md`）を実コードに落とし込み、9エージェントのワークフローで敵対的検証。

- **bf16-mixed 恒久化（採用・ON）**: `jvs_aligned`/`jvs_fast` にconfig化。CLI手打ち不要・fused誤設定クラッシュ防止。
  出荷モデルは既にbf16学習で全品質ゲート通過（FP32比+11% steps/sec、品質同等）
- **A-1プロファイリング（実行済み）**: 律速は「カーネル起動律速」だが対象はdecoder conv経路（GEMMは~2%）
- **A-2 Regional compile（見送り確定）**: A-1実測でtransformer（対象）は律速でないと判明 → 3段ゲート検証に進まず。
  scaffoldingはデフォルトOFFで保持（`compile_regional_blocks`）
- **C-1/C-2 ランタイム（実装済み・オプトイン）**: NCCL_P2P_DISABLE launcher、persistence/電力制限
- **結論**: bf16の+11%が実質到達点。現状（2500ep≈24h/$45）受容が合理的。**速度のため品質を賭ける変更は無し**

### タスク② ボコーダ改善（`docs/vocoder-improvement-survey.md`）
音質頭打ち（UTMOS 3.00 ≈ 汎用HiFi-GAN天井2.90）の改善を、CPU/モバイル/配布/ONNX化を制約に調査・実装。

| フェーズ | 成果 |
|------|------|
| 調査（5エージェント） | mel完全一致ボコーダ4つ発見。**ONNX速度基準でWaveNeXt本命確定**（VocosはiSTFTがONNX非対応、BSC実例が裏付け） |
| Phase A 実装 | WaveNeXt移植（`matcha/wavenext/`、純torch・iSTFT無し）+ cli統合。実重みキー1:1一致検証 |
| Phase A A/B | ゼロショット = HiFi-GAN**同等**（UTMOS 2.925 vs 2.983、有意差なし）。**mel互換を実証**（破綻0） |
| 診断（BigVGAN） | 濁りは**fmax=8000構造天井**（最強BigVGANでも改善せず）→ ボコーダfine-tuneは無効。GPU出費回避 |
| Phase C ONNX化 | Matcha+WaveNeXtを単一グラフ埋め込み成功。**非対応op無し**・onnx.checker PASS・**ONNX-CPU RTF 0.093** |

**結論**: WaveNeXtはデプロイ全要件（CPU/モバイル/ONNX/配布）を満たす。音質は現状受容（濁りはボコーダでは不可）。

---

## 3. 実装された主要な変更（コード）

- `matcha/wavenext/`（新規）: WaveNeXtボコーダ（modules/models/vocoder）。iSTFT無しのConvNeXt+線形ヘッド
- `matcha/cli.py`: `load_vocoder` に wavenext 分岐、VOCODER_URLS に BSC-LT/wavenext-mel
- `matcha/onnx/export.py`: `dynamo=False`（torch≥2.9のdynamo exporterがMatcha synthesise SymIntで失敗するため旧TorchScript exporterに固定）
- `matcha/callbacks/torch_profiler_callback.py`（新規）: A-1カーネルプロファイラ
- `matcha/models/components/decoder.py`: `compile_regions()`（A-2、デフォルトOFF、state_dictキー不変）
- `configs/experiment/jvs_aligned.yaml`/`jvs_fast.yaml`: bf16-mixed + fused=false 恒久化
- `scripts/`: `eval_vocoder_ab.py`, `eval_bigvgan_probe.py`, `profile_training.sh`, `train_jvs_aligned.sh`
- テスト: 1086+ passed（+5 wavenext, +6 regional compile, onnx 14 passed）、ruff clean

---

## 4. インフラ状況

- **vast.aiインスタンス 44019256（RTX 5090×4）**: **破棄済み**（2026-07-07、データ削除・課金停止）
  - ⚠️ 他に5インスタンス（`43931180`/`43982092`/`44074501`/`44095987`/`44096702`）がアクティブ。本セッション対象外で状態未確認
- **バックアップ方針**（`memory: training-infra-vastai`、過去checkpoint全消失の教訓）: モデルはHFに必須バックアップ。
  jvs_alignedはHFに保全済み。前処理データ（`data/jvs_precomputed_aligned`、~12,973件）はパイプライン再実行で~3分で再生成可
- **ローカル環境**: 全作業CPU・$0で完結、HF認証済み（ayousanz）、torch 2.10.0+cpu

---

## 5. 未解決 / 次の選択肢

| 項目 | 内容 | 規模 |
|------|------|:---:|
| **濁りの本質改善** | fmax=8000→11025 引き上げ。mel統計再計算・全前処理やり直し・2500ep再学習を伴う破壊的変更 | 大 |
| **PR #3マージ準備** | レビュー対応・整理 | 中 |
| **他vast.aiインスタンス** | 5インスタンスの状態確認・不要なら停止/破棄 | 小 |
| **WaveNeXt JVS fine-tune** | 濁りには無効と診断済み（BigVGANプローブ）。話者性微調整が必要なら別途 | 中 |

**設計上の恒久制約**（CLAUDE.md、変更禁止）: 実証済みレシピ（lr=1e-4固定・out_size=null・uniform sampling・
prior重み1.0・EMA decay=0.9995・有効バッチ128）は品質退化の実績があり変更しない。MARINE不採用。

---

## 6. ドキュメント索引

| ドキュメント | 内容 |
|------|------|
| **`docs/tsukuyomi-tts-verification-log.md`** | **★このセッション(2026-07-07〜09)の全検証ログ: fmax検証/ボコーダ交換/つくよみfine-tune/根本原因診断** |
| `docs/jvs-aligned-eval-report.md` | 出荷モデルの評価（退化率0%・UTMOS 3.00） |
| `docs/vocoder-improvement-survey.md` | ボコーダ改善 調査+Phase A/C実測+BigVGAN診断+fmax見送り決定（WaveNeXt本命） |
| `docs/moe-tsukuyomi-pipeline-plan.md` | MoeSpeech→つくよみ パイプライン計画（fmax=8000維持に方針変更） |
| `docs/wavenext-training-tdd-plan.md` | WaveNeXt学習コード TDD計画 |
| `docs/training-speedup-implementation-plan.md` | 学習高速化 実装+A-1実測（A-2見送り） |
| `docs/training-speed-optimization-survey.md` | 学習高速化 調査（18候補） |
| `docs/vastai-troubleshooting.md` | vast.ai運用トラブル対応 |
| `docs/next-steps-plan.md` | 次ステップ計画 |
| `docs/japanese-tts-implementation-plan.md` | 日本語TTS実装計画 |
| `CLAUDE.md` | プロジェクト全体ガイド（恒久制約・知見） |
