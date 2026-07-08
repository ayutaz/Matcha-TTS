# つくよみちゃんTTS 検証ログ（2026-07-07〜09）

このセッションで実施した全検証・実験・意思決定の記録。目的は「日本語Matcha-TTSの音質改善」
および「最終的につくよみちゃん単一話者TTSを作る」こと。各実験は**実測に基づいて意思決定**した。

関連ドキュメント: `docs/vocoder-improvement-survey.md`（ボコーダ調査）、
`docs/moe-tsukuyomi-pipeline-plan.md`（MoeSpeechパイプライン計画）、
`docs/wavenext-training-tdd-plan.md`（WaveNeXt学習TDD計画）、
`docs/training-speedup-implementation-plan.md`（学習高速化）。

---

## 0. 出発点

- 出荷済みモデル `jvs_aligned`（JVS 100話者・fmax=8000・Julius外部アライナー・退化率0%・UTMOS 3.00）
- 診断済みの課題: 音質に「濁り」がある。UTMOS 3.00 ≈ 汎用HiFi-GAN天井2.90でボコーダ由来と推定
- 最終目標: **つくよみちゃん（単一話者・高品質）のTTS**

---

## 1. 学習高速化（完了）

`docs/training-speedup-implementation-plan.md` 参照。要点:
- **bf16-mixed 恒久化**（config default化、+11% steps/sec、品質同等）
- A-1プロファイリング実行 → 律速はカーネル起動律速（decoder conv経路）。**A-2 Regional compile は対象がGEMM~2%で見送り**
- C-1/C-2（NCCL/電力）scaffolding実装。B-1/A-3は見送り
- 結論: **bf16の+11%が実質到達点**

---

## 2. ボコーダ改善調査 → WaveNeXt実装・ONNX化（完了）

`docs/vocoder-improvement-survey.md` 参照。要点:
- 5エージェント調査で **mel完全一致の公開ボコーダ4つ**発見。**ONNX速度基準でWaveNeXt本命確定**
  （VocosはiSTFTがONNX非対応。BSCが同じMatcha-TTSでVocosを捨てWaveNeXt採用した実例）
- **WaveNeXt移植**（`matcha/wavenext/`、iSTFT無し・純torch）+ cli統合。Phase A ゼロショットA/B = HiFi-GAN同等
- **ONNX化成功**: Matcha+WaveNeXt単一グラフ・非対応op無し・ONNX-CPU RTF 0.093
- **WaveNeXt学習コード（TDD）**: `wavenext_train/`（40テスト全CPU・wetdog逐語移植・D11補正）

---

## 3. fmax引き上げ検証（見送り確定）★重要

「濁り = fmax=8000構造天井」という仮説を、fmax=11025で検証した。

### 3-1. 交絡排除 A/B（同データ・同ステップ・唯一fmaxだけ違う）
WaveNeXtを MoeSpeech 3話者・50000バッチで **fmax=11025 と fmax=8000 の2本**学習し、
同一val 40 wav で高域(8-11kHz)忠実度を比較（`scripts/eval_highband_fidelity.py`）:

| 指標 | fmax=11025 | fmax=8000（同データ学習） |
|------|:---:|:---:|
| 高域 log-STFT L1（低い=GTに近い） | **0.825** | 1.031 |
| 高域エネルギー比（1が理想） | **0.851** | 0.553（45%欠損） |
| paired 高域L1 | fmax=11025が40/40でGTに近い | — |

### 3-2. 結論
- **fmaxは独立して効く**（同データ学習でもfmax=8000は高域45%欠損。データ変更では埋まらない）
- **しかしユーザ試聴では体感差なし**（`eval/pure_fmax_listen/`）
- → 体感差がないのに破壊的変更（mel統計再計算・全前処理やり直し・音響モデル全再学習 $27-53）は
  割に合わず、**fmax引き上げは見送り・fmax=8000維持**を決定
- UTMOSはfmax差を測れない（16kHzにダウンサンプルするため8kHz超を見ない）ことも判明

**成果物**: `checkpoints/wavenext_ja_8000_50kbatch.bin`（MoeSpeech日本語適合WaveNeXt fmax=8000）

---

## 4. ボコーダ交換検証（濁りはボコーダでは取れないと確定）★重要

音響モデル(jvs_aligned)を固定し、ボコーダだけ交換する3-way A/B（`scripts/eval_vocoder_ab.py`）:

| ボコーダ | UTMOS | 位置づけ |
|------|:---:|------|
| WaveNeXt BSC（汎用ゼロショット） | **3.049** | 汎用がJVSには最良 |
| HiFi-GAN univ（現行） | 2.945 | — |
| WaveNeXt 日本語学習（MoeSpeech） | **2.586** | 最悪（耳・UTMOS一致） |

### 結論
- **MoeSpeechで学習したボコーダはアニメ声ドメインに寄り、JVS(朗読)で最悪**（ドメインミスマッチ）
- **ボコーダ交換だけでは濁りは取れない**。汎用ボコーダがJVSには無難
- 前回のBigVGAN診断（最強ボコーダでも濁り改善せず）と合わせ、**濁りはボコーダをどう変えても取れない**
  ことが2回の独立実験で確定
- → 濁りの主因はボコーダの「種類」ではなく、音響モデルの予測mel or JVSコーパスの本質的限界

---

## 5. つくよみちゃん fine-tune（実施・部分的成功）

### 5-1. 方針
JVSの濁り追跡をやめ、最終目標のつくよみ学習へ。**土台の選択でユーザは「既存jvs_alignedを土台に」を選択**
（MoeSpeech事前学習$27-53を回避し、事前学習コスト$0）。

```
jvs_aligned（日本語100話者）→ 話者embedding遷移(100→101) → つくよみfine-tune
```

### 5-2. 実装（すべてローカル$0で準備）
- **話者遷移**: `scripts/transfer_speaker_embedding.py` で spk_emb 100→101 resize（つくよみ=slot 100）、
  weights-only（optimizer_states除去）、mel統計-6.55/2.38継承 → `checkpoints/tsukuyomi_init_from_jvs.ckpt`
- **データ**: つくよみ90発話 precompute（fmax=8000・jvs統計・MAS）→ train 90 / val 10 の `.pt`
- **config**: `tsukuyomi_finetune.yaml`（n_spks=101・単一GPU・bf16・MAS・num_buckets=1）

### 5-3. 学習（インスタンス）
- 単一RTX 5090・~33分・~$0.4。max_epochs=1500だがepoch924で終了（early_stopping相当）
- wandb loggerの `add_image` クラッシュを修正（`matcha/models/baselightningmodule.py` の `_log_image`、
  TensorBoard/wandb両対応）。commit `2552a60`
- 成果物: `checkpoints/tsukuyomi_finetune_last.ckpt`（334.9MB）、`eval/tsukuyomi_synth/`（10文試聴wav）

### 5-4. 評価（ユーザ試聴）
- **「似てはいるが、つくよみちゃんとしてはまだ足りない（声質が違う）」**
- 合成音声自体は正常（2-4.4秒・健全な振幅・破綻なし）

### 5-5. ★根本原因の診断（重要な知見）
学習済みつくよみ embedding（slot 100）を初期化の平均ベクトルと比較:

```
tsukuyomi vs mean-init: cos=1.000  L2=0.031
```

**つくよみの話者embeddingが学習でほぼ全く動いていない**（cos類似度1.000・L2距離0.031）。

原因:
- Matchaの話者embeddingへの勾配は encoder の detach 経由でしか流れない構造
- EMA（decay=0.9995）が話者embeddingの変化を強く平滑化 → 平均init付近に固定
- 90発話・33分ではこの平滑化を押しのける力が足りなかった

→ **fine-tuneは走ったが話者embeddingがほぼ初期値（JVS平均声）のまま** = だから「JVS平均っぽい声」。
**Matchaの単一話者特化には構造的限界がある可能性**。

### 5-6. 検討した改善策（未実施）
- **近い話者init**: 学習済みembeddingに最も近いJVS話者を特定（cos: 話者80, L2: 話者85）。
  `checkpoints/tsukuyomi_init_spk80.ckpt` / `_spk85.ckpt` を作成済み（未学習）
- EMA緩和 / LR引き上げ / 長めfine-tune
- これらは話者embeddingが動かない前提でも「出発点をつくよみに近づける」効果を狙う

---

## 6. 次のステップ（未決・保留）

音質・つくよみらしさをさらに上げる選択肢（コスト小さい順）:

| 案 | 内容 | コスト | 確実性 |
|----|------|:---:|:---:|
| **A. jvs土台で改善** | 近い話者init(80/85) + EMA緩和 + LR強め・長めfine-tune | ~$1-2・30分 | 実測ベース |
| **B. MoeSpeechベース学習** | キャラ声473話者で事前学習→つくよみfine-tune。ドメインが近い | $27-53+数日（全量）/ ~$10（サブセット） | 未検証 |

- **精度上限はBの可能性が高い**（キャラ声ドメインがつくよみに近い）が、**未検証で$27-53を賭けるリスク**あり
- **話者embeddingが動かない構造的制約はA/B共通** → init話者の選び方が分かれ目
- 推奨順序: A改善で上限を見る → 不足ならBサブセット検証 → 明確に良ければB全量

---

## 7. 成果物一覧（ローカル保全・gitignore）

| ファイル | 内容 |
|------|------|
| `checkpoints/tsukuyomi_finetune_last.ckpt` | つくよみfine-tuneモデル（334.9MB、jvs土台・n_spks=101） |
| `checkpoints/tsukuyomi_init_from_jvs.ckpt` | 話者遷移init（平均init） |
| `checkpoints/tsukuyomi_init_spk80.ckpt` / `_spk85.ckpt` | 近い話者init（未学習・改善用） |
| `checkpoints/wavenext_ja_8000_50kbatch.bin` | MoeSpeech学習WaveNeXt（JVSには不適と判明） |
| `eval/tsukuyomi_synth/text_00〜09.wav` | つくよみ試聴サンプル10文 |

## 8. 新規スクリプト（コミット済み・PR #3）

- `scripts/transfer_speaker_embedding.py`（話者遷移 n→n+1）
- `scripts/prepare_tsukuyomi.py`（つくよみ→precompute filelist）
- `scripts/eval_highband_fidelity.py`（8-11kHz高域忠実度、UTMOSでは測れない領域）
- `scripts/eval_wavenext_fmax_ab.py`（fmax A/B GT-mel再合成）
- `scripts/eval_vocoder_ab.py` に `--wavenext-ja-bin`（ローカルbin評価）
- `wavenext_train/`（WaveNeXt GAN学習・40テスト）
- `configs/experiment/tsukuyomi_finetune.yaml` ほか config一式

## 9. インフラ

- 本セッションで起動したRTX 5090インスタンスは学習ごとに破棄/停止（課金最小化）
- `44216608`（つくよみ学習・**停止済み**、データ保持・数分で再開可）
- ⚠️ 本作業外のインスタンス（`44096702` piper-v8、`44218646`）が稼働中の場合あり（別プロジェクト）
