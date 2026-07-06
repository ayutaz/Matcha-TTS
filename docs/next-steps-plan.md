# 再学習計画（フルスクラッチ）— 日本語サポート完成に向けたロードマップ

最終更新: 2026-07-06 / 対象ブランチ: `feature/japanese-support` (PR #3)

## 0. この計画の前提（重要）

**過去の学習資産は一切前提にしない。** 2026-07-06の調査で以下を確認済み:

- 過去の `jvs_fast`（MAS）500ep / 2500ep checkpointは、HuggingFace（`ayousanz`、private含む全108リポジトリ）にも開発PCにも存在しない
- MAS退化分析の生データ（duration統計JSON等）も同様に残っていない。**数値サマリーのみCLAUDE.mdに記録済み**（退化率 500ep=39.2% / 2500ep=43.0% 等）で、引用は可能
- さらに2026-07の日本語プロソディ表記変更（BREAKING）により、仮に旧資産が見つかっても新コードとは非互換

したがって本計画は、**生のJVSコーパスと空の学習マシンから全工程を再実行する**ランブックとして構成する。学習インフラは **vast.ai のレンタルGPUインスタンス**を使用する（2026-07-06決定）。インスタンスは揮発性（ホスト都合で消滅しうる）のため、データ・checkpoint・評価成果物のHuggingFaceバックアップを全Phaseに組み込む。マイルストーン定義の詳細は [tickets/MILESTONES.md](tickets/MILESTONES.md) を参照（コード実装はM1〜M3、評価ツーリング含めすべて完了済み。残るのは実行のみ）。

---

## 1. 全体ロードマップ

```
Phase 0: vast.aiインスタンス準備（1〜2時間、JVSダウンロード含む）
   ↓
Phase 1: データ準備           （~30分: prepare + Julius + precompute）
   ↓
Phase 2: 学習
   ├─ 2a: jvs_aligned 2500ep（本命、数日規模）── ~500ep時点で中間スモーク評価
   └─ 2b: jvs_fast 500ep    （MASベースライン、任意・推奨。2aの後に実行）
   ↓
Phase 3: 品質評価（M5実行）   （1〜3日）
   ↓
Phase 4: ドキュメント反映・PRマージ
   ↓
Phase 5: マージ後の拡張（任意バックログ）
```

| Phase | 内容 | 目安 | ブロッカー |
|-------|------|------|-----------|
| 0 | vast.aiインスタンス準備 + JVSダウンロード | 1〜2時間 | vast.aiクレジット |
| 1 | データ準備（リサンプル→Julius→precompute） | ~30分 | Phase 0 |
| 2a | `jvs_aligned` 2500ep学習 | 4x RTX 4090想定で~12〜36時間の見込み（要実測） | Phase 1 |
| 2b | `jvs_fast` 500ep学習（ベースライン、任意） | 2aの約1/5 | Phase 1 |
| 3 | 評価サンプル生成 + 全メトリクス | 1〜3日 | Phase 2a（比較評価は2bも） |
| 4 | CLAUDE.md追記・PR更新・マージ | 半日 | Phase 3 |
| 5 | ONNX ja検証・デモ公開等 | 任意 | Phase 4 |

---

## 2. Phase 0: vast.ai インスタンス準備

### インスタンス選定

| 項目 | 推奨 | 備考 |
|------|------|------|
| GPU | **RTX 4090 × 4**（代替: RTX 3090 × 4） | FP32学習のためFP32スループット重視（4090はT4の~10倍）。DDPの実証済みレシピ（4GPU × batch 32 = 有効バッチ128）をそのまま流用できる4枚構成を推奨 |
| GPU 1枚の場合 | 4090 × 1 + `trainer.devices=1 +trainer.accumulate_grad_batches=4` | 有効バッチ128を維持。単価は安いが総時間~4倍 |
| RAM | 64GB以上 | `preload_to_memory=true` + キャッシュで~10GB消費 |
| ディスク | 60GB以上 | JVS生~3GB + リサンプル2種 + precomputed + checkpoint |
| 回線 | down 500Mbps以上目安 | JVSダウンロード・HFバックアップ用 |
| 課金形態 | **on-demand推奨** | interruptible（入札）は安価だがプリエンプトされる。毎epoch保存+HF定期バックアップ体制なら利用可能だが初回はon-demandが無難 |
| イメージ | CUDA対応Ubuntu系なら何でも可（例: vastai/pytorch） | Python依存は `uv sync` が全て導入する。ホスト側はNVIDIAドライバのみ使用 |

価格帯の目安（変動が大きいので作成時に要確認）: 4090 1枚 ~$0.3〜0.5/hr → 4枚 ~$1.2〜2.0/hr。3090はその半額程度。

### セットアップ手順（ssh後、**tmux内で**実行）

```bash
# 基本ツール（rootコンテナが標準のためsudo不要な場合が多い）
apt-get update && apt-get install -y julius perl git curl tmux

# uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# リポジトリ + 依存
git clone -b feature/japanese-support https://github.com/ayutaz/Matcha-TTS.git
cd Matcha-TTS
uv sync --all-groups
bash scripts/setup_julius.sh        # tools/segmentation-kit を配置

# HF CLI（バックアップ用。writeスコープのトークンでログイン）
uv tool install "huggingface_hub[cli]"
hf auth login

# 動作確認
nvidia-smi
julius -help >/dev/null && echo "julius OK"
make test                            # 高速テストスイート（全パスすること）
```

- HiFi-GANボコーダは**学習には不要**（melまでで完結）。Phase 3のサンプル生成時に初回自動ダウンロードされる
- 注意: `run_julius_alignment.py` はsegkitを `tools/segmentation-kit` 固定で参照する。バグ修正済みのブランチ側スクリプトを必ず使うこと（旧コミットのsymlink方式は全件失敗する既知バグ）

### JVSコーパスの調達（3案）

| 案 | 方法 | 位置づけ |
|----|------|---------|
| A | 公式配布元（Google Drive、ライセンス同意）からインスタンスへ直接ダウンロード（~3GB） | 初回 |
| B | jvs_ver1をHF **private**データセット（例: `ayousanz/jvs-ver1-raw`）に一度アップし、以後 `hf download` で取得 | **再構築の定番経路として推奨**。JVSは再配布不可ライセンスのため必ずprivate維持 |
| C | 開発PCから scp/rsync | 手元にデータがあり回線が速い場合 |

推奨運用: 初回は案Aで取得し、**その足で案BのHF private化までやっておく**。インスタンスは使い捨てなので、次回以降の環境再構築を数分にできる。

### 完了条件

- [ ] `nvidia-smi` で想定どおりのGPU構成が見える
- [ ] `make test` 全パス / `julius` OK / `hf auth whoami` が `ayousanz`
- [ ] jvs_ver1（100話者）が展開済みで、HF privateデータセットにもバックアップ済み

---

## 3. Phase 1: データ準備（~30分）

### 手順

```bash
# 1. リサンプル + 無音トリミング + ひらがな転記（22.05kHz学習用 + 16kHz Julius用を同時出力）
uv run python scripts/prepare_jvs.py --jvs-dir /path/to/jvs_ver1 --output-dir data/jvs \
  --julius-output-dir data/julius_work/wav --num-workers 8

# 2. /dev/shmキャッシュ（高速I/O）
bash scripts/setup_shm_cache.sh --full

# 3. Julius並列アライメント + 統合precompute（フルで~3.1分の実績）
uv run python scripts/run_optimized_pipeline.py \
  --filelist data/jvs/train.txt data/jvs/val.txt \
  --output-dir /dev/shm/julius_work \
  --pt-output-dir /dev/shm/jvs_precomputed_aligned \
  --mel-mean -6.550095 --mel-std 2.383771 \
  --num-workers 16 --use-shm

# 4. 永続ディスクにバックアップ（/dev/shmは再起動で消える）
cp -r /dev/shm/jvs_precomputed_aligned data/jvs_precomputed_aligned

# 5. precomputedデータをHF privateへ（インスタンス再作成時にPhase 1を丸ごとスキップ可能にする）
tar czf jvs_precomputed_aligned.tar.gz -C data jvs_precomputed_aligned
hf upload ayousanz/jvs-ver1-raw jvs_precomputed_aligned.tar.gz precomputed/jvs_precomputed_aligned.tar.gz --repo-type dataset
```

**vast.ai注意（/dev/shm）**: Dockerコンテナのshmサイズはホスト・テンプレート依存。`df -h /dev/shm` で確認し、数GB未満なら手順2を省略し、手順3の出力先を `/dev/shm/...` からローカルNVMe（`data/julius_work` / `data/jvs_precomputed_aligned`）に変えて `--use-shm` を外す。fast precomputeはNVMeでも十分速い。学習時の `preload_to_memory=true` はshmではなくプロセスRAMを使うので影響なし。

### mel統計量について

文書化済みの値（`mel_mean: -6.550095` / `mel_std: 2.383771`）は**トリミング済みJVS音声から計算されたもので、音声処理パラメータ（`top_db=30`、50msマージン、22.05kHz）を変えない限り有効**。プロソディ変更はテキスト側のみでmelには影響しない。不安なら `matcha-data-stats` で再計算し、小数第2位程度まで一致することを確認してから進めてもよい（任意）。

### 検証（完了条件）

- [ ] Julius失敗数0（前回実績: 12,997件 / 0エラー。失敗があればログを調査してから進む）
- [ ] `scripts/validate_precomputed_durations.py` で全 `.pt` のシーケンス長整合（`len(x) == len(durations)`）とduration合計 == mel長を確認
- [ ] `scripts/verify_alignment_quality.py` でアライメント品質統計を取得
- [ ] 疑問文サンプル（トークン `?` を含む発話）を数件spot-checkし、`?` にdurationが割り当てられ、`#` `[` `]` がduration=0であることを確認

---

## 4. Phase 2: 学習

### 4.1 Phase 2a: `jvs_aligned` 2500ep（本命）

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python matcha/train.py \
  experiment=jvs_aligned compile_model=false \
  data.batch_size=32 data.num_workers=0 +data.preload_to_memory=true \
  test=false
```

- **fresh run**（`ckpt_path` なし）。設定は `configs/experiment/jvs_aligned.yaml`: max_epochs=2500、FP32（`32-true`）、lr=1e-4固定・scheduler無し、EMA（decay=0.9995、epoch 10から）、`use_precomputed_durations=true`（MAS完全バイパス）、FiLM DP + Blank zero-init有効
- 中断時は `ckpt_path=logs/train/jvs_aligned/runs/<run_dir>/checkpoints/last.ckpt` でフル再開可能（毎epoch保存済み）

**モニタリング**:

- `dur_loss`（最重要 — 正確なJuliusターゲットに対するDPの収束）、`prior_loss`、`diff_loss` をTensorBoardで監視
- `scripts/check_training_health.py` で定期ヘルスチェック（NaN・スパイク検出）
- **~500ep時点の中間チェック**: `last.ckpt` から数話者×数文をスモーク合成し、(1) 発話が明瞭か、(2) phoneme median durationが2フレーム超か、(3) blank[0]が暴走していないかを確認。異常があれば2500epを待たずに調査

**確認ポイント**:

- early_stoppingの実効patience: `check_val_every_n_epoch=10` × `patience=30` = 「300epoch改善なし」で停止する。dur_lossが素直に収束する経路なので早期停止は起きにくい見込みだが、意図せず停止した場合は `loss/val` の推移を見てpatience調整を判断
- CLAUDE.mdの既知の罠を再掲: FP16禁止 / `out_size=null` 必須 / 4GPU DDPでは `compile_model=false`、`data.batch_size=32` が安定上限

### 4.2 Phase 2b: MASベースライン `jvs_fast` 500ep（任意・推奨）

過去のMASサンプル・checkpointが失われたため、比較評価をするならベースラインも新conventionで再学習する必要がある:

| 案 | 内容 | コスト | 推奨 |
|----|------|-------|:---:|
| B' | `jvs_fast` を**500epのみ**再学習してベースラインにする。MAS退化は500epで既に発現する（前回実績39.2%）ため、退化率・UTMOS・ABテストの比較対象として十分 | 2aの約1/5 | **○** |
| C | ベースライン学習を省略し、UTMOS絶対値（>3.5）+ 退化率0%のみで判定。MASの数値はCLAUDE.mdの過去測定値を引用 | ゼロ | 次善 |

案B'のコマンド（Phase 2aの完了後に実行）:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python matcha/train.py \
  experiment=jvs_fast compile_model=false \
  data.batch_size=32 data.num_workers=0 +data.preload_to_memory=true \
  test=false trainer.max_epochs=500
```

副産物として、新conventionでのMAS退化率を`jvs_aligned`と同一データで再測定でき、PRの核心主張（MAS退化のアルゴリズム起因性）の再現証拠にもなる。

### 4.3 checkpointバックアップ（必須運用）

**今回の教訓（過去モデル消失）+ vast.aiインスタンスの揮発性を踏まえ、インスタンス単独保管を禁止する:**

```bash
# HF privateリポジトリを作成（初回のみ）
hf repo create matcha-tts-jvs-ja --private

# 学習中の定期バックアップ（tmuxの別ペインで起動しっぱなしにする）
while true; do
  RUN_DIR=$(ls -td logs/train/jvs_aligned/runs/* | head -1)
  hf upload ayousanz/matcha-tts-jvs-ja "$RUN_DIR/checkpoints/last.ckpt" jvs_aligned/last.ckpt
  sleep 10800   # 3時間ごと（プリエンプト・ホスト障害への保険）
done

# 学習完了時: 設定スナップショットとTensorBoardログも同梱
hf upload ayousanz/matcha-tts-jvs-ja "$RUN_DIR/.hydra" jvs_aligned/hydra
hf upload ayousanz/matcha-tts-jvs-ja "$RUN_DIR" jvs_aligned/tensorboard --include "*.tfevents.*"
```

### 完了条件

- [ ] 2a: 2500ep完走（またはval lossの明確なプラトーでの収束停止）、NaN・発散なし
- [ ] 2a: `last.ckpt` が新コードでstrict loadできる
- [ ] 2a/2b: checkpointをHF privateにアップロード済み
- [ ] （案B'採用時）2b: 500ep完走

---

## 5. Phase 3: 品質評価（M5実行）

判定基準の詳細は [tickets/M5_evaluation.md](tickets/M5_evaluation.md) を正とする。ツーリングは実装済みで、**実行と判定**が本Phaseの作業。

### 5.1 サンプル生成

```bash
uv run python scripts/generate_eval_samples.py \
  --checkpoint logs/train/jvs_aligned/runs/<run_dir>/checkpoints/last.ckpt \
  --text-file eval/texts_ja.txt --output-dir eval/samples/julius_model
# 案B'採用時はベースラインからも同一条件で生成
uv run python scripts/generate_eval_samples.py \
  --checkpoint logs/train/jvs_fast/runs/<run_dir>/checkpoints/last.ckpt \
  --text-file eval/texts_ja.txt --output-dir eval/samples/mas_baseline
```

100話者 × 10文 = 1,000サンプル。clamp_boundary_blanks 有り/無しの2セットを生成して比較（Julius duration学習後はclamp不要になる見込み → 結果に基づきデフォルト変更を判断）。

### 5.2 メトリクス実行と合格基準

| メトリクス | スクリプト | 合格基準（M5より） |
|-----------|-----------|------------------|
| 退化率 | `eval_degeneration_rate.py` | **0%**（MASベースライン: 39〜43%） |
| phoneme 1フレーム以下率 | 同上 | < 10%（MAS: 62.9%） |
| blank[0] duration平均 | 同上 | < 10フレーム（MAS退化時: 101） |
| Duration精度 | `eval_duration_accuracy.py` | Pearson r > 0.85、RMSE < 3.0フレーム |
| MCD | `eval_mcd.py` | < 7.0 dB、かつHiFi-GAN再合成下限（1.5〜2.5 dB）を併記 |
| UTMOS（主軸） | `eval_utmos.py` | 平均 > 3.5。案B'採用時はpaired t-test（p < 0.01）でベースライン超え |
| ABテスト（補助・任意） | `prepare_ab_test.py` / `analyze_ab_test.py` | 評価者10名以上を確保できる場合のみ。困難ならUTMOSのみで判定可 |
| 統合レポート | `generate_eval_report.py` | JSON + 図一式が `eval/report/` に出力される |

- 補足: 話者類似度評価（`eval_speaker_similarity.py`）は**未実装**（M5チケットに記載はあるがスクリプトなし）。UTMOS・MCD・退化率で判定可能なため任意項目とし、必要になった時点で実装する。

### 完了条件

- [ ] 退化率0%を数値で確認（最重要 — 本ブランチの核心的主張）
- [ ] UTMOS > 3.5（全体平均）
- [ ] duration精度・MCDが基準内
- [ ] `eval/report/eval_report.json` + 図が生成され、**HFリポジトリにも保管**（今後のベースラインとして）

---

## 6. Phase 4: ドキュメント反映・PRマージ

- [ ] **CLAUDE.md更新**: 「MASアライメント退化問題」セクションに解決結果（退化率・UTMOS・MCD実測値）を追記。clamp_boundary_blanks の要否結論、確定した推奨推論パラメータ（n_timesteps / temperature / length_scale）を反映
- [ ] **MILESTONES.md / チケットの状態更新**: M4・M5を完了に
- [ ] **PR #3 本文更新**: Test planの未チェック項目「`jvs_aligned` 本学習 + 品質評価」をチェックし、評価サマリーを追記
- [ ] **マージ**: CI最終確認 → `main` へマージ（マージ方式はリポジトリ運用に従う）

---

## 7. Phase 5: マージ後の拡張（任意バックログ）

優先度順ではなく候補リスト。着手時に個別にチケット化する。

- **ONNX日本語エクスポートの実機検証**: `matcha.onnx.export` → `infer` を新checkpointで通し、INT8量子化の品質確認
- **デモ・モデル公開**: Gradioデモ（`matcha-tts-app`）に日本語モデルを同梱、HF checkpointのpublic化検討
- **HiFi-GANの日本語fine-tune検討**: Phase 3のMCD分析でボコーダ下限がボトルネックと判明した場合（BigVGAN移行も選択肢）
- **話者類似度評価の実装**: ECAPA-TDNN等でembedding抽出、100×100類似度行列（M5チケット記載分）
- **Duration Predictor改善の続き**: 受容野拡大（dilated conv）等 — 現行5トークンは日本語プロソディには狭い
- **単話者高品質モデル**: JSUT等でのfine-tune（`prepare_jsut.py` は既存）

---

## 8. リスクと対策

| リスク | 影響 | 対策 |
|--------|------|------|
| **vast.aiインスタンス消滅**（ホスト障害・プリエンプト・停止忘れからの回収） | 学習中断・データ喪失 | 毎epoch保存 + 3時間ごとHFアップロード（§4.3）。JVS生データ・precomputedもHF private化（Phase 0/1）して再構築を数分に |
| コスト超過 | 予算圧迫 | 学習開始後10epochで epoch時間を実測 → 総時間・総額を見積もってから継続判断。評価だけの期間は安い1GPUインスタンスに切り替え。アイドル時はインスタンスをdestroy（stopでもストレージ課金は残る） |
| checkpoint・成果物の再消失 | 再々学習 | §4.3のHFバックアップを完了条件に組み込み済み。/dev/shm上のデータも必ず永続ディスクへコピー |
| Julius環境差異（バイナリ版数・segkit配置） | アライメント全滅 | Phase 0の動作確認 + Phase 1の「失敗数0」ゲートで検出。ブランチ側の修正済みスクリプトを使用 |
| mel統計量のずれ（prepare_jvsパラメータ変更時） | 学習品質劣化 | パラメータを変えない限り文書値が有効。変えた場合は `matcha-data-stats` で再計算必須 |
| early stoppingによる意図しない早期終了 | 学習不足 | §4.1の確認ポイント参照。val loss推移を見てpatience調整 |
| 長期学習中の障害（NCCLタイムアウト等） | 学習中断 | 毎epoch保存済み → `ckpt_path=last.ckpt` でフル再開。NCCLタイムアウトは7200秒設定済み |
| UTMOS < 3.5 の場合 | 品質不足 | ボコーダ下限MCDで切り分け（モデル起因かボコーダ起因か）→ 推論パラメータ掃引（n_timesteps増等）→ それでも不足ならPhase 5のDP改善・ボコーダfine-tuneへ |
| 旧convention資産の誤使用 | 静かな品質劣化 | フルスクラッチ前提なので原則発生しないが、学習マシンに旧 `data/jvs_precomputed*` が残っていた場合は先にリネーム退避 |
| torchaudio非互換（PyTorch 2.10+） | 前処理失敗 | soundfileフォールバック実装済み（既知対応） |

---

## 9. タイムライン・コスト概算

| 日程 | 作業 |
|------|------|
| Day 1 | インスタンス作成 + Phase 0（1〜2時間）→ Phase 1（~30分）→ **10epoch試走でepoch時間を実測** → 総額見積もりを確認して Phase 2a 継続 |
| Day 1〜3 | Phase 2a: `jvs_aligned` 2500ep（4x 4090想定で~12〜36時間の見込み。~500ep時点で中間スモーク評価） |
| 続き | Phase 2b: `jvs_fast` 500epベースライン（案B'採用時、2aの約1/5） |
| +1〜3日 | Phase 3: 評価（サンプル生成・メトリクスはGPU 1枚で十分 → 安いインスタンスに切り替え可） |
| +半日 | Phase 4: ドキュメント反映・PR #3マージ（開発PCで実施） |

**コスト概算（10epoch実測で必ず確定させる）**: 4x 4090 on-demand ~$1.2〜2.0/hr × 12〜36h ≈ **$15〜70**。案B'（2b）込みでも**~$85以下**の見込み。前回実績は4x T4で「数日」であり、FP32スループット~10倍の4090なら大幅短縮が期待できるが、データローダ律速の可能性もあるため試走実測を継続判断のゲートにする。

**試走の手順**: Phase 2aのコマンドをそのまま起動し、10epoch分のepoch時間をログ（progress bar / TensorBoard）から読み取る → `総時間 ≈ epoch時間 × 2500`。想定を大きく超える場合は一度停止し、GPU使用率を確認してボトルネック（データローダ等）を調査してから再開する。試走はそのまま本学習として継続してよい（停止・再開しても `ckpt_path=last.ckpt` で無駄にならない）。

---

## 10. すぐに着手できる次のアクション

1. **vast.aiのクレジット準備 + インスタンス作成**（§2の推奨スペック: 4x 4090 / RAM 64GB+ / ディスク60GB+ / on-demand）
2. Phase 0 → Phase 1（合計~2時間。JVSとprecomputedのHF privateバックアップまで済ませる）
3. **10epoch試走** → epoch時間から総額を見積もり、継続判断
4. Phase 2a起動 + 定期バックアップループ起動（§4.3、tmux別ペイン）
5. 学習中に Phase 2b（案B' vs 案C）を決定 — 推奨は案B'（比較評価の説得力が大きく上がり、コストは本学習の1/5）
