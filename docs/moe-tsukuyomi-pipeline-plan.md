検証完了。以下の事実を実コードで確認した: precompute_dataset.py L37 `F_MAX=8000` / L83-84 の `{parent.name}_{stem}.pt` 命名衝突 / L165 strict load、train.py L155 の optimizer_states 分岐、matcha_tts.py L54 の `if n_spks>1` 話者埋め込み、BucketBatchSampler L51 `bucket_size=max(1,len//num_buckets)` + L82-83 drop_last 断片化バグ。これらを反映した統合計画を出力する。

---

## ★方針変更（2026-07-08 確定）: fmax引き上げは見送り、fmax=8000 を維持する

**本計画は当初 fmax=8000→11025 引き上げ（濁り改善）を前提に書かれているが、実験の結果 fmax引き上げは見送ることに決定した。** 以下の記述で fmax=11025 とあるのは fmax=8000（既存デフォルト）に読み替える。

### 判断の根拠（Phase W 交絡排除実験、2026-07-08）
WaveNeXtボコーダを MoeSpeech 3話者・50000バッチで **fmax=11025 と fmax=8000 の2本**学習し、同一val 40 wav で高域(8-11kHz)忠実度を比較（`scripts/eval_highband_fidelity.py`）:

| 指標 | fmax=11025 | fmax=8000（同データ学習） |
|------|:---:|:---:|
| 高域(8-11kHz) log-STFT L1 | **0.825**（低い=GTに近い） | 1.031 |
| 高域エネルギー比（1が理想） | **0.851** | 0.553（45%欠損） |
| paired 高域L1 | fmax=11025が40/40でGTに近い | — |

- **客観的には fmax引き上げは独立して効く**（同データ・同ステップでもfmax=8000は高域45%欠損のまま。データ変更では埋まらない）
- **しかしユーザ試聴では体感差なし**（同一サンプルの純粋fmax差A/B、`eval/pure_fmax_listen/`）
- → **体感差がないのに、fmax引き上げの破壊的変更（mel統計再計算・全前処理やり直し・音響モデル全再学習 $27-53）を払うのは割に合わない**と判断
- 最終判断はユーザの聴感を優先（UTMOSより試聴を決め手にする方針と一貫）

### fmax=8000 維持による帰結（この計画への修正）
- **precompute の `--fmax` は既存デフォルト 8000 のまま**（moe/tsukuyomi の mel も fmax=8000）
- **mel統計は fmax=8000 で算出**（`prepare_moespeech.py stats` は F_MAX=11025 ハードコードを 8000 に直すか `--fmax` 化が必要 → §後述の未対応事項）
- **JVS mel統計（-6.550095 / 2.383771）が fmax=8000 で整合** → 一貫チェーンが既存資産と互換
- **ボコーダは今回学習した fmax=8000 版**（`checkpoints/wavenext_ja_8000_50kbatch.bin`、MoeSpeech日本語適合済み）を初期値/採用候補に使える
- 「日本語データでのボコーダ+音響モデル再学習」ぶんの濁り改善（実測で存在）は引き続き得られる
- **破壊的変更が消え、既存fmax=8000パイプライン・JVS資産と互換になる**（実装リスク・コスト大幅減）

### fmax=8000 維持で要修正の実装箇所（着手時に対応）
- `scripts/prepare_moespeech.py` L52 `F_MAX = 11025` → 8000（または `--fmax` CLI化）。stats/precompute の mel が fmax=8000 になる
- `configs/data/moespeech_precomputed.yaml` / `tsukuyomi_precomputed.yaml` のコメント「fmax=11025統計」を fmax=8000 に
- `configs_wavenext/wavenext_11025.yaml` は fmax=11025 用。fmax=8000 で使うなら `fmax: 8000` 版configで（今回の実験で実証済みの sed 変換と同じ）
- placeholder ガード（train.py D4）はそのまま有効

---

# MoeSpeech 事前学習 → つくよみちゃん fine-tune 実装計画（~~fmax=11025~~ **fmax=8000維持** / 非破壊）

本計画は 5 つの実装 spec と敵対的検証（全て `NEEDS_REVISION`）を統合し、verifier の corrections を全て反映した**実行可能な確定計画**である。**上記の方針変更により fmax は 8000 を維持する**（本文の fmax=11025 は 8000 に読み替え）。品質レシピ（lr=1e-4 / out_size=null / prior重み1.0 / uniform sampling / EMA 0.9995 / bf16-mixed + fused=false）は一切変更しない。既存 JVS 資産（`jvs_*` config/experiment、`jvs_precomputed*`、既存 `.pt`、既存 vocoder）は無改変で、全成果物は新規 file とする。

---

## 0. 統合で確定した設計判断（verifier corrections 反映）

5 spec は個別には妥当だが、相互の前提に **5 つの構造的欠陥**があり、それを次のとおり確定解決する。これが全フェーズの前提。

| # | 検証で露呈した欠陥 | 確定判断（本計画の方針） |
|---|---|---|
| D1 | **話者遷移が n_spks=473→1 を仮定**しており、spk_emb/FiLM/proj_m/proj_w/encoder先頭/decoder先頭の shape が全変化 → strict load 不能。`transfer_from_english.py` は流用不可 | **n_spks=1 化しない。** 多話者アーキ（n_spks>1 の bool 分岐）を保持し、`spk_emb.weight` を **473→474 にリサイズ**（新行 = つくよみ slot 473）。他テンソルは byte-identical。verifier が「VERIFIED CONSISTENT」と確認した唯一の安全経路 |
| D2 | **checkpoint load 規約**: `optimizer_states` があると train.py L155 で full resume（strict, epoch 2500 継続）に入り fine-tune が走らない | 遷移スクリプトは **weights-only（optimizer_states/averaging_state/current_model_state を除去）**の `{state_dict}` のみ保存 → L158 の weights-only 分岐（strict load → fresh optimizer/epoch0）に入る |
| D3 | **.pt 命名衝突**: precompute_dataset.py L83-84 は `{parent.name}_{stem}.pt`。MoeSpeech `<spk>/wav/<file>.wav` だと parent 名が全話者 `wav` で無言上書き=データ喪失 | MoeSpeech は**専用 `prepare_moespeech.py`** で処理し、`.pt` は **`{spk_id}_{stem}.pt`（spk-id 前置）** で出力。prepare 段で stem グローバル一意性を assert（非一意なら fail-fast） |
| D4 | **fmax=11025 mel統計の算出ツールが存在しない**。`matcha-data-stats` は TextMelDataModule 専用で precomputed config に使えない。config は placeholder 0.0/1.0 のまま | `prepare_moespeech.py stats`（二段階 two-pass、生 log-mel）で算出。**placeholder(0.0/1.0) のまま学習を弾く起動ガード**を追加。**Tsukuyomi は Moe の統計を再利用**（strict load で model buffer が Moe 値に上書きされ denormalize と precompute 正規化が一致するため必須） |
| D5 | **max_epochs=2500 を JVS から丸写し**。Moe は数十万発話（JVS の 20 倍超）で 2500ep = 数百万 step = 過学習+課金爆発 | max_epochs は**データ量非依存の品質定数ではない**。総ステップ予算（目標 150–300K step）から `max_epochs = 目標step × 有効batch / N_utt` で逆算し CLI 上書き |

その他の確定修正（各フェーズに織り込み済み）:
- **MAS 多話者退化ゲート**（D6）: MoeSpeech は MAS 採用（ユーザ確定）だが、CLAUDE.md 記載の退化リスク（JVS 100話者で 39–43%）が 473話者で増幅。**サブセットで退化率を実測し閾値超なら Julius へフォールバック**する分岐を計画に明記。
- **ASR 転記サニタイズ**（D7）: anime_whisper のノイズ転記は KeyError 無言ドロップ/誤読を招く。NFKC 正規化 + 注釈除去 + 日本語含有チェック + 長さフィルタ + g2p dry-run を prepare に組込み、ドロップ率をログ。
- **無音トリミング**（D8）: JVS 同様 `top_db=30` を prepare で適用（未トリム = mel統計歪み + MAS blank[0] 爆発）。
- **test=false**（D9）: PrecomputedTextMelDataModule に test_dataloader が無く、既定 `test=True` で末尾クラッシュ。全学習コマンドに `test=false`。
- **tsukuyomi num_buckets=1**（D10）: ~90発話 × batch16 で num_buckets=20 だと bucket_size=4 → drop_last で全 batch 消失（実コード L51/L82 で確認）。`num_buckets=1` に修正。`save_on_train_epoch_end=false` も付与。
- **WaveNeXt**（D11）: manual optimization の global_step 会計バグ回避、`y_hat.detach()`、`transformers` 依存を排し `LambdaLR` 自前実装、stft を `autocast(enabled=False)`、wetdog raw の逐語検証ゲート。

---

## 1. 実装フェーズの順序と依存関係

### 1.1 依存グラフ

```
Phase 0 (fmax infra, ローカル$0) ─────────────────────────┐
   │ precompute_dataset.py --fmax / precompute_with_alignment.py --fmax
   │ JVS byte-identity 検証
   ▼
Phase 1 (MoeSpeech data prep, ローカル$0)
   │ prepare → select(subset) → stats(fmax=11025) → precompute
   ▼
Phase 2 (subset smoke + 退化率ゲート, インスタンス・安価)
   │ 配管検証 / MAS 退化率実測 → 【判断ゲート: MAS継続 or Julius切替】
   ▼
Phase 3 (full pretrain, インスタンス・最高額)  ◀── stats/precompute(全量) はローカル$0
   │ base ckpt (n_spks=473, fmax=11025, mel統計焼込み) を出力
   ▼
Phase 4 (話者embedding遷移, ローカル$0)
   │ transfer_speaker_embedding.py: 473→474 resize, weights-only
   ▼
Phase 5 (つくよみ fine-tune, インスタンス・安価)
   │ prepare_tsukuyomi → precompute(Moe統計再利用) → finetune (単一GPU)
   ▼
Phase E (評価 + ONNX)  ◀── Phase W 完了が前提

Phase W (WaveNeXt fmax=11025 再学習, インスタンス) ── Phase 0 完了後は
   GT 波形→mel(11025) で学習でき、Phase 1-5 と【並行実行可能】
```

**クリティカルパス**: Phase 0 → 1 → 2(ゲート) → 3 → 4 → 5 → E。
**並行トラック**: Phase W は Phase 0 完了直後から着手可（音響モデルの完成を待たない。GT 波形のみで学習・GT-mel 再合成で先行評価できる）。

### 1.2 ローカル$0 / インスタンス課金の分離

| フェーズ | ローカル $0（CPU のみ） | インスタンス課金（GPU） |
|---|---|---|
| Phase 0 | 全て（コード編集 + JVS byte-identity 検証） | — |
| Phase 1 | 全て（DL / resample / trim / 転記sanitize / stats / precompute） | — |
| Phase 2 | .pt 目視・schema 検証 | サブセット smoke（4×5090 or 1×5090, ~1-3h） |
| Phase 3 | 全量 stats / precompute | 全量 pretrain（4×5090 DDP, 最高額） |
| Phase 4 | 全て（spk_emb resize, 数秒。strict-load 検証） | — |
| Phase 5 | prepare/precompute/config/合成検証 | fine-tune（**単一** 5090, ~数十分-1h） |
| Phase W | 実装 + init/extract/filelist + 数値検証 | WaveNeXt 学習（1×5090 複数 run） |
| Phase E | ONNX export | UTMOS 評価 / GT-mel 再合成 |

**鉄則**: 課金前に pass1(stats)→pass2(precompute)→少数 .pt 目視→スモーク学習までローカルで通す。パイプラインバグは全てローカルで潰す。

---

## 2. fmax=11025 一貫性チェックリスト（最重要・全ての土台）

fmax の**真の単一ソース**は `matcha/utils/audio.py::mel_spectrogram(..., fmax)` のフィルタバンク（L54 のキャッシュキーに fmax 込み=stale化なし）。ここに何を渡すかが全て。fmax は **precompute 時点で完全確定**し、学習時にはノブが存在しない（PrecomputedTextMelDataModule は焼き込み済み mel を読むだけ）。

### 2.1 一貫チェーン（1 箇所でもズレると無言破綻）

```
precompute F_MAX=11025
  → 同 fmax の生 log-mel で mel_mean/std 算出
    → その統計で normalize して .pt に焼込み
      → その .pt で学習したモデル buffer(mel_mean/std)  ← strict load で上書き
        → synthesise() の denormalize(mel, mel_mean, mel_std)
          → fmax=11025 の mel を逆変換する WaveNeXt(fmax=11025 再学習版)
```

### 2.2 fmax がズレうる全箇所と本計画での扱い

| # | 箇所 (file:line) | fmax の扱い | 本計画での処置 |
|---|---|---|---|
| 1 | `matcha/utils/audio.py:45/54` | 引数（唯一の実フィルタバンク源、キャッシュキーに fmax 込み） | **コード変更不要** |
| 2 | `scripts/precompute_dataset.py:37` `F_MAX=8000` | ハードコード → CLI 化 | **Phase 0 で `--fmax`（default=8000）**。Tsukuyomi は 11025 指定 |
| 3 | `scripts/precompute_with_alignment.py:78` `F_MAX=8000`（Julius経路） | ハードコード → CLI 化 | **Phase 0 で `--fmax`（default=8000）**。Julius フォールバック時の 8000 焼込み罠を消す |
| 4 | `scripts/prepare_moespeech.py`（新規） `F_MAX=11025` | 定数 | Moe の唯一の焼込み点。stats/precompute で共有 |
| 5 | mel統計（`prepare_moespeech.py stats`） | 11025 で算出 | precompute と**同一 mel_spectrogram 呼び出し**でドリフトゼロ |
| 6 | `configs/data/moespeech_precomputed.yaml` `data_statistics` | Moe 11025 実測値を焼込み | **f_max ライブキーは置かない**（PrecomputedTextMelDataModule は読まない=ドリフト罠）。コメントで `stats are fmax=11025` 明記 |
| 7 | `configs/data/tsukuyomi_precomputed.yaml` `data_statistics` | **Moe と同一値**（Tsukuyomi 自前ではない） | strict load で model buffer が Moe 値に上書きされるため必須 |
| 8 | `matcha/wavenext/vocoder.py:5,22` docstring `fmax=8000` | doc のみ | **doc 更新 + fmax=11025 で再学習した checkpoint に差替え**（現 BSC は 8000 束縛=非互換） |
| 9 | `matcha/hifigan/config.py:24` `fmax:8000` | HiFi-GAN 学習用（Matcha 推論の to_waveform は未使用） | WaveNeXt 採用でスコープ外。**ただし既定 HiFi-GAN も 11025 mel と非互換**=評価時にガード |

### 2.3 ドリフト検出方法（実行可能）

1. **JVS 非破壊（byte-identity）**: JVS wav 数件を `--fmax` 省略（=default 8000）で再 precompute → 既存 `.pt` と `torch.load` して `mel diff == 0.0`。
2. **fmax=11025 の効き**: 同一 wav を `--fmax 8000` と `11025` で precompute し、上位 mel ビン（8–11kHz 帯）が 11025 側で非ゼロを確認。`librosa` フィルタバンクは 11025 と `None`(=sr/2) が完全一致、8000 とは非一致（実測済み）。
3. **統計の自己整合（正規化サニティ）**: pass1 の mel_mean/std で焼いた `.pt` 群からランダムに数十件、mel 全体の平均≈0・標準偏差≈1 を確認。**逆に旧 JVS 統計(-6.550095/2.383771)で 11025 mel を正規化すると 0/1 から外れる**ことも確認（退行検知）。
4. **placeholder ガード**: 学習起動時に `mel_std==1.0 and mel_mean==0.0` なら abort（Phase 3/5 の config に対する起動チェック）。
5. **denormalize 対称性**: 保存前 mel を `denormalize(mel, mel_mean, mel_std)` すると生 log-mel に戻る（1 サンプル）。

---

## 3. 各フェーズの確定 spec

### Phase 0 — fmax インフラ整備（ローカル $0）

**変更 file: `scripts/precompute_dataset.py`**（非破壊、default=8000）
- L37 `F_MAX=8000` はモジュール定数として後方互換値で残す。
- argparse に追加:
  ```python
  parser.add_argument("--fmax", type=int, default=8000,
      help="Mel fmax(Hz). JVS=8000(既存byte-identical), MoeSpeech/Tsukuyomi=11025(Nyquist)")
  ```
- `process_sample(..., durations_dir=None, fmax=F_MAX)` にパラメータ追加（**末尾に default 付きで追加** → tests/test_precompute_dataset.py の 7 引数 positional 呼び出しと後方互換）。本文の `mel_spectrogram(audio, N_FFT, N_MELS, SAMPLE_RATE, HOP_LENGTH, WIN_LENGTH, F_MIN, F_MAX, center=False)` を `..., F_MIN, fmax, center=False)` に。
- CPU 経路 `executor.submit(process_sample, ..., Path(args.durations_dir) if args.durations_dir else None, args.fmax)`。
- GPU 経路（L248 付近）の `F_MAX` を `args.fmax` に置換。

**変更 file: `scripts/precompute_with_alignment.py`**（任意だが推奨、Julius フォールバック用）
- 同型で `--fmax`（default=8000）を CLI 化（L78/197/376/470 の F_MAX を通す）。moe/tsukuyomi は MAS 採用で一次不使用だが、Phase 2 ゲートで Julius 切替になった場合に 8000 焼込み罠を消すため事前に入れておく。

**検証**: §2.3 の 1・2。

---

### Phase 1 — MoeSpeech データ準備（ローカル $0）

**新規 file: `scripts/prepare_moespeech.py`**（4 サブコマンド: prepare / select / stats / precompute）

冒頭定数（precompute_dataset.py と同一、F_MAX のみ変更）:
```python
N_FFT=1024; N_MELS=80; SAMPLE_RATE=22050; HOP_LENGTH=256; WIN_LENGTH=1024
F_MIN=0.0; F_MAX=11025
SRC_SR=44100
REPO_ID="ayousanz/moe-speech-plus"
```
import（**verifier 指摘: random / defaultdict を必ず追加**）:
```python
import argparse, io, json, os, re, random, unicodedata, zipfile, time
from collections import defaultdict
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np, soundfile as sf, torch, torchaudio, librosa
from tqdm import tqdm
from huggingface_hub import HfApi, hf_hub_download
from matcha.text import text_to_sequence
from matcha.utils.audio import mel_spectrogram
from matcha.utils.model import normalize
from matcha.utils.utils import intersperse
```

**(a) 転記サニタイズ（D7、g2p を絶対に落とさない三重防御）**
```python
_HAS_JP = re.compile(r"[぀-ヿ㐀-䶿一-鿿]")
_ANNOT  = re.compile(r"[（(【\[「『][^）)】\]」』]*[）)】\]」』]|[♪♩♫♬※→←↑↓…‥〜~＿_]+")

def clean_transcript(t):
    if not t: return None
    t = unicodedata.normalize("NFKC", t)   # 全角英数/半角ｶﾅ 正規化
    t = _ANNOT.sub("", t)                  # （笑）【SE】♪ 等の注釈除去
    t = re.sub(r"\s+", "", t)
    return t.strip() or None

def accept_transcript(t, dur, min_c=2, max_c=140, min_d=0.4, max_d=14.0):
    if not t: return False
    if not _HAS_JP.search(t): return False       # 日本語文字ゼロ→除外
    if not (min_c <= len(t) <= max_c): return False
    if dur is not None and not (min_d <= float(dur) <= max_d): return False
    return True

def g2p_ok(t):                                    # 最終防御: KeyError等で全体を止めない
    try:
        seq, _ = text_to_sequence(t, ["japanese_cleaners"], language="ja")
    except Exception:
        return False
    return len(seq) > 0
```
採用転記は **`anime_whisper_transcription`（確定）**。`speechMOS` ではフィルタしない（ゲーム音声で過小評価）。`duration` は長さフィルタのみ。除外率をログ出力し、閾値（例 >5%）を超えたら `_ANNOT` を強化。

**(b) prepare サブコマンド**（DL→展開→sanitize→**無音トリム(D8)**→resample→per-speaker wav 書出し）
- 話者 id は zip 名 `sorted` で決定的に `spk_to_id = {Path(z).stem: i for i,z in enumerate(zips)}`（0..472）→ `speakers.json` 出力。
- 1 zip = 1 worker（ProcessPool）。`hf_hub_download` で個別 DL → 処理後 `os.remove`（ストリーミング削除で disk 節約）。
- **resampler キャッシュ（D-resampler バグ修正）**: ループ内で上書きせず `self._resamplers.setdefault(sr, Resample(sr, SAMPLE_RATE))` で sr キーにキャッシュ取得。
- **無音トリム**: 22.05k 書出し前に `librosa.effects.trim(y, top_db=30)`（JVS `prepare_jvs.py` と同条件、50ms マージン）。
- **命名衝突回避（D3）**: wav を `wav_root/<speaker_name>/<stem>.wav` に書き、`stem` はグローバル一意（`<spk>_NNN`）を assert。非一意なら fail-fast。
- manifest 各行 `{wav, spk, text}`（text = sanitize 済み。**precompute での二重 g2p を避けるため cleaned_text も保持推奨**）を `manifest.jsonl` に追記。

**(c) select サブコマンド**（サブセット/全量を同一スクリプトで駆動）
```python
def cmd_select(manifest, out, max_speakers, max_utts_per_spk, max_total, seed=42):
    rows = [json.loads(l) for l in open(manifest, encoding="utf-8")]
    rng = random.Random(seed); by_spk = defaultdict(list)
    for r in rows: by_spk[r["spk"]].append(r)
    spks = sorted(by_spk)
    if max_speakers: spks = spks[:max_speakers]      # id昇順で決定的
    sel = []
    for s in spks:
        u = by_spk[s]; rng.shuffle(u)
        if max_utts_per_spk: u = u[:max_utts_per_spk]
        sel += u
    rng.shuffle(sel)
    if max_total: sel = sel[:max_total]
    with open(out, "w", encoding="utf-8") as f:
        for r in sel: f.write(json.dumps(r, ensure_ascii=False)+"\n")
```
**重要**: サブセットでも **spk id は 0..472 のグローバル id を保存**（remap しない）。これで「サブセット ckpt → 全量 resume」の埋め込み id 空間が一致し、Phase 4 の遷移でも id が保存される。config の `n_spks` は常に 473。

**(d) stats サブコマンド**（fmax=11025、**生 log-mel**、二段階 two-pass = 数値安定 D4）
```python
def _raw_mel(wav):
    d, sr = sf.read(wav, dtype="float32")
    if d.ndim > 1: d = d.mean(axis=1).astype("float32")
    assert sr == SAMPLE_RATE
    return mel_spectrogram(torch.from_numpy(d)[None,:], N_FFT,N_MELS,SAMPLE_RATE,
           HOP_LENGTH,WIN_LENGTH,F_MIN,F_MAX,center=False).squeeze()  # 正規化しない
# pass1: mean = Σmel/(frames*n_mels)、pass2: std = sqrt(Σ(mel-mean)²/(frames*n_mels))
# frac<1.0 でサブサンプル可（全量は 0.1 で十分収束）。pass1/pass2 は同一 seed の同一 rows
```
式は `generate_data_statistics.py::compute_data_statistics` と同一。出力 `stats.json = {mel_mean, mel_std, fmax:11025, n_utts}`。

**(e) precompute サブコマンド**（mel(11025)+normalize+.pt、MAS=durations 無し）
- **.pt 命名（D3）**: `out = d / f'{r["spk"]}_{Path(r["wav"]).stem}.pt'`（spk-id 前置で衝突耐性）。
- `torch.save({"mel":mel, "text":intersperse後 IntTensor, "spk":int(spk), "cleaned_text":cleaned}, out)`（durations キー無し）。
- **スループット（D-perf）**: 各サンプルで `text_to_sequence` を都度実行する ProcessPool は ~8 samples/sec（legacy 相当）。**cleaned_text を manifest から再利用**（prepare で保存済み）するか、`build_text_sequence_cache` を流用して ~168 samples/sec に。全量数十万発話ではこれが必須。
- try/except で g2p 例外は skip（無言ドロップだが件数をログ）。
- **val split**: `val_frac=0.01` で train/val 分割。各 split は `os.scandir` 非再帰で読むため flat 配置。

**実行コマンド（ローカル $0）**
```bash
# (A) prepare（473 zip DL→trim→22.05k wav + manifest。zip は逐次削除）
uv run python scripts/prepare_moespeech.py prepare --work-dir data/moespeech --num-workers 16
# (B) サブセット選択（例 20話者×300発話）
uv run python scripts/prepare_moespeech.py select \
  --manifest data/moespeech/manifest.jsonl --out data/moespeech/manifest_subset.jsonl \
  --max-speakers 20 --max-utts-per-spk 300
# (C) stats（サブセット。全量は --manifest manifest.jsonl --frac 0.1）
uv run python scripts/prepare_moespeech.py stats \
  --manifest data/moespeech/manifest_subset.jsonl --out data/moespeech/stats_subset.json --num-workers 16
# (D) precompute（サブセット）
uv run python scripts/prepare_moespeech.py precompute \
  --manifest data/moespeech/manifest_subset.jsonl --out-dir /dev/shm/moespeech_precomputed_subset \
  --mel-mean <stats_subset.mel_mean> --mel-std <stats_subset.mel_std> --num-workers 16
```

**検証**（ローカル）: .pt schema `{mel,text,spk,cleaned_text}`（durations 無し）を JVS MAS .pt と keys 一致 assert / text 長 = 2L+1 / `PrecomputedTextMelDataset(n_spks=473, load_durations=False)` → collate 1 バッチ成功 / speakers.json 473 エントリ・値 0..472 連続 / precompute 後 .pt 全ユニーク・件数一致。

---

### Phase 2 — サブセット smoke + MAS 退化率ゲート（インスタンス・安価）

**新規 file: `configs/data/moespeech_precomputed.yaml`**
```yaml
_target_: matcha.data.precomputed_datamodule.PrecomputedTextMelDataModule
name: moespeech_precomputed
train_pt_dir: /dev/shm/moespeech_precomputed_subset/train   # 全量時に CLI 上書き
val_pt_dir: /dev/shm/moespeech_precomputed_subset/val
batch_size: 32
num_workers: 8
pin_memory: True
n_spks: 473                 # サブセットでも 473 固定（id 空間保存）
n_feats: 80
# NOTE: これらは fmax=11025 の生 log-mel 統計。JVS(8000, -6.55/2.38) を絶対に流用しない
data_statistics:
  mel_mean: <stats.json の値を貼付>   # placeholder 0.0/1.0 のまま学習禁止
  mel_std:  <stats.json の値を貼付>
seed: ${seed}
load_durations: false      # MAS 経路（durations 無し）
num_buckets: 20
```

**新規 file: `configs/experiment/moespeech_pretrain.yaml`**
```yaml
# @package _global_
defaults:
  - override /data: moespeech_precomputed.yaml
  - override /trainer: ddp_optimized.yaml
  - override /callbacks: default.yaml
tags: ["moespeech","japanese","multispeaker","pretrain","mas","fmax11025"]
run_name: moespeech_pretrain
model:
  n_vocab: 55
  optimizer:
    fused: false             # bf16-mixed + grad clip 必須
compile_model: false
compile_regional_blocks: false
gradient_checkpointing: false
callbacks:
  early_stopping: {_target_: lightning.pytorch.callbacks.EarlyStopping,
    monitor: "loss/val", patience: 30, min_delta: 0.001, mode: "min", check_finite: true, verbose: true}
  ema: {_target_: lightning.pytorch.callbacks.EMAWeightAveraging,
    decay: 0.9995, update_every_n_steps: 1, update_starting_at_epoch: 10}
trainer:
  max_epochs: 200            # サブセット検証値。全量は §4 で総step予算から逆算し CLI 上書き
  check_val_every_n_epoch: 10
  precision: bf16-mixed      # T4/V100 は CLI で 32-true
```

**サブセット学習コマンド**（`test=false` 必須, D9）
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python matcha/train.py \
  experiment=moespeech_pretrain compile_model=false \
  data.batch_size=32 data.num_workers=0 +data.preload_to_memory=true test=false
```

**MAS 退化率ゲート（D6・本フェーズの核心）**
- 数 epoch ごとに `scripts/eval_degeneration_rate.py`（`matcha/utils/alignment_metrics.is_degenerate`: 音素位置の 80% 以上が ≤1 frame で退化判定）で退化率を定点監視。
- **判断基準**:
  - 退化率が JVS 100話者の 39–43% を**著しく超えない**（例 <55%）→ **MAS 継続**（緩和策: blank zero-init `text_encoder.py:411`、DP FiLM `text_encoder.py:100-105` が既存で効く）。事前学習は表現獲得が目的で、DP 退化は単一話者 fine-tune でクリーンな MAS ターゲットに再学習される。
  - 閾値超（例 ≥55%）→ **Julius フォールバック**: `precompute_with_alignment.py --fmax 11025` で MoeSpeech に外部 duration を焼込み、`load_durations=true` の aligned config に切替（`jvs_aligned` を雛形）。ただし ASR 転記誤りで Julius が不安定化するトレードオフを承知の上で。
- **リスク受容の根拠を計画に明記**: ユーザ確定は「両段階 MAS」。CLAUDE.md の実証退化と正面衝突するため、この計測ゲートを通過条件とする。

**その他検証**: loss/train・sub_loss(dur/prior/diff) 単調減少・NaN 無し / TensorBoard alignment 画像が対角 / 異なる spk id で音色変化（FiLM/spk_emb 生存確認）。

---

### Phase 3 — 全量 pretrain（stats/precompute はローカル $0、学習はインスタンス最高額）

- ローカル: `select` を通さず `manifest.jsonl` 全量に対し `stats`（`--frac 0.1`）→ `precompute`（別 out-dir）。
- config 差分（§4 参照）: `data.train_pt_dir/val_pt_dir` を全量 .pt に、`data.data_statistics.mel_mean/mel_std` を**全量 stats で再取得した値**に、`trainer.max_epochs` を総 step 予算から逆算、`preload_to_memory=false`（数十万 .pt は RAM OOM）+ `num_workers=8` + `/dev/shm`。
- 出力 = **base ckpt**（`logs/train/moespeech_pretrain/runs/<run>/checkpoints/last.ckpt`、n_spks=473, fmax=11025, mel_mean/std 焼込み済み）。EMA callback により state_dict = EMA 重み、current_model_state = raw 重み。

---

### Phase 4 — 話者 embedding 遷移（ローカル $0、確定手順は §5）

`scripts/transfer_speaker_embedding.py` で base ckpt（n_spks=473）→ つくよみ init（n_spks=474、weights-only）を生成。詳細は §5。

---

### Phase 5 — つくよみちゃん fine-tune（precompute はローカル $0、学習は単一 GPU）

**新規 file: `scripts/prepare_tsukuyomi.py`**
- `ayousanz/tsukuyomi-chan-ljspeech` の `metadata.csv`（LJSpeech 形式）から各行 `parts[0]=name / parts[-1]=text`（列レイアウトに堅牢化）で `{wavs_dir}/{name}.wav|<slot_id>|{text}` を生成。
- `--slot-id 473`（= base n_spks、つくよみの新 slot）。`--val-ratio 0.1` で train/val split。つくよみは既に 22050Hz なのでリサンプル不要。任意 `--trim`（`prepare_jvs.trim_silence` 相当、top_db=30。要試聴判断）。

**precompute（Moe 統計を厳密転記、`--fmax 11025`）**
```bash
uv run python scripts/prepare_tsukuyomi.py --meta <tsukuyomi>/metadata.csv \
  --wavs-dir <tsukuyomi>/wavs --out-dir data/tsukuyomi --slot-id 473 --val-ratio 0.1
uv run python scripts/precompute_dataset.py --filelist data/tsukuyomi/train.txt \
  --output-dir data/tsukuyomi_precomputed/train \
  --mel-mean <BASE_MOE_MEAN> --mel-std <BASE_MOE_STD> --fmax 11025 --num-workers 4
# val も同一 mel-mean/std/fmax で
```
**つくよみは単一話者なので .pt 命名衝突は起きない**（precompute_dataset.py の parent.name 命名で可、Phase 0 の `--fmax` のみ使用）。

**新規 file: `configs/data/tsukuyomi_precomputed.yaml`**（**D10 修正: num_buckets=1**）
```yaml
_target_: matcha.data.precomputed_datamodule.PrecomputedTextMelDataModule
name: tsukuyomi_precomputed
train_pt_dir: /dev/shm/tsukuyomi_precomputed/train
val_pt_dir: /dev/shm/tsukuyomi_precomputed/val
batch_size: 16
num_workers: 2
pin_memory: True
n_spks: 474                 # base 473 + つくよみ新 slot。transfer 出力に一致
n_feats: 80
data_statistics:
  mel_mean: <BASE_MOE_MEAN>  # ★ base MoeSpeech(fmax=11025) と同一値。strict load で
  mel_std:  <BASE_MOE_STD>   #   model buffer がこの値に上書き=precompute正規化と一致必須
seed: ${seed}
load_durations: false        # MAS（単一話者=退化なし=原論文設定）
num_buckets: 1               # ★ ~90発話×bs16: num_buckets>=2 だと bucket断片化で全batch drop
```
> **num_buckets 根拠**（実コード確認）: `bucket_size=max(1, n//num_buckets)`, drop_last は bucket 内 partial batch を drop。n=90, num_buckets=20 → bucket_size=4 → 全 bucket(4) < batch(16) → 全 drop（0 batch、`__len__` は 5 と嘘をつく）。`num_buckets=1` → 単一 bucket 90 → `floor(90/16)=5` batch/epoch。~90 サンプルで length-bucketing の利得は無視できる。

**新規 file: `configs/experiment/tsukuyomi_finetune.yaml`**
```yaml
# @package _global_
# 実行: uv run python matcha/train.py experiment=tsukuyomi_finetune test=false \
#         ckpt_path=/abs/checkpoints/tsukuyomi_init.ckpt
defaults:
  - override /data: tsukuyomi_precomputed.yaml
  - override /trainer: gpu.yaml       # 単一GPU（DDP は総数<128 で drop_last により全batch消失）
  - override /callbacks: default.yaml
tags: ["tsukuyomi","japanese","single-speaker","finetune","mas"]
run_name: tsukuyomi_finetune
model:
  n_vocab: 55
  optimizer:
    fused: false
    lr: 1e-4          # 過学習/忘却が見えたら 5e-5〜2e-5 へ
compile_model: false
compile_regional_blocks: false
gradient_checkpointing: false
callbacks:
  model_checkpoint:
    monitor: "loss/val"
    mode: "min"
    save_top_k: 3
    save_last: true
    save_on_train_epoch_end: false    # ★ D10: val 終了時に best 判定（epoch cadence 変更に頑健）
    auto_insert_metric_name: false
    filename: "ft_{epoch:03d}"
    every_n_epochs: 25
  early_stopping:
    _target_: lightning.pytorch.callbacks.EarlyStopping
    monitor: "loss/val"
    patience: 8
    min_delta: 0.0
    mode: "min"
    check_finite: true
    verbose: true
  ema:
    _target_: lightning.pytorch.callbacks.EMAWeightAveraging
    decay: 0.9995
    update_every_n_steps: 1
    update_starting_at_epoch: 0        # fine-tune は総step少・早期蓄積
trainer:
  # steps/epoch = floor(train_utt/batch_size) 例 floor(81/16)=5
  # max_epochs  = 目標steps / steps/epoch      例 7500/5 = 1500
  max_epochs: 1500
  check_val_every_n_epoch: 25
  gradient_clip_val: 1.0                # 実証レシピ（default.yaml は 5.0）
  precision: bf16-mixed
```

**fine-tune 実行**
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python matcha/train.py \
  experiment=tsukuyomi_finetune compile_model=false test=false \
  ckpt_path=/abs/checkpoints/tsukuyomi_init.ckpt
```
推論時は `spks=473`（つくよみ slot）を指定。

**検証**: config 合成で n_spks=474 / n_vocab=55 / load_durations=false / out_size=null / devices=1 / precision=bf16-mixed / fused=false / ckpt_path 反映 / **train_dataloader が >=1 batch を返す（num_buckets=1 修正の確認）** / 数十 step で dur/prior/diff loss 有限・減少 / best-val ckpt で train/val 行を synthesise 比較。

---

### Phase W — WaveNeXt fmax=11025 再学習（Phase 0 後は並行可）

詳細スコープは §6。学習コードは新規 `wavenext_train/` に隔離（推論用 `matcha/wavenext/` の inference-only 境界を保つ）。

---

### Phase E — 評価 + ONNX

- **GT-mel 再合成**（音響非依存で vocoder 単体評価）: 本物録音 wav → `MatchaMelFeatures(fmax=11025)` → 学習済み WaveNeXt → UTMOS/mel-L1/試聴。fmax=8000 BSC ゼロショット（survey: 2.925）比で高域再現・濁り改善を確認。
- **end-to-end A/B**: fmax=11025 音響モデル予測 mel → 再学習 WaveNeXt vs 旧 fmax=8000 経路。8–11kHz エネルギー・濁り天井の改善を一次評価。
- **ONNX**: `matcha/onnx/export.py --vocoder-name wavenext --vocoder-checkpoint-path wavenext_ja_11025.bin --n-timesteps 5`（経路変更ゼロ）→ onnx.checker PASS・iSTFT-free 維持・ONNX-CPU RTF 計測。
- **評価ガード**: 既定 HiFi-GAN（`hifigan/config.py:24` fmax=8000）と旧 WaveNeXt（fmax=8000）は 11025 mel と非互換 → fmax=11025 vocoder が無い状態での合成をブロック/警告。

---

## 4. サブセット検証 → 全量の切替点

**目的**: 配管バグ（.pt schema / datamodule 互換 / MAS 経路 / 学習ループ / fmax 一貫性）を**安価なサブセットで潰してから**高額な全量 pretrain に進む。

| 項目 | サブセット | 全量 | 変更方法 |
|---|---|---|---|
| `data.train_pt_dir/val_pt_dir` | `/dev/shm/moespeech_precomputed_subset/*` | `/dev/shm/moespeech_precomputed/*` | CLI 上書き or config 編集 |
| `data.data_statistics.mel_mean/std` | サブセット stats | **全量 stats で再取得**（分布が違う） | config 編集（placeholder ガードで防御） |
| `data.n_spks` | 473（固定） | 473（固定） | **変更なし**（id 空間保存が切替の前提） |
| `model.n_spks` | `${data.n_spks}` 自動追従 | 同左 | **手動変更不要** |
| `+data.preload_to_memory` | true（数万で可） | **false**（数十万は RAM OOM） | CLI |
| `data.num_workers` | 0（preload 時） | >0（8 程度、/dev/shm 前提） | CLI |
| `trainer.max_epochs` | 200（配管確認） | **総 step 予算から逆算**（D5） | CLI 上書き |

**max_epochs の逆算式（D5、丸写し禁止）**:
```
steps/epoch = ceil(N_utt / (batch_size × n_gpu))
max_epochs  = 目標total_steps / steps/epoch      （目標 total_steps = 150K–300K）
```
例: N_utt=200,000, batch=32, n_gpu=4 → steps/epoch≈1563 → 目標 250K step なら max_epochs≈160。**JVS の 2500 は使わない**。

**切替の判断基準（全量へ進む条件）**:
1. サブセット smoke で loss が NaN 無く単調減少、alignment が対角。
2. **MAS 退化率がゲート閾値内**（§Phase 2。超なら Julius 切替を全量にも適用）。
3. 異なる spk id で音色が変わる（話者条件付け生存）。
4. fmax 一貫性検証（§2.3）全通過、placeholder ガード動作。

---

## 5. 話者 embedding 遷移の確定手順（多話者 → 単一、checkpoint load 整合）

**確定方針（D1/D2）**: n_spks=1 化はアーキ手術（spk_emb/FiLM/proj_m/proj_w/encoder先頭/decoder先頭の shape 変化）を招き strict load 不能。**多話者アーキを保持し `spk_emb.weight` を 473→474 にリサイズ**する（新行 = つくよみ slot 473）。他テンソルは byte-identical。これが verifier「VERIFIED CONSISTENT」の唯一安全経路。

**新規 file: `scripts/transfer_speaker_embedding.py`**
```python
"""MoeSpeech 多話者 ckpt -> つくよみ単一話者 init（話者embedding遷移, weights-only）"""
import argparse
from pathlib import Path
import torch

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--init-from-id", type=int, default=None,
                   help="新slotを既存話者idの埋め込みで初期化（未指定なら全話者平均）")
    p.add_argument("--use-current-model-state", action="store_true",
                   help="EMAでなくraw重み(current_model_state)を遷移元にする")
    args = p.parse_args(argv)

    ckpt = torch.load(args.source, map_location="cpu", weights_only=False)
    key = "current_model_state" if args.use_current_model_state else "state_dict"  # 既定=EMA重み
    if key not in ckpt:
        raise KeyError(f"'{key}' not in ckpt; keys={list(ckpt)[:20]}")
    sd = dict(ckpt[key])

    emb_key = "spk_emb.weight"
    if emb_key not in sd:
        raise KeyError(f"'{emb_key}' 無し=baseが単一話者(n_spks<=1)。多話者baseが必要。")
    old = sd[emb_key]                 # (n_spks_old, 64)
    n_old, dim = old.shape
    # ★ torch.empty は使わない（CLAUDE.md 禁止）。torch.cat で新行を連結
    if args.init_from_id is not None:
        assert 0 <= args.init_from_id < n_old
        new_row = old[args.init_from_id:args.init_from_id+1].clone()
        how = f"copied from id {args.init_from_id}"
    else:
        new_row = old.mean(dim=0, keepdim=True)
        how = "mean of existing rows"
    sd[emb_key] = torch.cat([old, new_row], dim=0)   # (n_old+1, 64)
    print(f"[+] spk_emb {tuple(old.shape)} -> {tuple(sd[emb_key].shape)}; slot id={n_old} ({how})")

    # ★ D4 ガード: mel_mean/std を print し placeholder でないことを確認させる
    for stat in ("mel_mean", "mel_std"):
        if stat in sd:
            v = float(sd[stat])
            print(f"[i] base {stat} = {v:.6f}  -> precompute --{stat.replace('_','-')} と config data_statistics に転記")
            assert not (stat=="mel_std" and abs(v-1.0)<1e-9), "mel_std==1.0=placeholder。fmax=11025統計未算出のbaseで遷移するな"

    hp = ckpt.get("hyper_parameters", {})
    if "n_spks" in hp: hp["n_spks"] = n_old + 1

    out = {"state_dict": sd}          # ★ optimizer_states/averaging_state/current_model_state を捨てる
    if hp: out["hyper_parameters"] = hp
    Path(args.target).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.target)
    print(f"[+] wrote {args.target}: n_spks={n_old+1}, filelist spk列={n_old}")

if __name__ == "__main__":
    main()
```

**実行**
```bash
uv run python scripts/transfer_speaker_embedding.py \
  --source logs/train/moespeech_pretrain/runs/<run>/checkpoints/last.ckpt \
  --target checkpoints/tsukuyomi_init.ckpt
# 近い女性話者idが判れば --init-from-id <id>（平均より収束速い）
```

**checkpoint load 整合の連鎖（実コードで確認済み）**:
1. 遷移出力は `{state_dict[, hyper_parameters]}` のみ = `optimizer_states` 無し → train.py L155 判定で **weights-only 分岐 L158**（fresh optimizer/epoch0/EMA）に入る。
2. L163 で `time_embeddings.emb_weights`（非 persistent 残骸）を除去 → L165 `model.load_state_dict(state_dict)`（strict）。
3. n_spks=474 で hydra instantiate（`data.n_spks=474` → `model.n_spks` 自動追従）すれば `spk_emb.weight`[474,64] が一致し **missing/unexpected keys ゼロ**。RoPE/SinusoidalPosEmb は非 persistent で strict 対象外。
4. EMA を遷移元にしたので EMA 品質を継承。`mel_mean/std` は persistent buffer で strict load により Moe 値が model に載る → denormalize が Moe 値を使う → **Tsukuyomi precompute も Moe 値で正規化必須**（§2.1 のチェーン）。

**遷移検証（ローカル $0）**:
- `torch.load(target)` の keys が `{state_dict[, hyper_parameters]}` のみ（optimizer_states 等が無い）。
- `spk_emb.weight.shape==(474,64)`、`new[:473]` が base と diff==0、`new[473]` が平均（or 指定 id 行）と一致。
- spk_emb 以外の全テンソルが base と byte-identical。
- n_spks=474/n_vocab=55/spk_emb_dim=64 で instantiate → `load_state_dict` が missing/unexpected 無しで成功。

---

## 6. WaveNeXt fmax=11025 再学習の現実的スコープ

**現状**: `matcha/wavenext/` は推論専用（学習ループ無し）。移植元 `wetdog/wavenext_pytorch`（vocos フォーク、GAN ボコーダ）から学習部品を移植し、本リポの mel(fmax=11025) 生成に適合させる。**新規 `wavenext_train/` に隔離**（非破壊）。

**最重要の設計判断（実確認済み）**:
1. ボコーダ入力は **z-score 正規化 mel ではなく denormalize 済み生 log-mel**（`matcha_tts.py:167` synthesise → `cli.py:148` to_waveform が直渡し）。よって学習 feature_extractor は `matcha/utils/audio.py::mel_spectrogram` を **fmax=11025 で呼び正規化しない**。mel_mean/std はボコーダに不要。
2. dataset は波形のみ返す（filelist=wav パス1行/行、22050 resample、num_samples=16384=256×64 ランダム crop）。center=False+reflect pad 384 のため head 出力長=入力長（長さ調整不要、num_samples は 256 の倍数）。
3. **初期値**: BSC の fmax=8000 重み（`backbone.*`/`head.*`）を**全流用可**（ネットワークに fmax 依存パラメータ皆無、input_channels=80 は不変）。discriminator は HF 未公開=scratch。→ ランダムより圧倒的に速い。
4. **2 段学習**: ① MoeSpeech（多話者・大量、BSC 初期化）でロバストなベース → ② つくよみ fine-tune。単一話者 100 発話のみでは scratch には過少。

**新規 file（~7-8 本）**:
- `wavenext_train/features.py` — `MatchaMelFeatures`（`mel_spectrogram(..., fmax=11025, center=False)`、正規化なし）。学習・推論の mel 領域を厳密一致させる中核。
- `wavenext_train/dataset.py` — 波形 dataset。torchaudio sox 依存を pure-torch peak/RMS 正規化に置換（torchaudio≥2.9 で sox backend 削除）。
- `wavenext_train/discriminators.py` — MPD(periods=2,3,5,7,11) + MRD(fft=2048,1024,512)。**wetdog raw を逐語移植**。
- `wavenext_train/loss.py` — hinge GAN + L1 mel(MatchaMelFeatures 再利用) + FM。
- `wavenext_train/experiment.py` — Lightning 2.x **manual optimization**（PL1.8 optimizer_idx 廃止対応）。backbone=VocosBackbone/head=WaveNextHead を `matcha/wavenext/models.py` から再利用。
- `wavenext_train/train.py`、`configs_wavenext/wavenext_11025.yaml`。
- `scripts/init_wavenext_from_bsc.py`（BSC bin → backbone/head 注入）、`scripts/extract_wavenext_generator.py`（学習後 → backbone/head のみの bin）、`scripts/make_wavenext_filelist.py`。

**verifier corrections 反映（D11、実装ゲート）**:
1. **manual optimization の global_step 会計**: disc 学習後は opt_d.step()+opt_g.step() で 2 step/batch 加算 → `pretrain_mel_steps` ゲートとスケジューラ総長がズレる。**専用 `register_buffer("n_batches")` カウンタ**でゲートと scheduler step を統一。
2. **D-step は `y_hat.detach()`** を使う（generator 2 回目 forward を廃す）。G-step の disc 呼び出しは adversarial/FM 専用と割り切る。
3. **wetdog raw 逐語検証を必須ゲート化**: `discriminators.py`/`loss.py`/`experiment.py` を 1 ファイルずつ raw 取得 → tiny batch で全 forward/backward の shape・loss タプル形状を assert してからマージ（推測コードのままマージ禁止）。
4. **pretrain_mel_steps を 1 表に統一**: MoeSpeech ベース（BSC gen init + scratch disc）= N（disc calibration 用）、つくよみ fine-tune = 0。config/§6/コマンドで同値。
5. **transformers 依存を排除**: `get_cosine_schedule_with_warmup` を `torch.optim.lr_scheduler.LambdaLR` の cosine+warmup 自前実装に置換（本体依存 lightning/torch/einops/torchaudio/scipy/numpy のみで完結）。
6. **stft を FP32 固定**: features.py/loss.py の mel 計算を `torch.autocast(device_type="cuda", enabled=False)` でラップ（bf16-mixed 下の数値安定）。
7. **EMA 記述の整合**: extract の「EMA 優先」コメントは、EMA callback を experiment.py に追加する（jvs と同じ decay=0.9995）か、コメント削除でどちらかに統一。
8. **validation §1 修正**: `mel_spectrogram(audio,1024,80,22050,256,1024,0.0,11025,center=False)` を直接呼んで `MatchaMelFeatures(fmax=11025)` と atol=0 比較（precompute 経路と比べるなら Phase 0 の `--fmax` 前提）。

**非破壊性**: `matcha/wavenext/vocoder.py` は docstring のみ更新（コード不変、param 13.6-13.8M・`backbone.`/`head.` prefix 不変で `test_wavenext.py` 緑のまま）。抽出 bin は `load_wavenext` を round-trip 通過（`feature_extractor.*` drop, strict=False, missing/unexpected ゼロ）。ONNX 経路は `--vocoder-checkpoint-path` 直渡しで変更ゼロ。**注意**: 学習用 `.ckpt` を直接 load_wavenext に渡すと `mpd./mrd.` が unexpected key で assert 失敗 → 必ず `extract_wavenext_generator.py` を経由。`VOCODER_URLS` にローカルパスを足すと DL を試みて失敗するので追加しない（path 直渡しで足りる）。feature_extractor n_mels は 80 維持（BSC init と input_channels=80 が壊れる。recon loss のみ高解像可）。

**工数**: 実装 ~1-2 人日（逐語移植 + shape 検証 + manual optimization 化）+ 補助 script/config ~0.5 人日。学習は BSC 初期化で scratch 比大幅短縮。

---

## 7. コスト / 時間の総括

### 7.1 ローカル $0（GPU 課金なし・CPU のみ）

| 作業 | 見積 |
|---|---|
| Phase 0 コード編集（precompute_dataset/with_alignment --fmax） | ~0.5h |
| `prepare_moespeech.py`（4 サブコマンド, ~350行）+ config 2 本 | 半日〜1日 |
| `transfer_speaker_embedding.py` + `prepare_tsukuyomi.py` + config 2 本 | ~1-1.5h |
| MoeSpeech prepare（DL+trim+resample、帯域律速） | 数十分〜数時間（総容量数十GB 想定、回線次第） |
| stats（全量 `--frac 0.1`） | 数分〜十数分 |
| precompute（**共有テキストキャッシュで ~168 samples/sec**。無ければ ~8/sec） | サブセット数分 / 全量数十万は数十分〜（キャッシュ必須） |
| つくよみ precompute（~100発話） | <1 分 |
| 全検証（byte-identity / 統計サニティ / strict-load / 合成） | 数十分 |

### 7.2 インスタンス課金（RTX 5090 vast.ai、~$0.5-1/GPU時 概算）

| 学習 | 見積 |
|---|---|
| **Phase 2** サブセット smoke（配管+退化率ゲート、1×or4×5090） | ~1-3h（安価に問題潰し） |
| **Phase 3** 全量 pretrain（4×5090 DDP, bf16, 目標 150-300K step） | 最高額。walltime は発話長分布/IO 律速で要実測。段階実行でバグを安く潰す |
| **Phase 4** 遷移 | ローカルで可（数秒、$0） |
| **Phase 5** つくよみ fine-tune（**単一** 5090, 6-8K step） | ~数十分-1h。lr/step sweep(2条件) 込みで +1h |
| **Phase W** WaveNeXt（① Moe ~300K step ② つくよみ ~50K step, 1×5090） | ①~8-16h ②~2-4h。**per-step は Matcha より重い**（MPD5+MRD3+mel-stft）ので 8-16h は下振れ、tiny run で it/s 実測後に予算確定 |
| **Phase E** 評価/ONNX | ~1-2 GPU時 |

**総括**: fmax 一貫性（precompute/統計/config）と話者遷移は**全てローカル $0 で確定・検証**でき、課金は「pretrain 学習 / WaveNeXt 学習 / 評価」に限定。最小構成（Moe 全量 pretrain 単発 + つくよみ fine-tune + WaveNeXt ①②単発）で概ね WaveNeXt ~$10-20 + Moe pretrain（要実測、最大項目）。**WaveNeXt は Phase 1-5 と並行でき、GT 波形で音響モデル完成前に着手・評価可能**。

---

## 8. 既存 JVS 資産を壊さない保証（非破壊）

| 保証 | 根拠（実コード確認） |
|---|---|
| `precompute_dataset.py --fmax` default=8000 | F_MAX 参照は L37/L98/L248 の 3 箇所のみ、default 維持で JVS 再生成は byte-identical。`process_sample` の `fmax` は**末尾 default 引数**で `tests/test_precompute_dataset.py` の 7 引数 positional 呼び出しと後方互換 |
| `audio.py` 無改変 | キャッシュキーに fmax 込み（L54）で 8000/11025 が別 basis、stale 化なし |
| 新 config/script は全て新規 | `jvs_*` config/experiment、`jvs_precomputed*.yaml`、`matcha.yaml`、`precomputed_datamodule.py`、`matcha_tts.py` を無改変 |
| `.pt` schema 一致 | Moe/つくよみ .pt = `{mel,text,spk,cleaned_text}`（durations 無し）は `PrecomputedTextMelDataset`(load_durations=False) と厳密一致 |
| fmax=11025 資産は物理分離 | `/dev/shm/moespeech_*` `/dev/shm/tsukuyomi_*` は JVS(8000) と別ディレクトリ |
| WaveNeXt | `vocoder.py` は docstring のみ更新（コード不変、test_wavenext.py 緑） |
| Julius 経路 | `precompute_with_alignment.py --fmax` default=8000 で既存 aligned 経路 byte-identical |

**唯一の破壊要因**: `--fmax` default を変える / precompute 命名ロジックを JVS 非互換に変える。→ **どちらもしない**（Moe は専用スクリプトで spk-id 前置命名、precompute_dataset.py の parent.name 命名は JVS 後方互換で温存）。

---

## 9. 最大リスクと対策

| リスク | 深刻度 | 対策（本計画での確定処置） |
|---|---|---|
| **fmax 不整合（無言破綻）** | 最高 | §2 の一貫チェーン + §2.3 の検出 5 手法。**placeholder(0.0/1.0) 起動ガード**。fmax は precompute で完全確定。旧 JVS 統計での 11025 正規化を退行検知に使う。**end-to-end で音響=vocoder の fmax 一致が必須**（片方 8000 だと濁り再発） |
| **MAS 多話者退化（473話者）** | 高 | §Phase 2 の**退化率実測ゲート**。閾値内なら MAS 継続（既存緩和策 blank zero-init + DP FiLM）+ 単一話者 fine-tune で DP 回復。閾値超なら **Julius(`--fmax 11025`) フォールバック**。リスク受容根拠と計測ゲートを計画に明記。「事前学習は表現獲得目的、最終品質は単一話者 fine-tune で確定」 |
| **ASR 転記の g2p 不適合** | 中 | §Phase 1 の**三重防御**（NFKC 正規化 + 注釈除去 + 日本語含有チェック + 長さフィルタ + g2p dry-run）。ドロップ率をログ、閾値超で `_ANNOT` 強化。KeyError は try/except で skip（件数記録、無言喪失を可視化） |
| **話者遷移 strict load 失敗** | 高 | §5 の確定手順。**n_spks=1 化しない**（473→474 resize のみ）。weights-only ckpt で train.py L158 分岐。遷移検証（keys/shape/byte-identity/load_state_dict）をローカルで通過 |
| **.pt 命名衝突（データ喪失）** | 高 | §Phase 1。Moe は spk-id 前置 `{spk}_{stem}.pt` + per-speaker wav dir + stem グローバル一意 assert（fail-fast）。つくよみは単一話者で衝突なし |
| **max_epochs 丸写しで課金爆発** | 中 | §4 の**総 step 予算逆算**。JVS 2500 を使わない。段階実行（サブセット→全量）で安くバグ潰し |
| **tsukuyomi 空 dataloader** | 高（config バグ） | §Phase 5。**num_buckets=1**（実コードで断片化バグ確認済み）+ `save_on_train_epoch_end=false` |
| **WaveNeXt 学習ループの推測実装** | 中 | §6。**wetdog raw 逐語検証ゲート**（shape assert 前提）+ manual optimization batch カウンタ + `y_hat.detach()` + LambdaLR（transformers 排除）+ stft autocast off。GT 波形で先行学習・並行実行 |
| **既定 HiFi-GAN/旧 WaveNeXt(8000) 誤用** | 中 | §Phase E の**評価ガード**（fmax=11025 vocoder が無い状態での合成をブロック/警告） |
| **preload_to_memory RAM OOM** | 中 | 全量は `preload_to_memory=false` + `num_workers>0` + `/dev/shm`。サブセットのみ preload 可 |

---

## 10. 大規模run向けプロファイリング（オプトイン・強制しない）

ユーザ方針: **無理に最適化はしない**。学習高速化のコード対応は既に完了（`docs/training-speedup-implementation-plan.md`: bf16-mixed恒久化・A-1実行済み・A-2見送り・C-1/C-2 scaffolding）。本パイプラインへの適用状況と、大規模化での唯一の追加確認点を記す。

### 適用済みの高速化（自動で効く）
- **bf16-mixed + fused=false**: `moespeech_pretrain.yaml` は `jvs_aligned`/`jvs_fast` 系を雛形にするため自動継承
- **precompute（.pt化）**: 本計画の Phase 1（mel+g2p事前計算→.pt）そのものが最大の高速化。学習をI/O律速化。共有テキストキャッシュで ~168 samples/sec（§7.1）
- **A-1結論の流用**: モデルアーキ同一のため律速（カーネル起動律速・decoder conv経路）も同一 → **A-2 Regional compileは同様に効かない**（本計画でも `compile_regional_blocks=false`）

### 追加確認（オプトイン・全量pretrain前に1回だけ、~$0.5）
大規模MoeSpeech（数十万発話・発話長ばらつき大）でのみ、律速が変わる可能性がある1点:
- **A-1プロファイリング再実行**: `bash scripts/profile_training.sh`（`experiment` を `moespeech_pretrain` に向ける）でサブセット学習の律速を1回測る
- **判定**:
  - カーネル起動律速のまま（JVSと同じ、想定通り）→ **何もしない**。bf16+precomputeで完了
  - **データ充填律速**（step境界のGPUアイドル・`aten::copy_`優勢）が新たに確認された場合**のみ** → B-1 frame batching（`matcha/data/precomputed_datamodule.py` の frame予算サンプラ）を検討
- **デフォルトは無最適化**。B-1は「実測で充填律速が出た時だけ」の保険であり、強制的な追加実装はしない
- 位置づけ: Phase 2（サブセットsmoke）に相乗りできる安価な確認。Phase 3（全量pretrain）の前に律速を1回確認しておくと、大規模runでの投資判断が根拠を持つ

### 実装チェックリスト（着手順）

- [ ] **Phase 0**: `precompute_dataset.py --fmax` / `precompute_with_alignment.py --fmax` → JVS byte-identity 検証
- [ ] **Phase 1**: `prepare_moespeech.py`（prepare/select/stats/precompute、D3/D7/D8/perf 反映）+ `moespeech_precomputed.yaml` + `moespeech_pretrain.yaml`
- [ ] **Phase 1 検証**: .pt schema / speakers.json / 統計サニティ / placeholder ガード
- [ ] **Phase 2**: サブセット smoke（`test=false`）+ **MAS 退化率ゲート** → MAS継続/Julius切替の判断
- [ ] **Phase 3**: 全量 stats/precompute（ローカル）→ max_epochs 逆算 → 全量 pretrain → base ckpt
- [ ] **Phase 4**: `transfer_speaker_embedding.py`（473→474, weights-only）→ strict-load 検証
- [ ] **Phase 5**: `prepare_tsukuyomi.py` + precompute（Moe 統計）+ `tsukuyomi_precomputed.yaml`(**num_buckets=1**) + `tsukuyomi_finetune.yaml` → 単一GPU fine-tune
- [ ] **Phase W**（並行）: `wavenext_train/` 移植（逐語検証ゲート）+ init/extract/filelist → Moe ベース → つくよみ fine-tune
- [ ] **Phase E**: GT-mel 再合成 / end-to-end A/B / ONNX（評価ガード）