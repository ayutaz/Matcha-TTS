# Matcha-TTS 日本語音声合成 実装計画書

## 目次

1. [概要](#1-概要)
2. [現在のアーキテクチャ分析](#2-現在のアーキテクチャ分析)
3. [日本語音素セット設計](#3-日本語音素セット設計)
4. [G2Pライブラリ選定](#4-g2pライブラリ選定)
5. [日本語データセット](#5-日本語データセット)
6. [実装対象ファイルと変更内容](#6-実装対象ファイルと変更内容)
7. [HiFi-GANボコーダの互換性](#7-hifi-ganボコーダの互換性)
8. [転移学習戦略](#8-転移学習戦略)
9. [学習・推論フロー](#9-学習推論フロー)
10. [実装ステップ](#10-実装ステップ)
11. [既存の日本語Matcha-TTS実装](#11-既存の日本語matcha-tts実装)
12. [参考文献](#12-参考文献)

---

## 実装状況

| Phase | 内容 | 状態 |
|-------|------|------|
| Phase 1 | テキスト処理パイプライン | ✅ 完了 |
| Phase 2 | データ準備 | ✅ 完了（スクリプト・設定作成済み、実データ処理は手動） |
| Phase 3 | 学習設定 | ✅ 完了 |
| Phase 4 | 学習実行 | ⏳ 未着手（手動実行ステップ） |
| Phase 5 | 推論・UI対応 | ✅ 完了 |

**変更ファイル一覧:**

| ファイル | 変更種別 | 内容 |
|---------|---------|------|
| `pyproject.toml` | 編集 | `japanese = ["pyopenjtalk-plus"]` optional dependency追加 |
| `matcha/text/symbols.py` | 編集 | `symbols_ja` (52シンボル) 追加 |
| `matcha/text/cleaners.py` | 編集 | `japanese_cleaners` + `_fullcontext_to_prosody` 追加 |
| `matcha/text/__init__.py` | 編集 | `language` パラメータ追加、スペース区切りトークン対応 |
| `matcha/data/text_mel_datamodule.py` | 編集 | `language` パラメータ伝播 |
| `matcha/cli.py` | 編集 | `--language`/`--cleaners` 引数、チェックポイント言語自動検出 |
| `matcha/app.py` | 編集 | 日本語モデル選択UI、JSUTモデルガード |
| `matcha/onnx/export.py` | 編集 | `get_inputs` の語彙範囲をモデル `n_vocab` から動的取得 |
| `matcha/onnx/infer.py` | 編集 | `--language`/`--cleaners` 引数追加 |
| `configs/data/jsut.yaml` | 新規 | JSUTデータ設定 |
| `configs/experiment/jsut.yaml` | 新規 | JSUT実験設定 (`n_vocab: 52`) |
| `scripts/prepare_jsut.py` | 新規 | JSUTデータ準備スクリプト (リサンプリング+ファイルリスト生成) |
| `scripts/transfer_from_english.py` | 新規 | 転移学習チェックポイント作成スクリプト |
| `tests/test_text_ja.py` | 新規 | 日本語テキスト処理テスト (17件) |
| `tests/test_onnx_ja.py` | 新規 | ONNX語彙範囲テスト (7件) |

---

## 1. 概要

Matcha-TTSに日本語テキスト→音声合成機能を追加する。現在のパイプラインは英語専用（espeak-ngベースのIPA音素化、178シンボル語彙）だが、モデルのコアアーキテクチャ（TextEncoder/CFM Decoder/HiFi-GAN）は言語非依存であり、テキスト処理パイプラインの拡張のみで日本語対応が可能。

### パイプライン概要（変更前→変更後）

```
【英語（現在）】
Text → english_cleaners2 → espeak IPA音素列 → 178シンボル語彙 → モデル → メル → HiFi-GAN → 波形

【日本語（新規）】
Text → japanese_cleaners → pyopenjtalk 韻律記号付き音素列 → 52シンボル語彙 → モデル → メル → HiFi-GAN → 波形
```

### 設計上の重要な注意点

- **後方互換性**: 英語の既存機能を壊さないよう、日本語対応は独立したシンボルテーブル・クリーナーとして追加する
- **複数文字音素**: 日本語音素には `ch`, `sh`, `cl`, `pau` 等の複数文字トークンがあるため、現在の1文字ずつ処理する `text_to_sequence()` の改修が必要
- **依存関係**: pyopenjtalkはoptional dependency（`[japanese]` extra）として追加し、英語のみのユーザーへの影響を最小化する

---

## 2. 現在のアーキテクチャ分析

### 2.1 テキスト処理パイプライン

| ファイル | 役割 |
|---------|------|
| `matcha/text/symbols.py` | 178シンボル定義（pad 1 + 句読点 16 + ASCII 52 + IPA 109） |
| `matcha/text/cleaners.py` | テキスト正規化・音素化（espeak-ng バックエンド） |
| `matcha/text/__init__.py` | `text_to_sequence()` / `sequence_to_text()` エントリーポイント |

**現在の処理フロー:**
```
入力テキスト
  → _clean_text(text, ["english_cleaners2"])
    → convert_to_ascii → lowercase → expand_abbreviations
    → global_phonemizer.phonemize() (espeak-ng, en-us)
    → remove_brackets → collapse_whitespace
  → 各シンボルを _symbol_to_id で整数IDに変換（1文字ずつ）
  → intersperse(seq, 0)  # ブランク挿入（DataModule側で実行）
  → テンソル化してモデルへ
```

**重要な制約**: 現在の `text_to_sequence()` は `for symbol in clean_text` で1文字ずつイテレートする設計。英語IPA音素は全て1文字のため問題ないが、日本語の複数文字音素（`ch`, `sh`, `ts`, `by`, `ny`, `cl`, `pau`, `sil` 等）には対応できない。日本語対応ではスペース区切りトークンへの対応が必須。

### 2.2 モデル構造とn_vocabの影響範囲

n_vocabが影響する箇所は **Embedding層のみ**:

```python
# matcha/models/components/text_encoder.py L346
self.emb = torch.nn.Embedding(n_vocab, self.n_channels)  # n_channels=192
```

それ以降のConformerエンコーダ、CFMデコーダ、Duration Predictor等は全て `n_channels` 次元で動作し、語彙サイズに依存しない。

### 2.3 データパイプライン

**ファイルリスト形式:**
```
# 単一話者
audio_path|text

# マルチスピーカー
audio_path|speaker_id|text
```

**メルスペクトログラム設定（全データセット共通）:**
- サンプルレート: 22050 Hz
- メルビン数: 80
- n_fft: 1024, hop_length: 256, win_length: 1024
- f_min: 0, f_max: 8000

**データ検証**: `text_mel_datamodule.py` の `get_mel()` で `assert sr == self.sample_rate` により厳密にチェックされる。サンプルレートが合わない場合は即座にエラーとなる。

### 2.4 Hydra設定構造

```
configs/
├── train.yaml           # マスター設定
├── data/
│   ├── ljspeech.yaml    # LJ Speech データ設定
│   └── vctk.yaml        # VCTK マルチスピーカー
├── model/
│   ├── matcha.yaml      # モデル設定 (n_vocab: 178)
│   ├── encoder/         # TextEncoder設定
│   ├── decoder/         # Decoder設定
│   ├── cfm/             # Flow Matching設定
│   └── optimizer/       # オプティマイザ設定
├── experiment/
│   ├── ljspeech.yaml
│   ├── ljspeech_min_memory.yaml
│   ├── ljspeech_from_durations.yaml
│   ├── multispeaker.yaml
│   └── hifi_dataset_piper_phonemizer.yaml
└── trainer/, callbacks/, logger/ ...
```

---

## 3. 日本語音素セット設計

### 3.1 OpenJTalk音素インベントリ

pyopenjtalkの`g2p()`が出力する標準的な音素一覧（44音素、ttslearn準拠）:

| カテゴリ | 音素 | 数 |
|---------|------|-----|
| 有声母音 | a, i, u, e, o | 5 |
| 無声化母音 | A, I, U, E, O | 5 |
| 基本子音 | b, d, f, g, h, j, k, m, n, p, r, s, t, v, w, y, z | 17 |
| 拗音子音 | by, dy, gy, hy, ky, my, ny, py, ry, ty | 10 |
| 破擦音等 | ch, sh, ts | 3 |
| 撥音 | N | 1 |
| 促音 | cl | 1 |
| ポーズ/無音 | pau, sil | 2 |

> **注意**: VOICEVOX等では外来語用の `gw`, `kw` を追加するが、ttslearn/ESPnetの標準音素セットには含まれない。本実装ではttslearn準拠を採用する。

### 3.2 韻律記号（prosody symbols）

ttslearn / ESPnet の `pyopenjtalk_prosody` モードで使用される7つの韻律記号:

| 記号 | 意味 |
|------|------|
| `^` | 発話開始 |
| `$` | 発話終了（平叙文） |
| `?` | 発話終了（疑問文） |
| `_` | ポーズ（句間無音） |
| `#` | アクセント句境界 |
| `[` | ピッチ上昇位置 |
| `]` | ピッチ下降位置 |

韻律記号方式では `pau` → `_`、`sil` → `^`/`$`/`?` に変換されるため、`pau`/`sil` は音素リストに含まれるが韻律記号使用時は直接出現しない。

### 3.3 推奨構成: ttslearn互換 52シンボル

ttslearn（r9y9氏作、書籍「Pythonで学ぶ音声合成」付属コード）と同じ構成を採用する。ESPnet の `pyopenjtalk_g2p_prosody` でも同等の構成が使用されている。

```python
# ttslearn互換: n_vocab = 52
# 構成: pad(1) + 韻律記号(7) + 音素(44) = 52

_pad = "~"                            # 1 （ttslearnに準拠、アンダースコアではなくチルダ）

# 韻律記号（7）
_extra_symbols = [
    "^",   # 発話開始
    "$",   # 発話終了（平叙文）
    "?",   # 発話終了（疑問文）
    "_",   # ポーズ
    "#",   # アクセント句境界
    "[",   # ピッチ上昇
    "]",   # ピッチ下降
]

# 音素（44）
_phonemes = [
    "A", "E", "I", "N", "O", "U",                          # 無声化母音 + 撥音 (6)
    "a", "b", "by", "ch", "cl", "d", "dy", "e", "f",      # (9)
    "g", "gy", "h", "hy", "i", "j", "k", "ky",            # (8)
    "m", "my", "n", "ny", "o", "p", "py",                  # (7)
    "r", "ry", "s", "sh", "t", "ts", "ty",                 # (7)
    "u", "v", "w", "y", "z",                               # (5)
    "pau", "sil",                                           # (2)
]

symbols_ja = [_pad] + _extra_symbols + _phonemes
# len(symbols_ja) = 1 + 7 + 44 = 52
```

**選定理由:**
- ttslearn/ESPnetで広く検証済み（書籍教材として定着）
- 韻律情報によりflow matchingデコーダがアクセントパターンを学習しやすい
- 語彙サイズが小さく（52）、Embedding層が軽量
- pyopenjtalkの`extract_fullcontext()` + `pp_symbols()`変換で生成可能

> **注意**: ttslearnではパッド記号に `~`（チルダ）を使用する。Matcha-TTSの英語版は `_`（アンダースコア）をパッドに使用しているが、日本語では `_` を韻律記号のポーズとして使用するため、パッドには `~` を採用する。

### 3.4 他実装との比較

| 実装 | シンボル数 | 方式 |
|------|-----------|------|
| **ttslearn (r9y9)** | **52** | **韻律記号付き（推奨）** |
| VOICEVOX | 45 | 基本音素 + gw/kw（韻律記号なし） |
| Bert-VITS2 | 42+記号 | 長音コロン表記（a:等）、トーン分離 |
| VITS-JP | ~40 | IPA風文字 |
| ESPnet prosody | ~52 | ttslearnと同等（`drop_unvoiced_vowels`オプションあり） |

> **ESPnet補足**: ESPnetの`pyopenjtalk_g2p_prosody`はデフォルトで`drop_unvoiced_vowels=True`であり、無声化母音（A,I,U,E,O）を有声母音（a,i,u,e,o）に変換する。本実装ではttslearnに合わせて無声化母音を区別する（`drop_unvoiced_vowels=False`相当）。

---

## 4. G2Pライブラリ選定

### 4.1 推奨: pyopenjtalk-plus

**日本語TTSのG2PはpyopenjtalkがデファクトスタンダードだがPython 3.11+ではビルド問題がある。プリビルドwheel提供のpyopenjtalk-plusを第一選択とする。**

| パッケージ | 最新版 | Python対応 | プリビルドwheel |
|----------|--------|-----------|----------------|
| pyopenjtalk（本家） | 0.4.1 (2025-04) | 3.8-3.10公式 | なし（ソースのみ） |
| **pyopenjtalk-plus** | **0.4.1.post7 (2025-11)** | **3.9-3.14** | **Win/Mac/Linux全対応** |
| pyopenjtalk-prebuilt | 0.3.0 (2023-04) | 3.6-3.11 | あり（非推奨、メンテ停止） |

pyopenjtalk-plusはpyopenjtalkのdrop-in replacement（`import pyopenjtalk`でそのまま使用可能）。

```python
import pyopenjtalk

# 基本G2P
pyopenjtalk.g2p("こんにちは")
# => 'k o N n i ch i w a'

# リスト出力
pyopenjtalk.g2p("こんにちは", join=False)
# => ['k', 'o', 'N', 'n', 'i', 'ch', 'i', 'w', 'a']

# フルコンテキストラベル（韻律情報を含む）
pyopenjtalk.extract_fullcontext("こんにちは")
# => HTS full-context labels（アクセント/韻律情報を含む）
```

**インストール:**
```bash
uv add pyopenjtalk-plus   # 推奨（Python 3.9-3.14対応、プリビルドwheel）
# または
uv add pyopenjtalk        # 本家（Python 3.10以下、ソースビルド必要）
```

### 4.2 比較検討した他の選択肢

| ライブラリ | 評価 | 理由 |
|----------|------|------|
| **pyopenjtalk(-plus)** | **採用** | 業界標準、G2P+韻律情報、全主要TTS実装で使用 |
| phonemizer + espeak-ng | 不採用 | 日本語で `RuntimeError: failed to load voice "ja"` 既知の問題 |
| cutlet | 不採用 | ローマ字変換のみ、音素レベルの粒度不足 |
| janome / MeCab | 不採用 | 形態素解析のみ、音素変換機能なし |
| Julius | 不採用 | 音素セグメンテーション用、テキスト→音素変換には不適 |

---

## 5. 日本語データセット

### 5.1 推奨: JSUT (Japanese Speech corpus of Saruwatari Lab)

| 項目 | 内容 |
|------|------|
| 話者 | 女性1名（単一話者） |
| 総録音時間 | 約10時間 |
| サンプルレート | 48000 Hz（22050 Hzにリサンプリング必要） |
| フォーマット | WAV |
| テキスト | 日本語テキスト（漢字仮名混じり） |
| サブセット | BASIC5000, TRAVEL1000, EMOTION100, LOANWORD128, VOICEACTRESS100, ONOMATOPEE300, REPEAT500, PRECEDENT130, COUNTERSUFFIX26 |
| ライセンス | CC-BY-SA 4.0 |
| ダウンロード | http://ss-takashi.sakura.ne.jp/corpus/jsut_ver1.1.zip |

**LJ Speech（英語）との対応:**
- LJ Speech: 英語女性1名、約24時間、22050 Hz
- JSUT: 日本語女性1名、約10時間、48000 Hz → リサンプリング必要

**リサンプリング手順:**
- 推奨手法: `torchaudio.transforms.Resample(48000, 22050)` または `librosa.resample`
- 48000→22050は非整数比のため、アンチエイリアスフィルタが重要
- 全約7,700ファイルの一括変換スクリプトが必要
- 変換後に `assert sr == 22050` で検証すること

### 5.2 その他の候補

| データセット | 話者 | 時間 | 用途 |
|------------|------|------|------|
| JVS | 100名 | ~30h | マルチスピーカー |
| Common Voice JA | 多数 | 可変 | 大規模だが品質にばらつき |
| CSS10 Japanese | 1名 | ~24h | LibriVox由来 |

### 5.3 ファイルリスト変換

JSUTのテキストファイルをMatcha-TTS形式に変換する必要がある:

```
# 変換前（JSUT transcript_utf8.txt）:
BASIC5000_0001:また東寺のある京都についても同様のことがいえる。

# 変換後（Matcha-TTS filelist形式）:
data/jsut/wavs/BASIC5000_0001.wav|また東寺のある京都についても同様のことがいえる。
```

**train/val分割**: 全サブセットから90%/10%のランダム分割を推奨。BASIC5000を主要な学習データとし、他サブセットは補助的に使用。

---

## 6. 実装対象ファイルと変更内容

### 6.1 変更されたファイル

| ファイル | 変更内容 | 状態 |
|---------|---------|--------|
| `pyproject.toml` | `[project.optional-dependencies]` に `japanese = ["pyopenjtalk-plus"]` 追加 | ✅ 完了 |
| `matcha/text/symbols.py` | 日本語シンボルリスト（`symbols_ja`）追加。既存の`symbols`は英語用として維持 | ✅ 完了 |
| `matcha/text/cleaners.py` | `japanese_cleaners` + `_fullcontext_to_prosody` 追加（pyopenjtalkは遅延import） | ✅ 完了 |
| `matcha/text/__init__.py` | スペース区切りトークン対応、`language` keyword-only引数追加、言語別マッピングキャッシュ | ✅ 完了 |
| `matcha/data/text_mel_datamodule.py` | `TextMelDataModule`/`TextMelDataset` に `language` パラメータ伝播 | ✅ 完了 |
| `matcha/cli.py` | `--language`/`--cleaners` 引数追加、`model.n_vocab` からの言語自動検出、`print_config` 拡張 | ✅ 完了 |
| `matcha/app.py` | 日本語モデル選択UI、`CURRENT_LANGUAGE` 追跡、JSUTモデル未存在時のガード | ✅ 完了 |
| `matcha/onnx/export.py` | `get_inputs()` に `n_vocab` パラメータ追加、モデルから語彙サイズ取得 | ✅ 完了 |
| `matcha/onnx/infer.py` | `--language`/`--cleaners` 引数追加、`process_text()` への伝播 | ✅ 完了 |
| `configs/data/jsut.yaml` | JSUTデータ設定（`cleaners: [japanese_cleaners]`, `language: ja`） | ✅ 完了 |
| `configs/experiment/jsut.yaml` | JSUT実験設定（`model.n_vocab: 52` オーバーライド） | ✅ 完了 |
| `scripts/prepare_jsut.py` | JSUTデータ準備スクリプト（リサンプリング + ファイルリスト生成 + train/val分割） | ✅ 完了 |
| `scripts/transfer_from_english.py` | 英語→日本語チェックポイント変換（Embedding層再初期化） | ✅ 完了 |
| `tests/test_text_ja.py` | 日本語テキスト処理テスト 17件 | ✅ 完了 |
| `tests/test_onnx_ja.py` | ONNX語彙範囲テスト 7件 | ✅ 完了 |

### 6.2 変更不要だったファイル

以下のモジュールは言語非依存のため変更不要:

- `matcha/models/matcha_tts.py` — メインモデル（`n_vocab` は設定で注入されるのみ）
- `matcha/models/components/text_encoder.py` — エンコーダ（`n_vocab` は `nn.Embedding` のサイズのみに影響）
- `matcha/models/components/flow_matching.py` — CFMデコーダ
- `matcha/models/components/decoder.py` — U-Netデコーダ
- `matcha/hifigan/` — ボコーダ全体
- `matcha/utils/` — ユーティリティ全般

> **注**: `matcha/data/text_mel_datamodule.py` は当初「変更不要」と想定していたが、`jsut.yaml` の `language: ja` キーを `TextMelDataModule` → `TextMelDataset` → `text_to_sequence()` へ伝播するために `language` パラメータの追加が必要だった。

### 6.3 主要な実装詳細

#### symbols.py の変更

既存の英語シンボル（`symbols`）はそのまま維持し、日本語シンボルを別名で追加:

```python
# === 日本語音素シンボル（ttslearn互換、52シンボル） ===

_pad_ja = "~"  # ttslearn準拠のパッド文字（英語の "_" と区別）

_extra_symbols_ja = ["^", "$", "?", "_", "#", "[", "]"]  # 韻律記号 (7)

_phonemes_ja = [
    # 無声化母音 + 撥音
    "A", "E", "I", "N", "O", "U",
    # 有声母音 + 子音（アルファベット順）
    "a", "b", "by", "ch", "cl", "d", "dy", "e", "f",
    "g", "gy", "h", "hy", "i", "j", "k", "ky",
    "m", "my", "n", "ny", "o", "p", "py",
    "r", "ry", "s", "sh", "t", "ts", "ty",
    "u", "v", "w", "y", "z",
    "pau", "sil",
]  # 44音素

symbols_ja = [_pad_ja] + _extra_symbols_ja + _phonemes_ja
# len(symbols_ja) = 1 + 7 + 44 = 52
```

> **実装済み**: `matcha/text/symbols.py` に上記の通り実装。

#### cleaners.py の変更（実装済み）

pyopenjtalkは遅延importにすることで、英語のみのユーザーへの影響を防ぐ:

```python
# 正規表現パターン（モジュールレベル定義）
_PHONEME_RE = re.compile(r"-([^+]+)\+")
_A1_RE = re.compile(r"/A:([0-9-]+)\+")
_A2_RE = re.compile(r"\+(\d+)\+")
_F1_RE = re.compile(r"/F:(\d+)_")


def _fullcontext_to_prosody(labels):
    """HTS full-context labels → 韻律記号付き音素列。

    ttslearnのpp_symbols()を参考に実装。主な処理:
    1. 各ラベルから音素部分を正規表現で抽出（p3フィールド）
    2. 発話開始/終了の sil を ^/$/? に変換
    3. pau を _ に変換
    4. アクセント句境界を # に変換
    5. ピッチ上昇/下降位置を [/] に変換
       - a1（モーラ位置）、a2（アクセント核位置）、f1（モーラ数）の比較で判定
    """
    phonemes = []
    for i, label in enumerate(labels):
        m = _PHONEME_RE.search(label)
        if m is None:
            continue
        ph = m.group(1)
        if ph == "sil":
            if i == 0:
                phonemes.append("^")
            elif i == len(labels) - 1:
                phonemes.append("$")
            else:
                phonemes.append("?")
            continue
        if ph == "pau":
            phonemes.append("_")
            continue
        a1 = int(_A1_RE.search(label).group(1))
        a2 = int(_A2_RE.search(label).group(1))
        f1 = int(_F1_RE.search(label).group(1))
        a2_next = int(_A2_RE.search(labels[i + 1]).group(1)) if i + 1 < len(labels) else -1
        if a1 == 1 and a2_next == 1:
            phonemes.append("#")
        if a2 == 1 and a1 == 1:
            phonemes.append("[")
        elif a1 == a2 + 1 and a2 != f1:
            phonemes.append("]")
        phonemes.append(ph)
    return phonemes


def japanese_cleaners(text):
    """日本語テキスト→韻律記号付き音素列への変換。"""
    import pyopenjtalk  # 遅延import（optional dependency）
    labels = pyopenjtalk.extract_fullcontext(text)
    phonemes = _fullcontext_to_prosody(labels)
    return " ".join(phonemes)
```

#### __init__.py の変更（実装済み）

複数文字音素に対応するため、`text_to_sequence` を拡張。既存の `_symbol_to_id`/`_id_to_symbol` モジュール変数は英語テスト（`test_text.py`）の後方互換のために維持:

```python
from matcha.text.symbols import symbols, symbols_ja

_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

# 言語別のシンボルマッピングを遅延構築
_symbol_maps = {}

def _get_symbol_map(language):
    """Return (symbol_to_id, id_to_symbol) for the given language."""
    if language not in _symbol_maps:
        if language == "en":
            _symbol_maps[language] = (_symbol_to_id, _id_to_symbol)
        elif language == "ja":
            s2i = {s: i for i, s in enumerate(symbols_ja)}
            i2s = {i: s for i, s in enumerate(symbols_ja)}
            _symbol_maps[language] = (s2i, i2s)
        else:
            raise ValueError(f"Unsupported language: {language}")
    return _symbol_maps[language]


def text_to_sequence(text, cleaner_names, *, language="en"):
    """テキスト→シンボルID列。language は keyword-only 引数（後方互換）。"""
    clean_text = _clean_text(text, cleaner_names)
    if language == "ja":
        sym2id, _ = _get_symbol_map("ja")
        sequence = [sym2id[symbol] for symbol in clean_text.split()]
    else:
        sequence = [_symbol_to_id[symbol] for symbol in clean_text]
    return sequence, clean_text


def cleaned_text_to_sequence(cleaned_text, *, language="en"):
    if language == "ja":
        sym2id, _ = _get_symbol_map("ja")
        return [sym2id[symbol] for symbol in cleaned_text.split()]
    return [_symbol_to_id[symbol] for symbol in cleaned_text]


def sequence_to_text(sequence, *, language="en"):
    if language == "ja":
        _, id2sym = _get_symbol_map("ja")
        return " ".join(id2sym[symbol_id] for symbol_id in sequence)
    return "".join(_id_to_symbol[symbol_id] for symbol_id in sequence)
```

#### configs/experiment/jsut.yaml（実装済み）

```yaml
# @package _global_

# Japanese TTS training with JSUT corpus
# To execute: python matcha/train.py experiment=jsut

defaults:
  - override /data: jsut.yaml

tags: ["jsut", "japanese"]
run_name: jsut

model:
  n_vocab: 52
```

#### matcha/data/text_mel_datamodule.py の変更（実装済み）

`language` パラメータを `TextMelDataModule.__init__` → `TextMelDataset.__init__` → `get_text()` → `text_to_sequence()` へ伝播:

```python
class TextMelDataModule(LightningDataModule):
    def __init__(self, ..., language="en"):  # デフォルト "en" で後方互換
        super().__init__()
        self.save_hyperparameters()  # self.hparams.language が自動保存

    def setup(self, stage=None):
        # getattr で古いチェックポイントとの互換性を確保
        TextMelDataset(..., language=getattr(self.hparams, 'language', 'en'))

class TextMelDataset(torch.utils.data.Dataset):
    def __init__(self, ..., language="en"):
        self.language = language

    def get_text(self, text, add_blank=True):
        text_norm, cleaned_text = text_to_sequence(text, self.cleaners, language=self.language)
```

#### matcha/cli.py の変更（実装済み）

言語自動検出: モデルロード後に `model.n_vocab` を確認し、`--language` が未指定の場合に自動設定:

```python
# --language のデフォルトは None（auto-detect）
parser.add_argument("--language", type=str, default=None, choices=["en", "ja"],
                    help="Language for text processing (default: auto-detect from model)")

# モデルロード後に自動検出
model = load_matcha(args.model, paths["matcha"], device)
if args.language is None:
    if hasattr(model, "n_vocab") and model.n_vocab == 52:
        args.language = "ja"
        print("[*] Auto-detected language: Japanese (n_vocab=52)")
    else:
        args.language = "en"
if args.cleaners is None:
    args.cleaners = ["japanese_cleaners"] if args.language == "ja" else None
```

#### matcha/onnx/export.py の変更（実装済み）

ダミー入力の語彙範囲をモデルの `n_vocab` から動的取得:

```python
def get_inputs(is_multi_speaker, n_vocab=178):
    x = torch.randint(low=0, high=n_vocab, size=(1, 50), dtype=torch.long)
    ...

# main() で呼び出し
n_vocab = matcha.n_vocab
dummy_input, input_names = get_inputs(is_multi_speaker, n_vocab=n_vocab)
```

#### matcha/app.py の変更（実装済み）

JSUTモデル未存在時のガード:

```python
def load_model_ui(model_type, textbox):
    ...
    if model_name != CURRENTLY_LOADED_MODEL:
        model_path = MATCHA_TTS_LOC(model_name)
        if not model_path.exists():
            gr.Warning(f"モデル '{model_name}' が見つかりません: {model_path}")
            return (textbox, gr.Button(interactive=False), ...)
```

---

## 7. HiFi-GANボコーダの互換性

### 結論: 既存のHiFi-GANをそのまま使用可能

HiFi-GANは80次元メルスペクトログラム→22050Hz波形の変換器であり、**完全に言語非依存**。

| コンポーネント | 言語依存性 | 日本語で再学習必要 |
|-------------|----------|-----------------|
| Generator (アップサンプリング) | なし | 不要 |
| Denoiser (バイアス除去) | なし | 不要 |
| メル設定 (80bins, 22050Hz) | なし | 不要 |

**使用するボコーダ:**
- `hifigan_univ_v1` (Universal) — 多話者データで学習済み、汎用性が高い
- 日本語特化の品質向上が必要な場合のみ、日本語音声データでファインチューニング

---

## 8. 転移学習戦略

### 英語→日本語の重み再利用

| コンポーネント | 再利用可否 | 理由 |
|-------------|----------|------|
| **Embedding層** | 不可 | n_vocab変更（178→52）でサイズ不一致 |
| **Prenet (ConvReluNorm)** | 可能 | n_channels=192で動作、語彙非依存 |
| **Conformerエンコーダ** | 可能 | 自己注意メカニズムは言語非依存 |
| **Duration Predictor** | 条件付き | 英語と日本語では継続時間分布が異なる（モーラ等時性 vs ストレスタイミング）。効果は限定的な可能性あり |
| **CFM Decoder (U-Net)** | 可能 | メル空間処理は言語非依存 |
| **HiFi-GAN** | 可能 | 完全に言語非依存 |

### 推奨戦略

```
1. 英語学習済みチェックポイントをロード（strict=Falseが必要）
2. Embedding層のみランダム初期化（新語彙サイズで再作成）
3. 残りの全レイヤーは英語の重みを保持
4. 初期は段階的アンフリーズを検討:
   - Step 1: Embedding層のみ学習（他レイヤーはfreeze）
   - Step 2: 全レイヤーを解凍してファインチューニング
5. 学習率は低め（例: 5e-5）からスタート
```

**転移学習ローダー実装（必須）:**

PyTorch Lightningの `load_from_checkpoint` はデフォルト `strict=True` であり、Embeddingサイズが異なるチェックポイントからのロードはエラーになる。カスタムローダーが必要:

```python
# state_dictからemb.weightを除外してロード
state_dict = torch.load(ckpt_path, weights_only=False)["state_dict"]
state_dict = {k: v for k, v in state_dict.items() if "emb.weight" not in k}
model.load_state_dict(state_dict, strict=False)
```

---

## 9. 学習・推論フロー

### 9.1 学習フロー

```
1. データ準備
   - JSUT音声を22050Hzにリサンプリング（torchaudio.transforms.Resample推奨）
   - ファイルリスト作成（train.txt / val.txt）

2. データ統計計算（Phase 1完了後に実行可能）
   $ uv run matcha-data-stats -i jsut.yaml
   → configs/data/jsut.json 出力 → jsut.yaml の data_statistics に手動コピー

3. 学習実行
   $ uv run python matcha/train.py experiment=jsut

4. 学習内部処理:
   テキスト → japanese_cleaners → スペース区切り音素列
     → text_to_sequence(language="ja") → 音素ID列
     → intersperse(blank) → TextEncoder(Conformer) → mu_x, logw
     → MAS alignment → duration target
     → CFM Decoder → flow matching loss
   損失 = duration_loss + prior_loss + flow_matching_loss
```

### 9.2 推論フロー

```
日本語テキスト入力
  → japanese_cleaners (pyopenjtalk extract_fullcontext + pp_symbols)
  → スペース区切り音素列 → text_to_sequence(language="ja") → 音素ID列
  → intersperse(blank)
  → TextEncoder → mu_x, logw
  → Duration予測 → alignment生成
  → CFM Decoder (Euler ODE, n_timesteps)
  → denormalize(mel_mean, mel_std)
  → HiFi-GAN → 22050Hz 波形
  → WAV出力
```

---

## 10. 実装ステップ

### Phase 1: テキスト処理パイプライン ✅ 完了

1. ✅ `pyproject.toml` に optional dependency 追加: `japanese = ["pyopenjtalk-plus"]`
2. ✅ `symbols.py` に `symbols_ja` (52シンボル) を追加（既存の `symbols` は維持）
3. ✅ `cleaners.py` に `japanese_cleaners` + `_fullcontext_to_prosody` 実装（正規表現パーサー約60行）
4. ✅ `__init__.py` を改修: `language` keyword-only引数、スペース区切りトークン対応、言語別マッピングキャッシュ
5. ✅ `text_mel_datamodule.py` に `language` パラメータ伝播（`TextMelDataModule` → `TextMelDataset` → `text_to_sequence`）
6. ✅ テスト作成: `tests/test_text_ja.py` (17件) — シンボル検証、roundtrip、cleaners（pyopenjtalk依存はスキップ可）、英語後方互換

### Phase 2: データ準備 ✅ 完了（スクリプト・設定作成済み）

1. ✅ `scripts/prepare_jsut.py` 作成 — リサンプリング (48kHz→22050Hz) + ファイルリスト生成 + train/val 90/10分割
2. ✅ `configs/data/jsut.yaml` 作成 — `cleaners: [japanese_cleaners]`, `language: ja`, `data_statistics: null`
3. ⏳ 実データ処理は手動実行:
   ```bash
   # JSUTデータ準備
   uv run python scripts/prepare_jsut.py --jsut-dir data/jsut_ver1.1 --output-dir data/jsut
   # データ統計計算
   uv run matcha-data-stats -i jsut.yaml
   # → jsut.yaml の data_statistics に値をコピー
   ```

### Phase 3: 学習設定 ✅ 完了

1. ✅ `configs/experiment/jsut.yaml` 作成（`model.n_vocab: 52` オーバーライド）
2. ✅ `scripts/transfer_from_english.py` 作成 — Embedding層を再初期化して日本語チェックポイント生成

### Phase 4: 学習実行 ⏳ 未着手（手動実行ステップ）

```bash
# スクラッチ学習
uv run python matcha/train.py experiment=jsut

# 転移学習
uv run python scripts/transfer_from_english.py \
    --source matcha_ljspeech.ckpt --target matcha_jsut_init.ckpt
uv run python matcha/train.py experiment=jsut ckpt_path=matcha_jsut_init.ckpt
```

### Phase 5: 推論・UI対応 ✅ 完了

1. ✅ `cli.py` に `--cleaners` / `--language` 引数追加、`model.n_vocab` からの言語自動検出（`--language` デフォルト=`None`→auto-detect）
2. ✅ `app.py` のGradio UIで日本語モデル選択追加、JSUTモデル未存在時の `gr.Warning` ガード、日本語サンプルテキスト
3. ✅ ONNX エクスポート対応: `get_inputs()` の語彙範囲を `model.n_vocab` から動的取得
4. ✅ ONNX 推論対応: `--language` / `--cleaners` 引数追加
5. ✅ テスト: `tests/test_onnx_ja.py` (7件) — `get_inputs` の語彙範囲テスト

---

## 11. 既存の日本語Matcha-TTS実装

| プロジェクト | 方法 | 状態 |
|------------|------|------|
| [akjava/Matcha-TTS-Japanese](https://github.com/akjava/Matcha-TTS-Japanese) | pyopenjtalk + ONNX | 非公式。自動G2Pを提供せず、ユーザーが事前に音素列を用意する設計 |
| [PR #166 (Multi-language)](https://github.com/shivammehta25/Matcha-TTS/pull/166) | 言語埋め込み追加（テキスト処理は含まない） | 公式PRだがOPEN状態（未マージ） |
| [公式Wiki: 多言語対応](https://github.com/shivammehta25/Matcha-TTS/wiki/) | ljspeech.yamlをテンプレートにカスタマイズ | ガイドラインのみ（日本語固有の情報なし） |

### 関連する日本語TTS（2024-2025年の動向）

| プロジェクト | 特徴 |
|------------|------|
| F5-TTS | フローマッチングベース、多言語対応 |
| CosyVoice (Alibaba) | LLMベース、日本語含む9言語公式サポート |
| Kokoro-82M | StyleTTS 2ベース軽量モデル、日本語含む8言語 |
| GPT-SoVITS | Few-shot音声クローニング、日本語対応 |

---

## 12. 参考文献

- ttslearn (r9y9): https://github.com/r9y9/ttslearn
  - 特に `ttslearn/tacotron/frontend/openjtalk.py` の `pp_symbols()` が `_fullcontext_to_prosody` の参照実装
- pyopenjtalk: https://github.com/r9y9/pyopenjtalk
- pyopenjtalk-plus: https://github.com/tsukumijima/pyopenjtalk-plus
- ESPnet JSUT recipe: https://github.com/espnet/espnet/tree/master/egs2/jsut/tts1
- ESPnet phoneme_tokenizer.py: https://github.com/espnet/espnet/blob/master/espnet2/text/phoneme_tokenizer.py
- JSUT corpus: http://ss-takashi.sakura.ne.jp/corpus/jsut_ver1.1.zip
- Style-Bert-VITS2: https://github.com/litagin02/Style-Bert-VITS2
- Matcha-TTS公式Wiki: https://github.com/shivammehta25/Matcha-TTS/wiki/
