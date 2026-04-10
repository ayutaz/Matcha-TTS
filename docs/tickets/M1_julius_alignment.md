# Milestone M1: Julius Forced Alignment パイプライン構築

## マイルストーン概要

MAS（Monotonic Alignment Search）が多話者JVS学習で構造的に退化する問題（学習サンプルの39-43%が退化アライメント）を解決するため、Julius forced alignerによる音素duration事前計算パイプラインを構築する。

Juliusは日本語音声認識のためのViterbiアルゴリズムベースforced alignerであり、JATTS（名古屋大/戸田研）がJulius + Matcha-TTSの日本語パイプラインを実証済みである。本マイルストーンでは、JVSコーパス全100話者（parallel100サブセット、約10,000発話）に対してJuliusアライメントを実行し、pyopenjtalk 55シンボル体系に準拠したdurationフレーム配列を生成する。

### 一から作り直すとしたらの思考（マイルストーン全体）

もしM1全体を白紙からやり直すなら、以下の設計判断を採る:

1. **アライナー選択**: Juliusの選択は正しい。MFA（Montreal Forced Aligner）にはJVSでの既知問題（GitHub issue #541）があり、日本語音素セットのIPA→ローマ字マッピングが複雑。Juliusは日本語ネイティブで10ms精度、JATTSで実証済み。ただし、将来的にはwhisper-alignのような言語非依存アライナーの台頭を注視すべき。

2. **音素マッピング設計**: 最初からpyopenjtalkのフルコンテキストラベルとJulius音素の両方を入力として受け取る統一的な正規化レイヤーを設計する。現行の`japanese_cleaners`がpyopenjtalkに依存しているため、Juliusの出力をpyopenjtalk空間に射影するのが最も低リスク。マッピングテーブルは外部JSONではなくPythonモジュール内の辞書として持ち、型チェックとテストを容易にする。

3. **データフロー設計**: `.lab`ファイルのパースとduration配列変換を1つの統合スクリプトにまとめるのではなく、(a) Julius実行、(b) .labパース＋音素マッピング、(c) duration配列生成の3段階に分離する。各段階で中間出力を検証可能にし、デバッグコストを下げる。

4. **blank intersperseの扱い**: blank durationのデフォルト値を0フレームではなく1フレームにする設計を検討する。MASの退化はblank[0]へのフレーム集中が本質であり、外部アライナーでは各blankに均等に最小durationを割り当てることで、Duration Predictorが学習しやすいターゲットを得られる。ただし、これは実験的検証が必要な設計判断であり、T-M1-04の品質検証結果を見て決定する。

5. **スケーラビリティ**: 10,000発話程度なら逐次処理で十分だが、将来的にJVS以外のコーパス（JSUT、Common Voice Japanese等）にも適用することを見据え、並列処理の仕組みを最初から入れておく。

### 依存関係

```
T-M1-01 (Julius環境構築・アライメント実行)
    └──→ T-M1-02 (音素マッピングテーブル作成)  ※T-M1-01のJulius出力音素セット確認が必要
            └──→ T-M1-03 (.lab→durationフレーム配列変換)  ※T-M1-02のマッピングが必須
                    └──→ T-M1-04 (品質検証・統計分析)  ※T-M1-03の出力が必須
```

- T-M1-01とT-M1-02は部分的に並行可能（マッピングテーブルの設計はJulius出力前に着手できるが、最終検証にはJulius出力が必要）
- 後続マイルストーンM2（PrecomputedDataModuleのduration対応）はT-M1-03の出力フォーマット確定後に着手可能

### 完了条件

1. JVS全100話者のparallel100発話（約10,000件）に対して`.lab`ファイルが生成されていること
2. Julius音素→pyopenjtalk 55シンボルのマッピングテーブルが網羅的に定義され、全`.lab`ファイルに未マッピング音素がないこと
3. 各発話に対して、blank intersperse済み音素列と同一長のdurationフレーム配列（`IntTensor`）が生成されていること
4. durationフレーム配列の合計がメルスペクトログラムのフレーム数と一致すること（許容誤差: +/-1フレーム）
5. 退化アライメント率（音素の80%以上が1フレーム以下のサンプル）が0%であること
6. 全テストが通過すること（各チケットで定義するユニットテスト＋統合テスト）

### 想定期間

| チケット | 想定工数 | 備考 |
|---------|---------|------|
| T-M1-01 | 2-3日 | Julius環境構築のトラブルシュートに時間を要する可能性あり |
| T-M1-02 | 1-2日 | 音素セットの差異調査が主な作業 |
| T-M1-03 | 2-3日 | エッジケースの処理とテストが中心 |
| T-M1-04 | 1-2日 | 分析スクリプト作成と結果の解釈 |
| **合計** | **6-10日** | |

---

## T-M1-01: Julius segmentation-kit環境構築・JVSアライメント実行

### 1. タスク目的とゴール

**目的**: JVSコーパス全100話者のparallel100発話に対して、Julius forced alignerを使用して音素レベルの時間アライメント（`.lab`ファイル）を生成する。

**ゴール**:
- Julius segmentation-kitが実行可能な環境を構築する
- JVS転記テキスト（ひらがな）をJuliusの入力フォーマットに変換する
- JVS音声ファイルを16kHz（Julius要求）にリサンプリングする
- 全約10,000発話のアライメントを実行し、`.lab`ファイル（音素開始/終了時刻）を出力する

**なぜ必要か**: MASが多話者設定で構造的に退化（39-43%の学習サンプルで退化アライメント）するため、外部forced alignerによる正確な音素duration取得が必要。Juliusは日本語ネイティブのViterbiアライナーで、JATTSが同一パイプライン（Julius + Matcha-TTS + JVS）を実証済み。

### 2. 実装する内容の詳細

#### 2.1 Julius segmentation-kitのインストール

Julius segmentation-kit（`julius-speech/segmentation-kit`）をクローンしセットアップする。

```bash
# segmentation-kitのクローン
git clone https://github.com/julius-speech/segmentation-kit.git tools/segmentation-kit

# Juliusバイナリの確認（segmentation-kitにバンドルされている場合あり）
# なければ別途インストール
# Ubuntu: sudo apt-get install julius
# macOS: brew install julius
```

segmentation-kitの構成確認:
- `segment_julius.pl` または `run.sh`: メインの実行スクリプト
- `model/`: 音響モデル（日本語トライフォンHMM）
- 入力要件: 16kHz 16bit モノラルWAV + ひらがなテキスト

**作成ファイル**: `scripts/setup_julius.sh`

```bash
#!/bin/bash
# Julius segmentation-kitのセットアップスクリプト
set -euo pipefail

TOOLS_DIR="tools"
SEGKIT_DIR="${TOOLS_DIR}/segmentation-kit"

if [ -d "${SEGKIT_DIR}" ]; then
    echo "segmentation-kit already exists at ${SEGKIT_DIR}"
    exit 0
fi

mkdir -p "${TOOLS_DIR}"
git clone https://github.com/julius-speech/segmentation-kit.git "${SEGKIT_DIR}"

# Juliusバイナリの存在確認
if ! command -v julius &> /dev/null; then
    echo "WARNING: julius command not found. Install via:"
    echo "  Ubuntu: sudo apt-get install julius"
    echo "  macOS:  brew install julius"
fi

echo "Setup complete. segmentation-kit at: ${SEGKIT_DIR}"
```

#### 2.2 JVS転記テキストの準備

JVSコーパスの転記テキストフォーマット:
- `jvs_ver1/jvsXXX/parallel100/transcripts_utf8.txt`: **漢字かな混じり文**を含む（例: "今日は天気がいいです"）
- `jvs_ver1/jvsXXX/parallel100/VOICEACTRESS100_001.txt`等: 個別テキストファイル（存在する場合）

Juliusのsegmentation-kitはひらがなテキストを必要とする。JVSにはひらがな転記が含まれている場合と含まれていない場合がある。含まれていない場合はpyopenjtalkで漢字→ひらがな変換を行う。

**テキスト形式の変換に関する注意事項**:
- `pyopenjtalk.g2p(text, kana=True)`は**カタカナ**を出力する（ひらがなではない）。例: "こんにちは" → "コンニチワ"
- segmentation-kitが期待するテキスト形式（カタカナ、ひらがな、ローマ字のいずれか）を最初のステップで確認すること
- segmentation-kitがひらがなのみを受け付ける場合、カタカナ→ひらがな変換が必要:
  ```python
  # カタカナ→ひらがな変換のフォールバック処理
  import unicodedata
  def katakana_to_hiragana(text: str) -> str:
      result = []
      for ch in text:
          cp = ord(ch)
          if 0x30A1 <= cp <= 0x30F6:  # ァ-ヶ
              result.append(chr(cp - 0x60))
          else:
              result.append(ch)
      return "".join(result)
  ```
- segmentation-kitがカタカナを直接受け付ける場合は、`pyopenjtalk.g2p(text, kana=True)`の出力をそのまま使用可能

**作成ファイル**: `scripts/prepare_julius_input.py`

```python
"""JVSコーパスからJulius segmentation-kit用の入力を準備する。

処理内容:
1. JVS転記テキストを読み込み、ひらがなに変換（必要に応じてpyopenjtalk使用）
2. 音声ファイルを16kHz 16bit モノラルWAVにリサンプリング
3. segmentation-kit用のディレクトリ構造を生成

出力構造:
    julius_input/
        jvs001/
            VOICEACTRESS100_001.wav  (16kHz)
            VOICEACTRESS100_001.txt  (ひらがなテキスト)
        jvs002/
            ...
"""

import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import soundfile as sf
import numpy as np
from tqdm import tqdm


JULIUS_SAMPLE_RATE = 16000


def read_transcripts(transcript_path: Path) -> dict[str, str]:
    """JVSのtranscripts_utf8.txtを読み込み、{utterance_id: text}の辞書を返す。"""
    transcripts = {}
    with open(transcript_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            utt_id, text = line.split(":", 1)
            transcripts[utt_id.strip()] = text.strip()
    return transcripts


def kanji_to_hiragana(text: str) -> str:
    """pyopenjtalkを使用して漢字かな混じり文をひらがなに変換する。"""
    import pyopenjtalk
    # pyopenjtalk.g2pでひらがな出力を得る
    # kana=Trueでカタカナ出力 → ひらがなに変換
    kana = pyopenjtalk.g2p(text, kana=True)
    # カタカナ→ひらがな変換
    hiragana = ""
    for ch in kana:
        cp = ord(ch)
        if 0x30A1 <= cp <= 0x30F6:  # カタカナ→ひらがな
            hiragana += chr(cp - 0x60)
        else:
            hiragana += ch
    return hiragana


def resample_to_16k(input_path: Path, output_path: Path):
    """音声ファイルを16kHz 16bit モノラルWAVにリサンプリングする。"""
    import torch
    import torchaudio

    waveform, sr = torchaudio.load(str(input_path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != JULIUS_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, JULIUS_SAMPLE_RATE)
        waveform = resampler(waveform)
    # 16bit PCM WAVとして保存
    sf.write(str(output_path), waveform.squeeze(0).numpy(), JULIUS_SAMPLE_RATE, subtype="PCM_16")


def process_speaker(
    spk_dir: Path,
    output_dir: Path,
    source_sr: int,
) -> tuple[str, int, list[str]]:
    """1話者分の音声リサンプリングとテキスト変換を行う。"""
    spk_name = spk_dir.name  # e.g., "jvs001"
    parallel_dir = spk_dir / "parallel100"
    wav_dir = parallel_dir / "wav24kHz16bit"

    if not wav_dir.exists():
        return spk_name, 0, [f"wav directory not found: {wav_dir}"]

    # 転記テキスト読み込み
    transcript_path = parallel_dir / "transcripts_utf8.txt"
    if not transcript_path.exists():
        return spk_name, 0, [f"transcripts not found: {transcript_path}"]

    transcripts = read_transcripts(transcript_path)
    out_spk_dir = output_dir / spk_name
    out_spk_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    errors = []

    for wav_path in sorted(wav_dir.glob("*.wav")):
        utt_id = wav_path.stem
        if utt_id not in transcripts:
            errors.append(f"No transcript for {utt_id}")
            continue

        try:
            # 音声リサンプリング
            out_wav = out_spk_dir / f"{utt_id}.wav"
            resample_to_16k(wav_path, out_wav)

            # ひらがなテキスト生成
            hiragana = kanji_to_hiragana(transcripts[utt_id])
            out_txt = out_spk_dir / f"{utt_id}.txt"
            out_txt.write_text(hiragana, encoding="utf-8")

            processed += 1
        except Exception as e:
            errors.append(f"{utt_id}: {e}")

    return spk_name, processed, errors
```

#### 2.3 Julius forced alignmentの実行

**作成ファイル**: `scripts/run_julius_alignment.sh`

```bash
#!/bin/bash
# JVS全話者に対してJulius forced alignmentを実行する
set -euo pipefail

SEGKIT_DIR="tools/segmentation-kit"
INPUT_DIR="$1"    # scripts/prepare_julius_input.pyの出力ディレクトリ
OUTPUT_DIR="$2"   # .labファイルの出力先

mkdir -p "${OUTPUT_DIR}"

for spk_dir in "${INPUT_DIR}"/jvs*; do
    spk_name=$(basename "${spk_dir}")
    out_dir="${OUTPUT_DIR}/${spk_name}"
    mkdir -p "${out_dir}"

    echo "Processing ${spk_name}..."

    for wav_file in "${spk_dir}"/*.wav; do
        utt_id=$(basename "${wav_file}" .wav)
        txt_file="${spk_dir}/${utt_id}.txt"

        if [ ! -f "${txt_file}" ]; then
            echo "  SKIP: ${utt_id} (no text file)"
            continue
        fi

        # Julius segmentation-kit実行
        # 出力: .lab ファイル（音素 開始時刻 終了時刻）
        # 具体的なコマンドはsegmentation-kitのバージョンにより異なる
        # 以下は一般的な形式
        cd "${SEGKIT_DIR}" && \
        perl segment_julius.pl "${wav_file}" "${txt_file}" "${out_dir}/${utt_id}.lab" \
            2>/dev/null || echo "  ERROR: ${utt_id}"

    done
    echo "  Done: ${spk_name}"
done

echo "All speakers processed. Output: ${OUTPUT_DIR}"
```

**注意**: Julius segmentation-kitの実際のインターフェースはバージョンにより異なる。`segment_julius.pl`のパラメータ、ディレクトリ構造、音響モデルのパス等は実際のインストール後に調整が必要。

**segmentation-kitのインターフェースに関する重要事項**:

segmentation-kitには2つの可能なインターフェースが存在する:

1. **ディレクトリ方式**（公式READMEに記載されている可能性が高い）: `wav/`と`txt/`ディレクトリにファイルを配置し、`segment_julius.pl`を引数なしで実行する方式。この場合、上記の`run_julius_alignment.sh`のコマンドライン引数方式は動作しない。
   ```bash
   # ディレクトリ方式の場合
   cd tools/segmentation-kit
   # wav/ と txt/ に入力ファイルを配置
   perl segment_julius.pl
   ```

2. **引数方式**: 上記スクリプト例のように、wavファイルとテキストファイルのパスを引数として渡す方式。バージョンやフォークによっては対応している場合がある。

**最初のステップとして、segmentation-kitのREADMEおよび`segment_julius.pl`のソースを確認し、実際のインターフェースを確定すること**。ディレクトリ方式の場合は、`prepare_julius_input.py`の出力構造をsegmentation-kitの期待するディレクトリ構造（`wav/`、`txt/`）に合わせるか、話者ごとにシンボリックリンクを作成する方式に変更する。

**フォールバック計画**: Juliusの環境構築に深刻な問題が発生した場合（segmentation-kitのPerl依存解決不能、音響モデル非互換等）、MFA（Montreal Forced Aligner）日本語モデルv2.0.1aにフォールバックする。具体的な手順:
- `pip install montreal-forced-aligner`
- MFA日本語pretrained modelのダウンロード: `mfa model download acoustic japanese_mfa`
- JVSのIPA転記生成（pyopenjtalk→IPA変換スクリプトの追加実装が必要）
- 既知問題: MFA GitHub issue #541（JVS特有の問題報告あり）のため、事前にissueの状況を確認すること

**Docker化による再現性確保（推奨）**: Julius segmentation-kitはPerl依存があり、環境差異によるトラブルが予想される。以下のDockerfileによる再現可能な環境構築を推奨する:
```dockerfile
FROM ubuntu:22.04
RUN apt-get update && apt-get install -y julius perl sox
COPY tools/segmentation-kit /opt/segmentation-kit
WORKDIR /opt/segmentation-kit
```

#### 2.4 .labファイルの期待フォーマット

Julius segmentation-kitの出力`.lab`ファイルは、HTK形式のラベルファイル:

```
0 2100000 silB
2100000 3200000 k
3200000 4500000 o
4500000 5800000 N
5800000 7100000 n
7100000 8900000 i
8900000 10200000 ch
10200000 11500000 i
11500000 12800000 w
12800000 14100000 a
14100000 16000000 silE
```

時間単位は100ns（10^-7秒）。秒への変換: `seconds = value / 10_000_000`

#### 2.5 出力ディレクトリ構造

```
data/julius_alignment/
    jvs001/
        VOICEACTRESS100_001.lab
        VOICEACTRESS100_002.lab
        ...
    jvs002/
        ...
    ...
    jvs100/
        ...
```

### 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当内容 |
|------|------|---------|
| リードエンジニア | 1名 | Julius環境構築、スクリプト実装、JVSデータフォーマット調査 |
| 音声処理エンジニア | 1名 | リサンプリング処理、音声フォーマット変換、Julius入力要件の検証 |
| QAエンジニア | 1名 | テスト作成、.labファイルの妥当性検証、エラーケースの網羅 |

**合計: 3名**

### 4. 提供範囲とテスト項目

#### 提供範囲
- `scripts/setup_julius.sh`: Julius segmentation-kitセットアップスクリプト
- `scripts/prepare_julius_input.py`: JVS→Julius入力変換スクリプト
- `scripts/run_julius_alignment.sh`: Juliusアライメント実行スクリプト
- `data/julius_alignment/`: 全100話者の`.lab`ファイル出力

#### ユニットテスト

**ファイル**: `tests/test_prepare_julius_input.py`

| テスト名 | 検証内容 |
|---------|---------|
| `test_read_transcripts_parses_jvs_format` | JVSのtranscripts_utf8.txtパースが正しくutt_id:text形式を読み取ること |
| `test_read_transcripts_handles_empty_lines` | 空行やコメント行を正しくスキップすること |
| `test_kanji_to_hiragana_basic` | 基本的な漢字→ひらがな変換が正しいこと（例: "今日は" → "きょうわ"） |
| `test_kanji_to_hiragana_katakana_conversion` | pyopenjtalkのカタカナ出力がひらがなに正しく変換されること |
| `test_resample_to_16k_output_sample_rate` | 出力WAVが16kHzであること |
| `test_resample_to_16k_mono` | ステレオ入力がモノラルに変換されること |
| `test_resample_to_16k_preserves_duration` | リサンプリング前後で音声長が保たれること（許容誤差: 10ms以内） |
| `test_jvs_all_utterances_text_conversion_success` | JVS全発話のテキスト変換（漢字かな混じり→ひらがな/カタカナ）が成功率100%であること |

#### 統合テスト

**ファイル**: `tests/test_julius_alignment_e2e.py`

| テスト名 | 検証内容 |
|---------|---------|
| `test_lab_file_exists_for_all_utterances` | 全発話に対して`.lab`ファイルが存在すること |
| `test_lab_file_format_is_valid_htk` | `.lab`ファイルがHTK形式（3列: 開始時刻 終了時刻 音素）であること |
| `test_lab_timestamps_are_monotonic` | 各`.lab`ファイル内の時刻が単調増加であること |
| `test_lab_timestamps_cover_audio_duration` | `.lab`の最終時刻が音声ファイルの長さと概ね一致すること（許容誤差: 50ms） |
| `test_lab_phonemes_are_nonempty` | 各セグメントの音素ラベルが空でないこと |
| `test_alignment_success_rate_above_threshold` | アライメント成功率が99%以上であること |
| `test_segmentation_kit_interface_confirmed` | segmentation-kitのREADMEを確認し、実際のインターフェース（ディレクトリ方式 or 引数方式）が確定していること |

### 5. 懸念事項とレビュー項目

#### 懸念事項

| 懸念 | 影響度 | 対策 |
|------|-------|------|
| Julius segmentation-kitのバージョン差異 | 高 | READMEに動作確認済みバージョンを記載。Perlスクリプトのインターフェースが変更されている可能性がある |
| JVSの一部話者で録音品質が低い | 中 | アライメント失敗率を話者ごとに計測し、閾値以上の話者を報告する |
| pyopenjtalkのひらがな変換精度 | 中 | JVSにひらがな転記が含まれている場合はそちらを優先使用する |
| 16kHzリサンプリングでのエイリアシング | 低 | torchaudioのResampleはデフォルトでローパスフィルタ適用 |
| segmentation-kitの音響モデルがJVSの録音条件に合わない | 中 | アライメント結果の外れ値検出を品質検証（T-M1-04）で実施 |

#### コードレビュー項目

- [ ] `setup_julius.sh`が冪等であること（再実行で上書きしない）
- [ ] `prepare_julius_input.py`のリサンプリングでアンチエイリアシングフィルタが適用されていること
- [ ] ひらがな変換で句読点・記号の扱いが正しいこと（Juliusの入力要件に準拠）
- [ ] `.lab`出力パスの命名規則がJVSの話者ID/発話IDと対応していること
- [ ] エラー処理: 個別発話の失敗が全体のパイプラインを停止しないこと
- [ ] 並列処理のワーカー数がCPUコア数に応じて設定可能であること

### 6. 一から作り直すとしたら

**Juliusの代わりにwhisper-alignを使う選択肢**を再検討する。whisper-alignは多言語対応で音響モデルの品質が高いが、(a) 日本語TTSでの検証事例が少ない、(b) JATTSでのJulius実証済みという安心感がない、(c) 音素セットが異なりマッピングが複雑になる可能性がある。現時点ではJuliusが最適だが、Juliusの環境構築に深刻な問題が発生した場合のフォールバックとして、MFA日本語モデル（v2.0.1a）も準備しておく。

**JVSのデータ前処理をDockerコンテナ化**する。Julius segmentation-kitはPerl依存があり、環境差異によるトラブルが予想される。Dockerfileで再現可能な環境を保証できれば、チーム全体の作業効率が上がる。

**`prepare_julius_input.py`と`run_julius_alignment.sh`を1つのPythonスクリプトに統合**する。現在の設計ではBashスクリプトとPythonスクリプトが混在しているが、`subprocess`でJuliusを呼び出すPython統合スクリプトにすれば、エラーハンドリングとロギングが統一される。進捗表示（tqdm）、並列処理（ProcessPoolExecutor）、リトライロジックもPython側で制御可能になる。

### 7. 後続タスクへの連絡事項

**T-M1-02（音素マッピング）への連絡**:
- `.lab`ファイルに出現するJulius音素の完全リストを提供する。`scripts/prepare_julius_input.py`の実行ログまたは別途集計スクリプトで抽出可能
- Julius segmentation-kitのバージョンと使用した音響モデルを記録する（音素セットはモデルに依存）
- silB/silE（文頭/文末無音）の扱いをT-M1-02に申し送る

**T-M1-03（duration変換）への連絡**:
- `.lab`ファイルの時間単位（100ns = 10^-7秒）を明記する
- 出力ディレクトリ構造（`data/julius_alignment/{spk_name}/{utt_id}.lab`）を確定する

**M2（PrecomputedDataModule対応）への連絡**:
- `.lab`ファイルのパスとファイル名の命名規則を共有する。`precompute_dataset.py`で`.pt`に埋め込む際のキーとして使用

---

## T-M1-02: 音素マッピングテーブル作成（Julius → pyopenjtalk 55シンボル）

### 1. タスク目的とゴール

**目的**: Julius forced alignerの出力音素セットを、Matcha-TTSの日本語学習で使用するpyopenjtalk 55シンボル体系に正確にマッピングするPythonモジュールを作成する。

**ゴール**:
- Julius音素セットとpyopenjtalk 55シンボルの完全な対応表を作成する
- マッピング関数を`matcha/text/julius_to_pyopenjtalk.py`として実装する
- JVSコーパス全発話の`.lab`ファイルに対してマッピングカバレッジ100%を達成する

**なぜ必要か**: Juliusとpyopenjtalkは異なる音素セットを使用している。Matcha-TTSの学習では`japanese_cleaners`（pyopenjtalk経由）で生成された55シンボル音素列を使用するため、Juliusの`.lab`出力をこの体系に変換しないとduration配列とテキストシーケンスの対応が取れない。

### 2. 実装する内容の詳細

#### 2.1 音素セットの差異分析

**pyopenjtalk 55シンボル**（`matcha/text/symbols.py`の`symbols_ja`から）:

```python
# Index 0: "~" (pad)
# Index 1-7: "^", "$", "?", "_", "#", "[", "]" (韻律記号)
# Index 8-54: 音素47種
# "A","E","I","N","O","U" (無声化母音+撥音)
# "a","b","by","ch","cl","d","dy","e","f","fy"
# "g","gw","gy","h","hy","i","j","k","kw","ky"
# "m","my","n","ny","o","p","py"
# "r","ry","s","sh","t","ts","ty"
# "u","v","w","y","z"
# "pau","sil"
```

**Julius segmentation-kitの音素セット**（日本語トライフォンHMM、一般的な構成）:

```
母音: a, i, u, e, o
子音: k, s, t, n, h, m, y, r, w, g, z, d, b, p
拗音: ky, sh, ch, ts, ty, ny, hy, ry, gy, by, py, my, dy, fy
特殊: N (撥音), cl (促音), pau (ポーズ)
無音: silB (文頭無音), silE (文末無音), sp (短ポーズ)
```

#### 2.2 マッピングテーブル

**作成ファイル**: `matcha/text/julius_to_pyopenjtalk.py`

```python
"""Julius音素セットからpyopenjtalk 55シンボルへのマッピング。

Julius segmentation-kitが出力する音素ラベルを、
matcha/text/symbols.pyで定義されたsymbols_ja（55シンボル）に変換する。

設計方針:
- 1対1マッピングが存在する音素はそのまま変換
- silB/silE → sil（文頭/文末の無音をpyopenjtalkの"sil"に統一）
- sp → pau（短ポーズをpyopenjtalkの"pau"に統一）
- 韻律記号（^, $, ?, _, #, [, ]）はJuliusでは生成されない
  → T-M1-03でpyopenjtalk音素列との位置合わせ時に挿入
- 無声化母音（A, I, U, E, O）はJuliusでは区別されない
  → 通常母音にマッピングし、pyopenjtalk側の無声化判定と照合
"""

from matcha.text.symbols import symbols_ja

# pyopenjtalk 55シンボルのセット（検証用）
_VALID_PYOPENJTALK_SYMBOLS = set(symbols_ja)

# Julius音素 → pyopenjtalk音素のマッピング
# キー: Juliusの.labファイルに出現する音素ラベル
# 値: pyopenjtalk 55シンボルの中の対応する音素
JULIUS_TO_PYOPENJTALK: dict[str, str] = {
    # --- 母音 ---
    "a": "a",
    "i": "i",
    "u": "u",
    "e": "e",
    "o": "o",
    # --- 基本子音 ---
    "k": "k",
    "s": "s",
    "t": "t",
    "n": "n",
    "h": "h",
    "m": "m",
    "y": "y",
    "r": "r",
    "w": "w",
    "g": "g",
    "z": "z",
    "d": "d",
    "b": "b",
    "p": "p",
    "f": "f",
    "v": "v",
    # --- 拗音・複合子音 ---
    "ky": "ky",
    "sh": "sh",
    "ch": "ch",
    "ts": "ts",
    "ty": "ty",
    "ny": "ny",
    "hy": "hy",
    "ry": "ry",
    "gy": "gy",
    "by": "by",
    "py": "py",
    "my": "my",
    "dy": "dy",
    "fy": "fy",
    "j": "j",
    "kw": "kw",
    "gw": "gw",
    # --- 特殊音素 ---
    "N": "N",       # 撥音
    "cl": "cl",     # 促音
    "pau": "pau",   # ポーズ
    # --- 無音（Julius固有） ---
    "silB": "sil",  # 文頭無音 → pyopenjtalkの"sil"
    "silE": "sil",  # 文末無音 → pyopenjtalkの"sil"
    "sp": "pau",    # 短ポーズ → pyopenjtalkの"pau"
    "sil": "sil",   # 汎用無音
}

# 韻律記号（pyopenjtalkには存在するがJuliusにはない）
# T-M1-03でpyopenjtalk音素列との照合時に使用
PROSODY_SYMBOLS = {"^", "$", "?", "_", "#", "[", "]"}


def map_julius_phoneme(julius_phoneme: str) -> str:
    """Julius音素1つをpyopenjtalk音素に変換する。

    Args:
        julius_phoneme: Juliusの.labファイルから読み取った音素ラベル

    Returns:
        pyopenjtalk 55シンボルの中の対応する音素

    Raises:
        KeyError: 未知のJulius音素が入力された場合
    """
    if julius_phoneme not in JULIUS_TO_PYOPENJTALK:
        raise KeyError(
            f"Unknown Julius phoneme: '{julius_phoneme}'. "
            f"Known phonemes: {sorted(JULIUS_TO_PYOPENJTALK.keys())}"
        )
    mapped = JULIUS_TO_PYOPENJTALK[julius_phoneme]
    assert mapped in _VALID_PYOPENJTALK_SYMBOLS, (
        f"Mapped phoneme '{mapped}' not in pyopenjtalk symbols"
    )
    return mapped


def map_julius_sequence(julius_phonemes: list[str]) -> list[str]:
    """Julius音素列をpyopenjtalk音素列に変換する。

    Args:
        julius_phonemes: Juliusの.labファイルから読み取った音素ラベルのリスト

    Returns:
        pyopenjtalk 55シンボルに変換された音素リスト
    """
    return [map_julius_phoneme(ph) for ph in julius_phonemes]


def get_unmapped_phonemes(julius_phonemes: list[str]) -> set[str]:
    """マッピングされていないJulius音素を返す（デバッグ用）。"""
    return {ph for ph in julius_phonemes if ph not in JULIUS_TO_PYOPENJTALK}
```

#### 2.3 マッピングカバレッジ検証スクリプト

**作成ファイル**: `scripts/validate_julius_mapping.py`

```python
"""JVSの全.labファイルに対してJulius→pyopenjtalkマッピングのカバレッジを検証する。

出力:
- 全.labファイルに出現するJulius音素の完全リスト
- 各音素の出現頻度
- 未マッピング音素のリスト（あれば）
"""

import argparse
from collections import Counter
from pathlib import Path

from matcha.text.julius_to_pyopenjtalk import JULIUS_TO_PYOPENJTALK


def parse_lab_file(lab_path: Path) -> list[str]:
    """HTK形式の.labファイルを読み込み、音素リストを返す。"""
    phonemes = []
    with open(lab_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                phonemes.append(parts[2])
    return phonemes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lab-dir", type=str, required=True)
    args = parser.parse_args()

    lab_dir = Path(args.lab_dir)
    all_phonemes = Counter()
    unmapped = set()
    total_files = 0

    for lab_path in sorted(lab_dir.rglob("*.lab")):
        phonemes = parse_lab_file(lab_path)
        all_phonemes.update(phonemes)
        total_files += 1
        for ph in phonemes:
            if ph not in JULIUS_TO_PYOPENJTALK:
                unmapped.add(ph)

    print(f"Total .lab files: {total_files}")
    print(f"Unique phonemes: {len(all_phonemes)}")
    print(f"\nPhoneme frequencies:")
    for ph, count in all_phonemes.most_common():
        mapped = JULIUS_TO_PYOPENJTALK.get(ph, "*** UNMAPPED ***")
        print(f"  {ph:8s} -> {mapped:8s}  (count: {count})")

    if unmapped:
        print(f"\nWARNING: {len(unmapped)} unmapped phonemes: {sorted(unmapped)}")
    else:
        print(f"\nAll phonemes are mapped. Coverage: 100%")


if __name__ == "__main__":
    main()
```

#### 2.4 無声化母音の取り扱い

pyopenjtalk 55シンボルには無声化母音（A, I, U, E, O）が含まれるが、Juliusは通常母音と無声化母音を区別しない。この差異は以下のように処理する:

1. Julius `.lab` からは通常母音（a, i, u, e, o）が出力される
2. pyopenjtalk（`japanese_cleaners`）の出力テキスト列で無声化母音が使用されている箇所は、T-M1-03でduration配列を構築する際に位置合わせで対応する
3. マッピングモジュール自体は1対1変換のみを担当し、無声化の判定はT-M1-03に委譲する

### 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当内容 |
|------|------|---------|
| 言語処理エンジニア | 1名 | Julius/pyopenjtalk音素セット調査、マッピングテーブル作成、エッジケース対応 |
| QAエンジニア | 1名 | テスト作成、JVS全.labファイルでのカバレッジ検証 |

**合計: 2名**

### 4. 提供範囲とテスト項目

#### 提供範囲
- `matcha/text/julius_to_pyopenjtalk.py`: マッピングモジュール
- `scripts/validate_julius_mapping.py`: カバレッジ検証スクリプト

#### ユニットテスト

**ファイル**: `tests/test_julius_to_pyopenjtalk.py`

| テスト名 | 検証内容 |
|---------|---------|
| `test_map_basic_vowels` | a,i,u,e,o が正しくマッピングされること |
| `test_map_basic_consonants` | k,s,t,n,h,m,y,r,w,g,z,d,b,p が正しくマッピングされること |
| `test_map_compound_consonants` | ky,sh,ch,ts,ty,ny,hy,ry,gy,by,py,my,dy,fy が正しくマッピングされること |
| `test_map_special_phonemes` | N（撥音）,cl（促音）,pau（ポーズ）が正しくマッピングされること |
| `test_map_silence_silB_to_sil` | silB（文頭無音）がsilにマッピングされること |
| `test_map_silence_silE_to_sil` | silE（文末無音）がsilにマッピングされること |
| `test_map_sp_to_pau` | sp（短ポーズ）がpauにマッピングされること |
| `test_map_unknown_phoneme_raises_keyerror` | 未知音素でKeyErrorが発生すること |
| `test_map_sequence_basic` | `["silB", "k", "o", "N", "n", "i", "ch", "i", "w", "a", "silE"]` → `["sil", "k", "o", "N", "n", "i", "ch", "i", "w", "a", "sil"]` |
| `test_all_mapped_values_are_valid_symbols` | JULIUS_TO_PYOPENJTALKの全バリューがsymbols_jaに含まれること |
| `test_get_unmapped_phonemes_empty_for_known` | 既知音素のみのリストで空集合が返ること |
| `test_get_unmapped_phonemes_detects_unknown` | 未知音素が正しく検出されること |
| `test_prosody_symbols_not_in_julius_mapping` | 韻律記号（^,$,?,_,#,[,]）がJULIUS_TO_PYOPENJTALKのキーに含まれないこと |

#### 統合テスト

**ファイル**: `tests/test_julius_mapping_coverage.py`（T-M1-01の.labファイル出力後に実行）

| テスト名 | 検証内容 |
|---------|---------|
| `test_all_lab_phonemes_are_mapped` | JVS全.labファイルの全音素がマッピングされること |
| `test_no_unknown_phonemes_in_corpus` | 未マッピング音素がゼロであること |

### 5. 懸念事項とレビュー項目

#### 懸念事項

| 懸念 | 影響度 | 対策 |
|------|-------|------|
| Julius音素セットがバージョンや音響モデルにより異なる可能性 | 高 | T-M1-01の出力から実際の音素リストを抽出し、それに基づいてマッピングを確定する |
| 無声化母音（A,I,U,E,O）のJuliusでの扱い | 中 | Juliusは通常区別しないが、特定の音響モデルでは区別する場合がある。実データで確認 |
| JVS固有の発音バリエーション（方言、個人差） | 低 | Juliusのforced alignmentは音響モデルに基づくためテキスト側では対応不要 |
| gw, kw音素がJuliusに存在するか不明 | 中 | 実データで確認。存在しない場合はマッピングテーブルから除外しデッドコードにならないようにする |

#### コードレビュー項目

- [ ] `JULIUS_TO_PYOPENJTALK`辞書の全バリューが`symbols_ja`に存在すること
- [ ] 逆マッピング（pyopenjtalk→Julius）が必要になった場合に拡張可能な構造であること
- [ ] `map_julius_phoneme`のエラーメッセージが十分に情報を含むこと
- [ ] docstringがJuliusのバージョン/音響モデルの前提条件を明記していること
- [ ] 無声化母音の扱いについてのコメントが十分であること

### 6. 一から作り直すとしたら

**双方向マッピングテーブル**を最初から設計する。現在の設計はJulius→pyopenjtalkの一方向だが、デバッグやテストで逆方向が必要になる場面が多い。マッピングを`(julius, pyopenjtalk)`のタプルリストとして定義し、両方向の辞書を自動生成する。

**マッピングの不確実性を型で表現**する。1対1マッピングが確実な音素と、文脈依存で曖昧な音素（無声化母音など）を型レベルで区別する。例えば`MappingResult`を`Exact | Ambiguous`のunion型で定義し、下流処理で曖昧なマッピングに対するフォールバック処理を強制する。

**マッピングテーブルをJSONではなくPythonの型付き辞書で持つ判断は正しい**。JSONにすると実行時の型チェックが効かず、テストでカバーすべき範囲が広がる。Python辞書なら静的解析とテストの両方で品質を担保できる。

### 7. 後続タスクへの連絡事項

**T-M1-03（duration変換）への連絡**:
- 韻律記号（`^, $, ?, _, #, [, ]`）はJuliusでは生成されない。pyopenjtalkの`japanese_cleaners`出力に含まれるこれらの記号は、T-M1-03でのduration配列構築時に「duration=0フレーム」として挿入する必要がある
- 無声化母音（A, I, U, E, O）はJuliusでは通常母音（a, i, u, e, o）として出力される。pyopenjtalkの音素列との位置合わせでは、大文字/小文字の違いを考慮した照合ロジックが必要
- `map_julius_sequence`関数は韻律記号を含まない音素列を返す。韻律記号の挿入はT-M1-03の責務

**T-M1-04（品質検証）への連絡**:
- `get_unmapped_phonemes`関数を品質検証スクリプトで使用可能
- JVS全体でのマッピングカバレッジ統計は`scripts/validate_julius_mapping.py`で取得可能

---

## T-M1-03: .lab → durationフレーム配列変換 + blank intersperse

### 1. タスク目的とゴール

**目的**: Juliusの`.lab`ファイル（音素の時間アライメント）を、Matcha-TTSの学習で使用するdurationフレーム配列（`IntTensor`）に変換する。blank intersperse後の音素列と同一長の配列を生成し、各要素が対応する音素のメルフレーム数を表す。

**ゴール**:
- `.lab`ファイルをパースし、時刻→フレーム数に変換する機能を実装する
- pyopenjtalkの`japanese_cleaners`が生成する音素列（韻律記号含む）とJuliusアライメント結果を位置合わせする
- blank intersperse（`[0, p1, 0, p2, ..., 0]`）後のシーケンス長に一致するduration配列を生成する
- 全JVS発話についてduration配列を生成し、duration合計がメルフレーム数と一致することを検証する

**なぜ必要か**: Matcha-TTSの`use_precomputed_durations=True`パスは、`generate_path(durations.squeeze(1), attn_mask.squeeze(1))`でアライメント行列を直接構築する（`matcha/models/matcha_tts.py` L193-194）。このduration配列はblank intersperse済みのテキストシーケンス長と完全に一致する必要がある。

### 2. 実装する内容の詳細

#### 2.1 核となるフレーム変換ロジック

**時刻→フレーム数変換（絶対時刻ベース）**:
```
start_frame = round(start_time * sample_rate / hop_length)
end_frame = round(end_time * sample_rate / hop_length)
duration = end_frame - start_frame
```
- `sample_rate = 22050` (Matcha-TTSのメルスペクトログラム)
- `hop_length = 256`
- Juliusの時刻単位: 100ns (10^-7秒) → 秒に変換: `time_sec = time_100ns / 10_000_000`
- **注意**: Julius入力は16kHzだが、フレーム変換はMatcha-TTSのメル計算パラメータ（22050Hz / hop_length=256）で行う
- **重要**: 相対計算（`frames = round((end - start) * sr / hop)`）ではなく絶対時刻ベースの計算を採用する。相対計算では各セグメントの`round()`で累積丸め誤差が発生するが、絶対時刻ベースでは各セグメントの開始/終了フレームを独立に計算するため、全セグメントのduration合計が自動的にメルフレーム数に近づく

#### 2.2 音素列の位置合わせ問題

pyopenjtalkの`japanese_cleaners`が生成する音素列（韻律記号含む）とJuliusアライメントの音素列には構造的な差異がある。

**pyopenjtalkの出力例**（"こんにちは"）:
```
^ k o [N n i ch i w a $
```
→ 韻律記号を含む: `^`, `[`, `$`

**Juliusの.lab出力例**:
```
silB k o N n i ch i w a silE
```
→ 韻律記号なし、silB/silEが先頭/末尾

**位置合わせの方針**:
1. Juliusの`silB`/`silE`はpyopenjtalkの`^`（文頭=sil）/`$`（文末=sil）に対応
2. pyopenjtalkの韻律記号（`#`, `[`, `]`, `?`, `_`）にはduration=0を割り当てる
3. Juliusの音素列（silB/silE除去、pyopenjtalk音素にマッピング済み）と、pyopenjtalkの音素列（韻律記号除去）を照合し、1対1の対応を確認

#### 2.3 blank intersperseとduration配列の対応

`intersperse(phoneme_ids, 0)`後のシーケンス:
```
元: [p1, p2, p3, ..., pN]  (長さN)
後: [0, p1, 0, p2, 0, p3, 0, ..., 0, pN, 0]  (長さ2N+1)
```

duration配列も同じ長さ `2N+1` が必要:
```
durations: [d_blank0, d_p1, d_blank1, d_p2, d_blank2, ..., d_blankN-1, d_pN, d_blankN]
```

**blankのduration**: 0フレーム（blankは仮想トークンであり、実際の音声区間を持たない）

#### 2.4 メインスクリプト

**作成ファイル**: `scripts/convert_julius_to_durations.py`

```python
"""Julius .labファイルからdurationフレーム配列を生成する。

各JVS発話について:
1. .labファイルの時刻情報を22050Hz/hop_length=256のフレーム数に変換
2. Julius音素をpyopenjtalkシンボルにマッピング
3. pyopenjtalkのjapanese_cleaners出力と位置合わせ
4. blank intersperse後のシーケンス長に合わせたduration配列を生成
5. duration配列を.npyファイルとして保存

Usage:
    python scripts/convert_julius_to_durations.py \
        --lab-dir data/julius_alignment \
        --filelist data/jvs/train.txt \
        --output-dir data/jvs_durations \
        --num-workers 8
"""

import argparse
import logging
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

from matcha.text import text_to_sequence
from matcha.text.julius_to_pyopenjtalk import (
    PROSODY_SYMBOLS,
    map_julius_phoneme,
)
from matcha.utils.utils import intersperse

SAMPLE_RATE = 22050
HOP_LENGTH = 256
# Julius .lab時間単位: 100ns
JULIUS_TIME_UNIT = 10_000_000  # 1秒 = 10^7 (100ns単位)

log = logging.getLogger(__name__)


def parse_lab_file(lab_path: Path) -> list[tuple[float, float, str]]:
    """HTK形式の.labファイルをパースする。

    Returns:
        [(start_sec, end_sec, phoneme), ...] のリスト
    """
    segments = []
    with open(lab_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            start = int(parts[0]) / JULIUS_TIME_UNIT
            end = int(parts[1]) / JULIUS_TIME_UNIT
            phoneme = parts[2]
            segments.append((start, end, phoneme))
    return segments


def time_to_frames(start_sec: float, end_sec: float) -> int:
    """時刻区間をメルフレーム数に変換する（絶対時刻ベース）。

    累積丸め誤差を回避するため、開始/終了フレームを絶対時刻から独立に計算し、
    その差分をdurationとする。
    start_frame = round(start_time * sample_rate / hop_length)
    end_frame = round(end_time * sample_rate / hop_length)
    duration = end_frame - start_frame
    """
    start_frame = round(start_sec * SAMPLE_RATE / HOP_LENGTH)
    end_frame = round(end_sec * SAMPLE_RATE / HOP_LENGTH)
    return max(0, end_frame - start_frame)


def align_julius_with_pyopenjtalk(
    julius_phonemes: list[str],
    pyopenjtalk_phonemes: list[str],
    julius_durations: list[int],
) -> list[int]:
    """Julius音素列とpyopenjtalk音素列を位置合わせし、
    pyopenjtalk音素列に対応するduration配列を返す。

    pyopenjtalkの韻律記号（^, $, ?, _, #, [, ]）にはduration=0を割り当てる。
    Julius側のsilB/silEはpyopenjtalkのsil（^, $に対応）のdurationとして使用する。

    Args:
        julius_phonemes: Julius .labから取得しmap_julius_phoneme済みの音素列
        pyopenjtalk_phonemes: japanese_cleaners出力の音素列（スペース分割）
        julius_durations: Julius各音素のフレーム数

    Returns:
        pyopenjtalk音素列の各音素に対応するduration配列
    """
    result = []
    j_idx = 0  # Juliusの音素インデックス

    for py_ph in pyopenjtalk_phonemes:
        if py_ph in PROSODY_SYMBOLS:
            # 韻律記号にはduration=0
            # ただし ^(文頭sil)と$(文末sil)はJuliusのsilB/silEに対応
            if py_ph == "^" and j_idx < len(julius_phonemes) and julius_phonemes[j_idx] == "sil":
                result.append(julius_durations[j_idx])
                j_idx += 1
            elif py_ph == "$" and j_idx < len(julius_phonemes) and julius_phonemes[j_idx] == "sil":
                result.append(julius_durations[j_idx])
                j_idx += 1
            elif py_ph == "_" and j_idx < len(julius_phonemes) and julius_phonemes[j_idx] == "pau":
                result.append(julius_durations[j_idx])
                j_idx += 1
            else:
                result.append(0)
        else:
            # 通常音素: Julius側と照合
            if j_idx >= len(julius_phonemes):
                log.warning(
                    "Julius phonemes exhausted at pyopenjtalk index. "
                    "Assigning duration=1 for '%s'", py_ph
                )
                result.append(1)
                continue

            j_ph = julius_phonemes[j_idx]
            # 無声化母音の照合: pyopenjtalkの A,I,U,E,O ↔ Juliusの a,i,u,e,o
            py_ph_lower = py_ph.lower() if py_ph in ("A", "I", "U", "E", "O") else py_ph
            if j_ph == py_ph_lower or j_ph == py_ph:
                result.append(julius_durations[j_idx])
                j_idx += 1
            elif py_ph in ("A", "I", "U", "E", "O"):
                # 無声化母音マッチング: Juliusが無声化母音セグメントを
                # 省略した可能性がある場合、duration=0を割り当てる
                # （Juliusは無声化母音を極端に短く出力するか省略することがある）
                log.warning(
                    "Devoiced vowel '%s' not found in Julius output at index %d "
                    "(julius='%s'). Assigning duration=0", py_ph, j_idx, j_ph
                )
                result.append(0)
                # j_idxは進めない（Juliusがこの音素を出力していないため）
            else:
                log.warning(
                    "Phoneme mismatch: julius='%s' vs pyopenjtalk='%s'. "
                    "Assigning duration=1", j_ph, py_ph
                )
                result.append(1)

    return result


def align_julius_with_pyopenjtalk_dtw(
    julius_phonemes: list[str],
    pyopenjtalk_phonemes: list[str],
    julius_durations: list[int],
) -> list[int]:
    """DTWフォールバック: 逐次照合で不一致率が高い場合に使用する。

    音素シーケンス間のDTW（Dynamic Time Warping）アライメントにより、
    音素の挿入・削除・置換を許容した柔軟な位置合わせを行う。

    逐次照合（align_julius_with_pyopenjtalk）で不一致率が5%を超える場合に
    この関数にフォールバックする。
    """
    # pyopenjtalkから韻律記号を除去して実音素列を取得
    py_real = [(i, ph) for i, ph in enumerate(pyopenjtalk_phonemes)
               if ph not in PROSODY_SYMBOLS]

    # DTWコスト行列を構築（一致=0、無声化母音照合=0、不一致=1）
    n, m = len(julius_phonemes), len(py_real)
    cost = np.full((n + 1, m + 1), float("inf"))
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            j_ph = julius_phonemes[i - 1]
            py_ph = py_real[j - 1][1]
            py_lower = py_ph.lower() if py_ph in ("A", "I", "U", "E", "O") else py_ph
            match_cost = 0.0 if (j_ph == py_lower or j_ph == py_ph) else 1.0
            cost[i, j] = match_cost + min(
                cost[i - 1, j - 1],  # 対角: 1対1マッチ
                cost[i - 1, j] + 0.5,  # Julius側スキップ（削除）
                cost[i, j - 1] + 0.5,  # pyopenjtalk側スキップ（挿入）
            )

    # バックトラックでアライメントを復元
    alignment = {}  # py_real_index -> julius_index
    i, j = n, m
    while i > 0 and j > 0:
        j_ph = julius_phonemes[i - 1]
        py_ph = py_real[j - 1][1]
        py_lower = py_ph.lower() if py_ph in ("A", "I", "U", "E", "O") else py_ph
        match_cost = 0.0 if (j_ph == py_lower or j_ph == py_ph) else 1.0
        if cost[i, j] == match_cost + cost[i - 1, j - 1]:
            alignment[j - 1] = i - 1
            i -= 1
            j -= 1
        elif cost[i, j] == cost[i - 1, j] + 0.5:
            i -= 1
        else:
            j -= 1

    # pyopenjtalk全音素列にdurationを割り当て
    result = []
    real_idx = 0
    for py_ph in pyopenjtalk_phonemes:
        if py_ph in PROSODY_SYMBOLS:
            result.append(0)
        else:
            if real_idx in alignment:
                result.append(julius_durations[alignment[real_idx]])
            else:
                result.append(0)
            real_idx += 1

    return result


def build_duration_array_with_blanks(
    phoneme_durations: list[int],
    total_mel_frames: int,
) -> np.ndarray:
    """音素durationにblank intersperse対応のduration配列を構築する。

    blank intersperse後: [0, p1, 0, p2, ..., 0, pN, 0]
    duration: [0, d1, 0, d2, ..., 0, dN, 0]

    blankのdurationは0。音素durationの合計がtotal_mel_framesに一致するよう
    最終音素のdurationで端数調整する。

    Args:
        phoneme_durations: 各音素のフレーム数（韻律記号含む）
        total_mel_frames: メルスペクトログラムの総フレーム数

    Returns:
        blank intersperse後のduration配列 (IntTensor相当のndarray)
    """
    n_phonemes = len(phoneme_durations)
    # intersperse後の長さ: 2*n + 1
    duration_array = np.zeros(2 * n_phonemes + 1, dtype=np.int64)

    # 音素位置（奇数インデックス）にdurationを設定
    for i, dur in enumerate(phoneme_durations):
        duration_array[2 * i + 1] = dur

    # 合計フレーム数の端数調整
    current_total = duration_array.sum()
    diff = total_mel_frames - current_total

    if diff != 0:
        # 最後の非ゼロ音素のdurationで調整
        for idx in range(len(duration_array) - 1, -1, -1):
            if duration_array[idx] > 0:
                duration_array[idx] = max(1, duration_array[idx] + diff)
                break
        else:
            # 全て0の場合（異常ケース）: 最後のblankに割り当て
            duration_array[-1] = total_mel_frames

    return duration_array


def process_single_utterance(
    lab_path: Path,
    text: str,
    total_mel_frames: int,
    output_path: Path,
) -> tuple[bool, str]:
    """1発話分のduration配列を生成して保存する。

    Returns:
        (success, message) タプル
    """
    try:
        # 1. .labファイルをパース
        segments = parse_lab_file(lab_path)
        if not segments:
            return False, f"Empty lab file: {lab_path}"

        # 2. Julius音素→pyopenjtalkマッピング + フレーム変換
        julius_phonemes = []
        julius_durations = []
        for start, end, ph in segments:
            mapped_ph = map_julius_phoneme(ph)
            frames = time_to_frames(start, end)
            julius_phonemes.append(mapped_ph)
            julius_durations.append(frames)

        # 3. pyopenjtalkのjapanese_cleaners出力を取得
        text_sequence, cleaned_text = text_to_sequence(
            text, ["japanese_cleaners"], language="ja"
        )
        pyopenjtalk_phonemes = cleaned_text.split()

        # 4. Julius音素列とpyopenjtalk音素列の位置合わせ
        aligned_durations = align_julius_with_pyopenjtalk(
            julius_phonemes, pyopenjtalk_phonemes, julius_durations
        )

        # 5. blank intersperse対応のduration配列を構築
        duration_array = build_duration_array_with_blanks(
            aligned_durations, total_mel_frames
        )

        # 6. intersperse後のテキスト長と一致するか検証
        text_interspersed = intersperse(text_sequence, 0)
        if len(duration_array) != len(text_interspersed):
            return False, (
                f"Length mismatch: duration={len(duration_array)} "
                f"vs text={len(text_interspersed)}"
            )

        # 7. 保存
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, duration_array)

        return True, "OK"

    except Exception as e:
        return False, f"Error: {e}"
```

#### 2.5 フレーム数端数調整の設計

メルスペクトログラムのフレーム数とduration合計に差異が生じる原因:
- Juliusの時間精度（10ms）とメルのhop_length（256/22050 = 11.6ms）の不一致
- `.lab`の最終セグメント終了時刻と実際の音声長の差異
- `round()`による累積丸め誤差

**調整方針**:
- duration合計 < メルフレーム数: 最後のsil音素（文末無音）のdurationを増加
- duration合計 > メルフレーム数: 最後のsil音素のdurationを減少（ただし最小1フレーム）
- 差異が大きい場合（+/-10フレーム以上）は警告を出力

#### 2.6 出力フォーマット

各発話に対して`.npy`ファイルを出力:
```
data/jvs_durations/
    jvs001_VOICEACTRESS100_001.npy  # shape: (2*N+1,), dtype: int64
    jvs001_VOICEACTRESS100_002.npy
    ...
```

ファイル名はprecompute_dataset.pyの命名規則（`{spk_name}_{utt_id}.npy`）に合わせる。

### 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当内容 |
|------|------|---------|
| リードエンジニア | 1名 | 位置合わせアルゴリズム設計・実装、フレーム変換ロジック |
| テキスト処理エンジニア | 1名 | pyopenjtalk音素列との整合性確認、韻律記号・無声化母音の処理 |
| QAエンジニア | 1名 | テスト作成、エッジケースの網羅、全発話の整合性検証 |

**合計: 3名**

### 4. 提供範囲とテスト項目

#### 提供範囲
- `scripts/convert_julius_to_durations.py`: メインの変換スクリプト
- `data/jvs_durations/`: 全発話のduration配列（`.npy`ファイル）

#### ユニットテスト

**ファイル**: `tests/test_convert_julius_to_durations.py`

| テスト名 | 検証内容 |
|---------|---------|
| `test_parse_lab_file_basic` | 3行の.labファイルが正しくパースされること |
| `test_parse_lab_file_empty` | 空の.labファイルで空リストが返ること |
| `test_time_to_frames_exact` | 1秒 → round(22050/256)=86フレーム |
| `test_time_to_frames_short_segment` | 10ms → round(0.01*22050/256)=1フレーム |
| `test_time_to_frames_zero_duration` | 0秒 → 0フレーム |
| `test_align_julius_with_pyopenjtalk_simple` | "こんにちは"のケースで正しく位置合わせされること |
| `test_align_julius_with_pyopenjtalk_with_prosody` | 韻律記号（^, [, ], $）にduration=0が割り当てられること |
| `test_align_julius_with_pyopenjtalk_with_pause` | `_`（pau）にJuliusのpaudurationが割り当てられること |
| `test_align_devoiced_vowels` | 無声化母音（A,I,U,E,O）がJuliusの通常母音と正しく照合されること |
| `test_prosody_bracket_between_phonemes` | `[` が2つの通常音素の間に挿入されるケース（例: `k o [N n i ch i w a`）でJuliusインデックスがずれないこと |
| `test_prosody_hash_at_accent_boundary` | `#` がアクセント句境界に出現するケースでduration=0が割り当てられ、前後の音素durationが正しいこと |
| `test_multiple_prosody_consecutive` | 韻律記号が連続するケース（例: `] #`）で各記号にduration=0が割り当てられ、Juliusインデックスが正しく維持されること |
| `test_prosody_symbols_duration_zero` | 全韻律記号(^,$,?,_,#,[,])のdurationが0であることの検証（`_`がpauに対応する場合を除く） |
| `test_align_devoiced_vowel_missing_in_julius` | Juliusが無声化母音セグメントを出力しない場合にduration=0が割り当てられること |
| `test_build_duration_array_with_blanks_length` | 音素数Nに対してduration配列長が2N+1であること |
| `test_build_duration_array_with_blanks_blank_positions` | 偶数インデックス（blank位置）のdurationが0であること |
| `test_build_duration_array_with_blanks_sum_matches_mel` | duration合計がtotal_mel_framesと一致すること |
| `test_build_duration_array_with_blanks_adjustment_positive` | duration合計 < mel_framesの場合、正しく端数調整されること |
| `test_build_duration_array_with_blanks_adjustment_negative` | duration合計 > mel_framesの場合、正しく端数調整されること |
| `test_duration_array_consistent_with_intersperse` | `intersperse(text_sequence, 0)`の長さとduration配列の長さが一致すること |

#### 統合テスト

**ファイル**: `tests/test_duration_conversion_e2e.py`

| テスト名 | 検証内容 |
|---------|---------|
| `test_all_npy_files_exist` | filelistの全発話に対して.npyが存在すること |
| `test_duration_sum_matches_mel_frames` | 全発話でduration合計がメルフレーム数と一致すること（許容: +/-1） |
| `test_duration_array_length_matches_text` | 全発話でduration配列長がintersperse後のテキスト長と一致すること |
| `test_no_negative_durations` | 全発話でduration値が非負であること |
| `test_phoneme_durations_are_reasonable` | 全音素のdurationが0-500フレーム範囲であること |
| `test_adjustment_diff_within_threshold` | 端数調整量が+/-5フレーム以内であること（99%のサンプル） |

### 5. 懸念事項とレビュー項目

#### 懸念事項

| 懸念 | 影響度 | 対策 |
|------|-------|------|
| pyopenjtalkの出力がテキストにより非決定的な場合がある | 高 | `text_to_sequence`の出力を.ptファイルの`cleaned_text`と照合し、一致を確認 |
| 韻律記号（^,$等）とJuliusのsilB/silEの対応が発話によっては1対1でない | 高 | 文頭`^`→silB、文末`$`→silEの固定対応を基本とし、例外を個別ログ出力 |
| 累積丸め誤差が一部の長い発話で大きくなる | 中 | 絶対時刻ベースのフレーム変換を初期実装で採用済み（セクション2.1参照）。各セグメントの開始/終了フレームを絶対時刻から独立に計算し、差分でdurationを求める |
| 無声化母音のマッチ失敗 | 中 | pyopenjtalkの無声化母音（大文字）をJuliusの通常母音（小文字）と照合する明示的ロジック |
| Juliusが無声化母音セグメントを極端に短く出力するか省略する可能性 | 中 | pyopenjtalkのA,I,U,E,OをJuliusの小文字母音と照合する際、Juliusが当該セグメントを出力しない場合はduration=0を割り当てる。不一致率5%超の場合、音素シーケンス間のDTWアライメントにフォールバック |
| 韻律記号の挿入位置がJuliusインデックスとの対応関係を破綻させるリスク | 高 | pyopenjtalkの韻律記号（^,$,?,_,#,[,]）は音素間に挿入されるため、逐次照合時にJulius側のインデックスを進めてはならない。特に`[`や`#`がアクセント句境界で連続出現する場合に注意。韻律記号スキップ後のJuliusインデックス整合性を全テストケースで検証する |
| JVS一部発話でpyopenjtalkとJuliusの音素数が一致しない | 高 | 不一致サンプルをエラーログに出力し、除外リストを生成。T-M1-04で分析 |

#### コードレビュー項目

- [ ] フレーム変換で絶対時刻ベースの計算（累積誤差回避）が実装されていること
- [ ] blank位置（偶数インデックス）のdurationが全て0であること
- [ ] 端数調整が最後のsil音素のみで行われ、他の音素のdurationを変更しないこと
- [ ] `text_to_sequence`のcleaner/languageパラメータが学習設定と一致すること（`["japanese_cleaners"]`, `language="ja"`）
- [ ] `.npy`ファイル名がprecompute_dataset.pyの命名規則と一致すること
- [ ] 並列処理でtext_to_sequenceのLRUキャッシュが安全であること（プロセス単位で独立）

### 6. 一から作り直すとしたら

**フレーム変換を相対ではなく絶対時刻ベースで行う** → **初期実装で採用済み**。各セグメントの開始/終了フレームを`start_frame = round(start * sr / hop)`、`end_frame = round(end * sr / hop)`で計算し、`duration = end_frame - start_frame`とする方式を、セクション2.1の`time_to_frames`関数に反映済み。これにより累積丸め誤差が回避され、全セグメントのduration合計が自動的にメルフレーム数に近づく。

**位置合わせをDTW（Dynamic Time Warping）で行う**。現在の逐次照合ロジックは、pyopenjtalkとJuliusの音素列がほぼ同一構造であることを前提としている。実際にはpyopenjtalkの音素分割とJuliusの音素分割が微妙に異なるケース（例: 「っ」の扱い、長母音の分割）がありうる。DTWを使えばこうした差異を吸収できるが、計算コストとコード複雑性が増す。10,000発話程度なら逐次照合で問題ないが、不一致率が5%を超える場合はDTWへの切り替えを検討する。

**`.npy`ではなくdurationを`.pt`ファイルに直接埋め込む設計**にする。現在の設計では`.npy`と`.pt`が別ファイルで管理されるが、M2のPrecomputedDataModule対応で`.pt`にdurationキーを追加するなら、最初から`precompute_dataset.py`を拡張してdurationも含めた`.pt`を生成するほうが、ファイル管理が単純になる。ただし、T-M1-03の段階では`.npy`として出力し、M2で`.pt`への統合を行う設計のほうが、マイルストーン間の依存を緩くできる。

### 7. 後続タスクへの連絡事項

**T-M1-04（品質検証）への連絡**:
- 各発話のduration配列とメルフレーム数の差異（端数調整量）を記録し、品質検証で分析可能にする
- pyopenjtalkとJuliusの音素列不一致が発生した発話リストを提供する
- duration=0の音素（韻律記号）の数と位置を品質検証で確認可能にする

**M2（PrecomputedDataModule対応）への連絡**:
- 出力フォーマット: `data/jvs_durations/{spk_name}_{utt_id}.npy`、shape: `(2*N+1,)`、dtype: `int64`
- ファイル名は`precompute_dataset.py`の出力`.pt`ファイル名と対応（拡張子のみ異なる）
- `.pt`ファイルへの統合方法: `torch.from_numpy(np.load(npy_path))` でテンソルに変換し、`"durations"` キーに追加
- blank位置（偶数インデックス）のdurationは全て0。これにより`TextMelBatchCollate`の`torch.zeros`初期化と整合する
- duration合計はメルフレーム数と一致（端数調整済み）。`matcha_tts.py`の`generate_path`が期待する条件を満たす

---

## T-M1-04: アライメント品質検証・統計分析

### 1. タスク目的とゴール

**目的**: Julius forced alignmentで生成したduration配列の品質を多角的に検証し、MAS退化問題が解消されていることを定量的に確認する。

**ゴール**:
- 退化アライメント率が0%であることを確認する（MASでは39-43%）
- 音素クラスごとのduration分布統計を生成し、言語学的に妥当であることを確認する
- MAS durationとの比較分析を行い、改善点を定量化する
- 品質に問題のある発話を特定し、除外リストを生成する（必要に応じて）

**なぜ必要か**: 外部アライナーを導入しても、アライメント品質が不十分であれば学習品質は改善しない。MAS退化の定量指標（CLAUDE.mdに記載）と同じ基準でJuliusアライメントを評価し、問題解消を数値で証明する必要がある。

### 2. 実装する内容の詳細

#### 2.1 品質検証指標の定義

CLAUDE.mdのMAS退化分析で使用された指標を踏襲し、Juliusアライメントに適用する:

| 指標 | MAS退化基準 | Juliusの目標値 |
|------|-----------|--------------|
| 退化サンプル率（音素の80%以上が1フレーム以下） | 39-43% | 0% |
| blank[0]の平均duration | 退化時:101, 正常時:4 | N/A（blank=0フレーム） |
| 1フレーム以下の音素の割合 | 62.9% | <10% |
| 2フレーム以下の音素の割合 | 73.8% | <20% |
| 音素の中央durationフレーム数 | 退化:1.0, 正常:2.0 | >=4.0（期待値） |

#### 2.2 メイン検証スクリプト

**作成ファイル**: `scripts/verify_alignment_quality.py`

```python
"""Juliusアライメントの品質検証・統計分析スクリプト。

MAS退化問題の解消を定量的に検証する。

Usage:
    python scripts/verify_alignment_quality.py \
        --duration-dir data/jvs_durations \
        --filelist data/jvs/train.txt \
        --output-report data/alignment_quality_report.json

出力:
    1. コンソールに統計サマリを表示
    2. JSONレポートファイル（詳細統計）
    3. 問題のある発話のリスト（あれば）
"""

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from matcha.text import text_to_sequence
from matcha.text.symbols import symbols_ja
from matcha.utils.utils import intersperse

log = logging.getLogger(__name__)


def is_degenerate(durations: np.ndarray, threshold_ratio: float = 0.8) -> bool:
    """退化アライメントかどうかを判定する。

    音素位置（奇数インデックス）のdurationの80%以上が1フレーム以下なら退化と判定。
    CLAUDE.mdのMAS退化基準と同一。
    """
    # blank位置（偶数インデックス）を除外
    phoneme_durations = durations[1::2]
    if len(phoneme_durations) == 0:
        return True
    n_short = np.sum(phoneme_durations <= 1)
    return (n_short / len(phoneme_durations)) >= threshold_ratio


def compute_phoneme_class_stats(
    durations: np.ndarray,
    phoneme_ids: list[int],
    id_to_symbol: dict[int, str],
) -> dict[str, list[int]]:
    """音素クラスごとのduration分布を集計する。

    Returns:
        {phoneme_symbol: [duration_frames, ...], ...}
    """
    stats = defaultdict(list)
    # intersperse後: [0, p1, 0, p2, ..., 0]
    for i in range(1, len(phoneme_ids), 2):  # 奇数インデックス=音素位置
        if i < len(durations):
            pid = phoneme_ids[i]
            symbol = id_to_symbol.get(pid, f"UNK_{pid}")
            stats[symbol].append(int(durations[i]))
    return dict(stats)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration-dir", type=str, required=True)
    parser.add_argument("--filelist", type=str, required=True)
    parser.add_argument("--output-report", type=str, default="data/alignment_quality_report.json")
    parser.add_argument("--mas-duration-dir", type=str, default=None,
                        help="MAS durationディレクトリ（比較分析用、省略可）")
    args = parser.parse_args()

    duration_dir = Path(args.duration_dir)
    id_to_symbol = {i: s for i, s in enumerate(symbols_ja)}

    # ファイルリスト読み込み
    with open(args.filelist, encoding="utf-8") as f:
        entries = [line.strip().split("|") for line in f if line.strip()]

    # --- 統計集計 ---
    total_samples = 0
    degenerate_count = 0
    all_phoneme_durations = []
    phoneme_class_durations = defaultdict(list)
    problematic_samples = []
    frame_diff_stats = []  # duration合計とメルフレーム数の差異

    for entry in tqdm(entries, desc="Verifying alignment quality"):
        wav_path, spk_str, text = entry[0], entry[1], entry[2]
        wav_p = Path(wav_path)
        spk_name = wav_p.parent.name
        utt_id = wav_p.stem

        npy_path = duration_dir / f"{spk_name}_{utt_id}.npy"
        if not npy_path.exists():
            problematic_samples.append({
                "utterance": f"{spk_name}_{utt_id}",
                "issue": "missing_duration_file",
            })
            continue

        durations = np.load(npy_path)
        total_samples += 1

        # テキスト処理（intersperse後のID列を取得）
        text_seq, _ = text_to_sequence(text, ["japanese_cleaners"], language="ja")
        text_interspersed = intersperse(text_seq, 0)

        # 長さ一致チェック
        if len(durations) != len(text_interspersed):
            problematic_samples.append({
                "utterance": f"{spk_name}_{utt_id}",
                "issue": f"length_mismatch: dur={len(durations)} text={len(text_interspersed)}",
            })
            continue

        # 退化判定
        if is_degenerate(durations):
            degenerate_count += 1
            problematic_samples.append({
                "utterance": f"{spk_name}_{utt_id}",
                "issue": "degenerate_alignment",
            })

        # 音素durationの集計（blank除外）
        phoneme_durs = durations[1::2]
        all_phoneme_durations.extend(phoneme_durs.tolist())

        # 音素クラスごとの集計
        cls_stats = compute_phoneme_class_stats(
            durations, text_interspersed, id_to_symbol
        )
        for sym, durs in cls_stats.items():
            phoneme_class_durations[sym].extend(durs)

    # --- 統計計算 ---
    all_durs = np.array(all_phoneme_durations)
    degenerate_rate = degenerate_count / total_samples if total_samples > 0 else 0

    report = {
        "total_samples": total_samples,
        "degenerate_count": degenerate_count,
        "degenerate_rate": f"{degenerate_rate:.4f}",
        "phoneme_duration_stats": {
            "mean": float(np.mean(all_durs)) if len(all_durs) > 0 else 0,
            "median": float(np.median(all_durs)) if len(all_durs) > 0 else 0,
            "std": float(np.std(all_durs)) if len(all_durs) > 0 else 0,
            "min": int(np.min(all_durs)) if len(all_durs) > 0 else 0,
            "max": int(np.max(all_durs)) if len(all_durs) > 0 else 0,
            "pct_le_1frame": float(np.mean(all_durs <= 1)) if len(all_durs) > 0 else 0,
            "pct_le_2frame": float(np.mean(all_durs <= 2)) if len(all_durs) > 0 else 0,
        },
        "phoneme_class_stats": {},
        "problematic_samples": problematic_samples[:100],  # 上位100件
        "problematic_count": len(problematic_samples),
    }

    # 音素クラスごとの統計
    for sym in sorted(phoneme_class_durations.keys()):
        durs = np.array(phoneme_class_durations[sym])
        report["phoneme_class_stats"][sym] = {
            "count": len(durs),
            "mean": round(float(np.mean(durs)), 2),
            "median": float(np.median(durs)),
            "std": round(float(np.std(durs)), 2),
        }

    # --- コンソール出力 ---
    print("=" * 60)
    print("Julius Alignment Quality Report")
    print("=" * 60)
    print(f"Total samples:        {total_samples}")
    print(f"Degenerate samples:   {degenerate_count} ({degenerate_rate:.2%})")
    print(f"Problematic samples:  {len(problematic_samples)}")
    print(f"\nPhoneme duration statistics (blank excluded):")
    print(f"  Mean:     {report['phoneme_duration_stats']['mean']:.2f} frames")
    print(f"  Median:   {report['phoneme_duration_stats']['median']:.1f} frames")
    print(f"  Std:      {report['phoneme_duration_stats']['std']:.2f} frames")
    print(f"  <= 1 frame: {report['phoneme_duration_stats']['pct_le_1frame']:.2%}")
    print(f"  <= 2 frame: {report['phoneme_duration_stats']['pct_le_2frame']:.2%}")

    print(f"\nComparison with MAS (from CLAUDE.md):")
    print(f"  MAS degenerate rate:    39-43%")
    print(f"  Julius degenerate rate: {degenerate_rate:.2%}")
    print(f"  MAS phoneme median:     1.0-2.0 frames")
    print(f"  Julius phoneme median:  {report['phoneme_duration_stats']['median']:.1f} frames")
    print(f"  MAS <= 1 frame:         62.9%")
    print(f"  Julius <= 1 frame:      {report['phoneme_duration_stats']['pct_le_1frame']:.2%}")

    # --- レポート保存 ---
    output_path = Path(args.output_report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed report saved to: {output_path}")


if __name__ == "__main__":
    main()
```

#### 2.3 MAS durationとの比較分析（オプション）

既存の`matcha/utils/get_durations_from_trained_model.py`で抽出したMAS durationが利用可能な場合、以下の比較分析を行う:

- 同一発話でのduration分布の比較（ヒストグラム）
- 音素クラスごとのMAS duration vs Julius durationの散布図
- MASで退化した発話がJuliusでどのようなduration分布になったか
- blank[0]のduration比較（MAS: 平均101フレーム → Julius: 0フレーム）

#### 2.4 言語学的妥当性チェック

日本語音素の典型的なduration（文献値）との比較:

| 音素クラス | 期待durationフレーム数 (概算) | 根拠 |
|-----------|---------------------------|------|
| 母音（a,i,u,e,o） | 5-15フレーム（60-180ms） | 日本語母音の平均長 |
| 無声子音（k,t,s,ch,ts） | 3-10フレーム（35-120ms） | 閉鎖+解放 |
| 有声子音（g,d,z,b） | 3-8フレーム（35-95ms） | 閉鎖+解放 |
| 撥音（N） | 5-12フレーム（60-140ms） | モーラ拍 |
| 促音（cl） | 5-15フレーム（60-180ms） | 閉鎖持続 |
| ポーズ（pau） | 5-50フレーム（60-580ms） | 句間ポーズ |
| 無音（sil） | 3-100フレーム（35-1160ms） | 文頭/文末無音 |

これらの範囲を大きく逸脱する音素があれば警告を出力する。

### 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当内容 |
|------|------|---------|
| データ分析エンジニア | 1名 | 統計分析スクリプト実装、レポート生成、可視化 |
| QAエンジニア | 1名 | 品質基準の定義、テスト作成、問題サンプルの分析 |

**合計: 2名**

### 4. 提供範囲とテスト項目

#### 提供範囲
- `scripts/verify_alignment_quality.py`: 品質検証・統計分析スクリプト
- `data/alignment_quality_report.json`: 品質レポート（JSON）
- 問題サンプルリスト（レポートに含む）

#### ユニットテスト

**ファイル**: `tests/test_verify_alignment_quality.py`

| テスト名 | 検証内容 |
|---------|---------|
| `test_is_degenerate_all_one_frame` | 全音素が1フレームの場合にTrueを返すこと |
| `test_is_degenerate_normal_durations` | 正常なduration分布でFalseを返すこと |
| `test_is_degenerate_boundary_80_percent` | 閾値ちょうど80%のケースで正しく判定されること |
| `test_is_degenerate_empty_array` | 空配列でTrueを返すこと |
| `test_compute_phoneme_class_stats_basic` | 音素クラスごとにdurationが正しく集計されること |
| `test_compute_phoneme_class_stats_blank_excluded` | blank位置（偶数インデックス）がクラス統計に含まれないこと |

#### 統合テスト（品質基準テスト）

**ファイル**: `tests/test_alignment_quality_criteria.py`

| テスト名 | 検証内容 |
|---------|---------|
| `test_degenerate_rate_is_zero` | 退化サンプル率が0%であること |
| `test_phoneme_median_duration_above_threshold` | 音素中央durationが3フレーム以上であること |
| `test_short_phoneme_ratio_below_threshold` | 1フレーム以下の音素の割合が10%未満であること |
| `test_no_missing_duration_files` | 全発話のdurationファイルが存在すること |
| `test_no_length_mismatch` | 全発話でduration配列長がテキスト長と一致すること |
| `test_vowel_duration_linguistically_valid` | 母音の平均durationが3-30フレーム範囲であること |
| `test_silence_duration_reasonable` | sil/pauのdurationが0-200フレーム範囲であること |
| `test_blank_durations_all_zero` | 全発話のblank位置durationが0であること |

### 5. 懸念事項とレビュー項目

#### 懸念事項

| 懸念 | 影響度 | 対策 |
|------|-------|------|
| 退化判定基準がMASと同じでよいか | 中 | MASとJuliusで退化のメカニズムが異なるため、Julius固有の品質指標も追加検討 |
| 一部話者でJuliusアライメント品質が低い可能性 | 高 | 話者ごとの統計を出力し、外れ値話者を特定できるようにする |
| 文献値との比較が困難（日本語TTSの音素duration統計データが限られる） | 低 | ESPnetのJVSレシピ、JATTSの公開結果を参考にする |
| MAS durationとの比較にはMASを再実行する必要がある | 中 | 既存の学習チェックポイントから`get_durations_from_trained_model.py`で抽出可能 |

#### コードレビュー項目

- [ ] `is_degenerate`関数のblank除外ロジックが正しいこと（奇数インデックスのみ評価）
- [ ] 音素クラス統計でsymbols_jaの全音素がカバーされていること
- [ ] JSONレポートのフォーマットが後続分析ツールで読み取り可能であること
- [ ] 大量データ（10,000発話）でのメモリ使用量が妥当であること
- [ ] 比較分析の文言がMASの結果を正確に引用していること（CLAUDE.mdの数値と一致）

### 6. 一から作り直すとしたら

**自動化された品質ゲート**を設計する。CI/CDパイプラインに品質基準テストを組み込み、duration生成パイプラインの変更時に自動的に品質検証が走るようにする。現在のスクリプトベースの検証は手動実行が前提だが、`pytest`のフィクスチャとしてレポートJSONを読み込み、品質基準をassertする設計にすれば、回帰テストとして機能する。

**可視化をスクリプトに内蔵**する。matplotlibでのヒストグラム、箱ひげ図、話者ごとのヒートマップ等をレポート生成時に自動出力する。現在のJSON出力は数値分析には十分だが、チームメンバーへの共有やプレゼンテーションには可視化が不可欠。

**サンプルごとの品質スコア**を定義する。退化判定は2値（退化/正常）だが、連続的な品質スコア（例: 音素duration分布のエントロピー、期待durationからのKLダイバージェンス）を定義すれば、ボーダーラインのサンプルを特定しやすくなる。学習時にサンプル品質スコアで重み付けする拡張も可能になる。

### 7. 後続タスクへの連絡事項

**M2（PrecomputedDataModule対応）への連絡**:
- 品質検証で問題なしと判定された発話のリストを提供する。問題のある発話は`.pt`ファイルからdurationを除外（`None`として保持）することを推奨
- 退化率0%が確認された場合、`use_precomputed_durations=True`での学習が安全であることの証拠として本レポートを参照

**M4（学習設定変更）への連絡**:
- 音素クラスごとのduration統計は、Duration Predictorの品質評価基準として使用可能
- blank durationが全て0であることの確認結果は、Duration Predictorのblank予測ターゲットの設計に影響する
- MASとの比較結果は、学習前後の品質変化を評価する際のベースラインとして使用

**M5（推論・評価・品質検証）への連絡**:
- 本レポートの音素duration分布を、推論時のDuration Predictor出力と比較するためのゴールドスタンダードとして提供する
- 話者ごとの統計データは、話者別の音声品質評価の参考値として使用可能
