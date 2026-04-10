# M2: PrecomputedDataModule・前処理スクリプトのduration対応

## マイルストーン概要

M1（Julius forced alignment）で生成された音素duration `.npy` ファイルを、学習データパイプラインに統合する。具体的には、前処理スクリプト `precompute_dataset.py` でdurationを `.pt` ファイルに埋め込み、`PrecomputedTextMelDataset` がそれを読み込んでモデルの `forward()` まで渡す経路を整備する。

これにより `model.use_precomputed_durations=true` でMASを完全にバイパスし、外部アライナーの正確なdurationターゲットでDuration Predictorを学習できるようになる。

### 依存関係

- **M1（Julius forced alignmentパイプライン構築）に完全に依存**: M1が各発話の `.npy` durationファイルを生成する。M2はそれを消費するだけであり、M1完了前にコード変更とテストは実施できるが、実データでの `.pt` 再生成（T-M2-03）はM1完了後に行う。

### 完了条件

1. `precompute_dataset.py --durations-dir` で duration付き `.pt` ファイルが生成される
2. `PrecomputedTextMelDataset` が `load_durations=True` 時に `.pt` からdurationを読み込む
3. `TextMelBatchCollate` がdurationをパディングしてバッチに含める（既存実装を活用）
4. `model.forward()` にdurationが `(B, 1, T_text)` shapeで到達し、`generate_path()` がMASなしでアライメントを生成する
5. train/valの全10,000サンプルのduration付き `.pt` ファイルが `/dev/shm/jvs_precomputed/` に配置される
6. 全ユニットテスト・結合テストがパスする

### 想定期間

- T-M2-01: 0.5日（コード変更 + テスト）
- T-M2-02: 0.5日（コード変更 + テスト）
- T-M2-03: 0.5日（実行 + 検証、M1完了後）
- **合計: 1.5日**（T-M2-01とT-M2-02は並行可能だが、T-M2-03はM1完了後）

### 一から作り直すとしたらの思考

M2全体を白紙から設計する場合、「.ptファイルにdurationを埋め込む」アプローチ自体は変えない。理由は:

1. **既存パスの活用**: `TextMelBatchCollate` は既にdurationのパディング・バッチ化を実装済み（`text_mel_datamodule.py` L267-282, L296）。`matcha_tts.py` L193-194に `use_precomputed_durations` パスが既存。新規アーキテクチャ設計は不要。
2. **代替案: `.npy` を別途読み込む方式**: `.pt` と `.npy` を別ディレクトリから同期して読む方法もあるが、ファイル名の一致確認・I/Oの二重化・`preload_to_memory` 非対応など問題が多い。`.pt` に統合する方が単純で高速。
3. **代替案: HDF5やLMDB**: 10,000サンプル程度ではオーバーヘッドのみ増え、利点がない。
4. **duration shapの設計判断**: モデル側 `generate_path()` は `(B, T_text)` を受け取る。collateでは `(B, T_text)` で格納し、`get_losses()` → `forward()` 経由で `durations` として渡す。`forward()` 内の `durations.squeeze(1)` は `(B, 1, T_text)` 入力を想定しているため、collate出力に `.unsqueeze(1)` を追加するか、`squeeze(1)` がno-opとなる `(B, T_text)` のまま渡すかの選択がある。既存 `TextMelBatchCollate` は `(B, T_text)` で返し `get_losses()` がそのまま渡すため、`forward()` の `durations.squeeze(1)` が `(B, T_text)` に対してno-opとなることを確認した上で、既存挙動を維持する。

---

## T-M2-01: precompute_dataset.pyへのduration埋め込み機能追加

### タスク目的とゴール

`scripts/precompute_dataset.py` に `--durations-dir` 引数を追加し、M1で生成されたduration `.npy` ファイルを読み込んで `.pt` ファイルに `"durations"` キーとして埋め込む。durationが存在しないサンプルは警告付きでスキップする。

**ゴール**: `--durations-dir` 指定時に出力される `.pt` ファイルが `{"mel", "text", "spk", "cleaned_text", "durations"}` の5キーを持ち、`durations` のshapeが `text` の長さと一致すること。

### 実装する内容の詳細

#### 1. コマンドライン引数の追加

**ファイル**: `scripts/precompute_dataset.py` の `main()` 関数内、L104-141

```python
parser.add_argument(
    "--durations-dir",
    type=str,
    default=None,
    help="Directory containing .npy duration files from Julius forced alignment (M1 output). "
         "File naming convention: {spk_name}_{utterance_id}.npy (e.g., jvs001_VOICEACTRESS100_001.npy). "
         "If not specified, durations are not included in .pt files.",
)
```

#### 2. duration読み込み関数の追加

**ファイル**: `scripts/precompute_dataset.py`、`process_sample()` の前に新規関数を追加

```python
def load_duration(durations_dir: Path, spk_name: str, stem: str, expected_text_len: int):
    """Load a .npy duration file and validate its length against text sequence.

    Args:
        durations_dir: Directory containing .npy duration files.
        spk_name: Speaker directory name (e.g., "jvs001").
        stem: Utterance stem name (e.g., "VOICEACTRESS100_001").
        expected_text_len: Length of text sequence after intersperse (2*n_phonemes + 1).

    Returns:
        torch.IntTensor of duration values, or None if file not found.

    Raises:
        ValueError: If duration length does not match expected text length.
    """
    npy_name = f"{spk_name}_{stem}.npy"
    npy_path = durations_dir / npy_name
    if not npy_path.exists():
        return None
    dur = np.load(str(npy_path)).astype(int)
    dur_tensor = torch.from_numpy(dur).int()
    if len(dur_tensor) != expected_text_len:
        raise ValueError(
            f"Duration length mismatch for {npy_name}: "
            f"duration has {len(dur_tensor)} elements, "
            f"text (after intersperse) has {expected_text_len} elements"
        )
    return dur_tensor
```

**注意**: `import numpy as np` が必要（現在のファイルには未インポート）。

#### 3. `process_sample()` の修正

**ファイル**: `scripts/precompute_dataset.py` L46-92

現在の `process_sample()` シグネチャと保存ロジックを修正する:

```python
def process_sample(
    wav_path: str,
    spk: int,
    text: str,
    output_dir: Path,
    mel_mean: float,
    mel_std: float,
    durations_dir: Path | None = None,
):
    """Compute mel + text sequence for a single sample and save as .pt.

    Returns:
        tuple: (out_path, skipped) where skipped is True if duration was required but missing.
    """
    wav_p = Path(wav_path)
    spk_name = wav_p.parent.name
    out_path = output_dir / f"{spk_name}_{wav_p.stem}.pt"

    # -- mel spectrogram -- (既存コードそのまま)
    data, sr = sf.read(wav_path, dtype="float32")
    assert sr == SAMPLE_RATE, f"Expected {SAMPLE_RATE} Hz, got {sr} Hz for {wav_path}"
    audio = torch.from_numpy(data).unsqueeze(0)
    mel = mel_spectrogram(
        audio, N_FFT, N_MELS, SAMPLE_RATE, HOP_LENGTH, WIN_LENGTH, F_MIN, F_MAX, center=False,
    ).squeeze()
    mel = normalize(mel, mel_mean, mel_std)

    # -- text sequence -- (既存コードそのまま)
    text_norm, cleaned_text = text_to_sequence(text, ["japanese_cleaners"], language="ja")
    text_norm = intersperse(text_norm, 0)
    text_norm = torch.IntTensor(text_norm)

    # -- duration (新規) --
    duration = None
    skipped = False
    if durations_dir is not None:
        duration = load_duration(durations_dir, spk_name, wav_p.stem, len(text_norm))
        if duration is None:
            return out_path, True  # スキップ

    # -- save --
    save_dict = {
        "mel": mel,
        "text": text_norm,
        "spk": spk,
        "cleaned_text": cleaned_text,
    }
    if duration is not None:
        save_dict["durations"] = duration

    torch.save(save_dict, out_path)
    return out_path, False
```

#### 4. GPUモードの修正

**ファイル**: `scripts/precompute_dataset.py` L188-226（GPUモードのメインスレッドmel計算部分）

GPUモードでは `process_sample_text_only()` でテキスト処理を並列実行した後、メインスレッドでmel計算と保存を行う。duration読み込みはメインスレッドの保存部分に追加する:

```python
# GPUモード保存部分に追加
duration = None
if args.durations_dir is not None:
    dur_dir = Path(args.durations_dir)
    duration = load_duration(dur_dir, spk_name, wav_p.stem, len(text_norm))
    if duration is None:
        skipped_count += 1
        tqdm.write(f"WARNING: Duration not found for {spk_name}_{wav_p.stem}, skipping")
        continue

save_dict = {
    "mel": mel,
    "text": text_norm,
    "spk": spk,
    "cleaned_text": cleaned_text,
}
if duration is not None:
    save_dict["durations"] = duration
torch.save(save_dict, out_path)
```

#### 5. CPUモードのエラーハンドリング修正

**ファイル**: `scripts/precompute_dataset.py` L228-256

CPUモードでは `process_sample()` を `ProcessPoolExecutor` で呼び出す。`durations_dir` 引数を追加で渡す:

```python
future = executor.submit(
    process_sample,
    wav_path,
    spk,
    text,
    output_dir,
    args.mel_mean,
    args.mel_std,
    Path(args.durations_dir) if args.durations_dir else None,
)
```

結果処理でスキップをカウント:

```python
out_path, skipped = future.result()
if skipped:
    skipped_count += 1
    tqdm.write(f"WARNING: Duration not found, skipped: {out_path}")
```

#### 6. サマリ出力の修正

最終出力にスキップ数を表示:

```python
if args.durations_dir:
    print(f"Duration embedding: {total_processed - skipped_count} samples with durations, {skipped_count} skipped")
```

### エージェントチームの役割と人数

- **実装担当**: 1名。`precompute_dataset.py` の引数追加・関数修正・テスト作成
- **レビュー担当**: 1名。duration長の一致検証ロジック、エラーハンドリングの網羅性確認

### 提供範囲とテスト項目

#### ユニットテスト

**ファイル**: `tests/test_precompute_dataset.py`（新規作成）

```python
# test_load_duration_valid
# - 正常な.npyファイルを作成し、load_duration()がtorch.IntTensorを返すことを確認
# - duration長がexpected_text_lenと一致

# test_load_duration_missing_file
# - 存在しない.npyパスに対してNoneが返ることを確認

# test_load_duration_length_mismatch
# - text長と異なるdurationを用意し、ValueErrorが発生することを確認

# test_process_sample_with_duration
# - ダミーwav + ダミーduration .npyで process_sample() を呼び、
#   出力.ptに "durations" キーが存在し、shapeがtextと一致することを確認
# - @pytest.mark.slow (wav生成・mel計算が必要)

# test_process_sample_without_duration_dir
# - durations_dir=None でprocess_sample()を呼び、
#   出力.ptに "durations" キーが存在しないことを確認
# - @pytest.mark.slow

# test_process_sample_skip_missing_duration
# - durations_dirは指定するが対応.npyが存在しないケースで、
#   process_sample()がskipped=Trueを返すことを確認
# - @pytest.mark.slow
```

#### 結合テスト

```python
# test_precompute_roundtrip_with_durations
# - 小規模filelist(2-3サンプル)を作成
# - ダミーduration .npyを生成
# - main()をargsリスト経由で実行
# - 生成された.ptファイルを読み込み、全キーの存在とshape整合性を確認
# - @pytest.mark.slow
```

### 懸念事項とレビュー項目

1. **duration `.npy` のファイル命名規則**: M1のJulius出力が `{spk_name}_{utterance_stem}.npy` 形式であることを前提とする。M1チケットと命名規則を事前に合意する必要がある。`.pt` ファイルの命名 `{spk_name}_{stem}.pt`（`precompute_dataset.py` L58）と一致させること。

2. **intersperse後の長さ一致**: テキストは `intersperse(text_norm, 0)` により長さが `2 * n_phonemes + 1` になる。M1のdurationもblank挿入後の長さで生成されている必要がある。長さ不一致はハードエラー（`ValueError`）とし、サイレントな不整合を防ぐ。

3. **`ProcessPoolExecutor` でのnumpy読み込み**: `load_duration()` は `np.load()` を使うが、子プロセス内でも安全に動作する。ただし `durations_dir` が `Path` オブジェクトであることを確認（pickleシリアライズ可能性）。

4. **既存 `.pt` ファイルとの後方互換性**: `--durations-dir` 未指定時は従来通り `"durations"` キーなしで保存する。`PrecomputedTextMelDataset` 側が `.get("durations", None)` でフォールバックすることで後方互換を維持する（T-M2-02で対応）。

5. **durationの型**: `torch.IntTensor`（int32）。`generate_path()` は内部で `torch.cumsum()` を呼ぶため、int型で十分。float変換は不要。

### 一から作り直すとしたら

`precompute_dataset.py` を白紙から書く場合、duration読み込みを最初から設計に含め、以下の構造にする:

- **設定ファイル駆動**: コマンドライン引数ではなくYAML設定ファイルで入出力パスを管理。ただし現状のスクリプトはCLI引数方式で統一されており、1回限りの前処理ツールとしてはCLI引数が適切。
- **検証パス**: `.pt` 生成後に自動検証（shape一致、NaN検出、duration合計値とmel長の比較）を組み込む。現設計では別スクリプトでの検証を想定しているが、`--verify` フラグで組み込むこともできた。
- **判断**: 既存スクリプトへの最小限の追加が最もリスクが低い。全面書き直しは不要。

### 後続タスクへの連絡事項

- **T-M2-02へ**: `.pt` ファイルに `"durations"` キーが含まれる場合と含まれない場合の両方がある。`_load_from_disk()` では `data.get("durations", None)` でフォールバックすること。
- **T-M2-03へ**: `--durations-dir` の命名規則は `{spk_name}_{utterance_stem}.npy`。M1の出力ディレクトリ構造を確認の上、パスを指定すること。
- **M1へ**: duration `.npy` のファイル命名規則として `{spk_name}_{utterance_stem}.npy` を使用する。中身は `int64` または `int32` のnumpy配列で、長さは intersperse後のテキスト長（`2 * n_phonemes + 1`）と一致させること。

---

## T-M2-02: PrecomputedDataModuleのduration読み込み対応

### タスク目的とゴール

`PrecomputedTextMelDataset._load_from_disk()` の `"durations": None` ハードコードを修正し、`load_durations=True` 設定時に `.pt` ファイルからdurationを読み込んでモデルの `forward()` まで渡す。

**ゴール**: `configs/data/jvs_precomputed.yaml` で `load_durations: true` を設定した場合、DataLoaderから出力されるバッチの `"durations"` キーに正しくパディングされたdurationテンソルが含まれ、`matcha_tts.py` L193-194の `generate_path(durations.squeeze(1), ...)` が正常に動作すること。

### 実装する内容の詳細

#### 1. `PrecomputedTextMelDataset` に `load_durations` パラメータを追加

**ファイル**: `matcha/data/precomputed_datamodule.py` L162-227

```python
class PrecomputedTextMelDataset(Dataset):
    """Dataset for pre-computed .pt files containing mel spectrograms and text sequences.

    Each .pt file is expected to contain a dict with keys:
        - "mel": Tensor of shape (n_feats, mel_length)
        - "text": IntTensor of phoneme indices
        - "spk": int speaker id
        - "cleaned_text": str
        - "durations": IntTensor of phoneme durations (optional, when load_durations=True)
    """

    def __init__(self, pt_dir, n_spks, seed=None, preload_to_memory=False, load_durations=False):
        self.pt_dir = Path(pt_dir)
        self.n_spks = n_spks
        self.load_durations = load_durations
        # ... (以下既存コードそのまま)
```

変更点: `__init__` シグネチャに `load_durations=False` を追加し、`self.load_durations` として保持。

#### 2. `_load_from_disk()` の修正

**ファイル**: `matcha/data/precomputed_datamodule.py` L195-212

現在のコード:
```python
def _load_from_disk(self, index):
    pt_path = self.pt_paths[index]
    data = torch.load(pt_path, weights_only=True)
    mel = data["mel"]
    text = data["text"]
    spk = data["spk"] if self.n_spks > 1 else None
    cleaned_text = data["cleaned_text"]
    return {
        "x": text,
        "y": mel,
        "spk": spk,
        "filepath": str(pt_path),
        "x_text": cleaned_text,
        "durations": None,
    }
```

修正後:
```python
def _load_from_disk(self, index):
    pt_path = self.pt_paths[index]
    data = torch.load(pt_path, weights_only=True)
    mel = data["mel"]
    text = data["text"]
    spk = data["spk"] if self.n_spks > 1 else None
    cleaned_text = data["cleaned_text"]

    durations = None
    if self.load_durations:
        durations = data.get("durations", None)
        if durations is None:
            raise KeyError(
                f"load_durations=True but 'durations' key not found in {pt_path}. "
                f"Re-run precompute_dataset.py with --durations-dir to embed durations."
            )
        # 長さ整合性チェック
        if len(durations) != len(text):
            raise ValueError(
                f"Duration length ({len(durations)}) != text length ({len(text)}) in {pt_path}"
            )

    return {
        "x": text,
        "y": mel,
        "spk": spk,
        "filepath": str(pt_path),
        "x_text": cleaned_text,
        "durations": durations,
    }
```

**設計判断**: `load_durations=True` でdurationが見つからない場合はハードエラー（`KeyError`）とする。サイレントに `None` を返すと、モデル側で `generate_path(None.squeeze(1), ...)` がクラッシュし、原因特定が困難になるため。

#### 3. `PrecomputedTextMelDataModule.setup()` の修正

**ファイル**: `matcha/data/precomputed_datamodule.py` L250-262

`setup()` 内で `PrecomputedTextMelDataset` に `load_durations` を渡す:

```python
def setup(self, stage: str | None = None):
    self.trainset = PrecomputedTextMelDataset(
        self.hparams.train_pt_dir,
        self.hparams.n_spks,
        self.hparams.seed,
        preload_to_memory=self.hparams.preload_to_memory,
        load_durations=self.hparams.load_durations,
    )
    self.validset = PrecomputedTextMelDataset(
        self.hparams.val_pt_dir,
        self.hparams.n_spks,
        self.hparams.seed,
        preload_to_memory=self.hparams.preload_to_memory,
        load_durations=self.hparams.load_durations,
    )
```

#### 4. collate関数の確認（変更不要）

**ファイル**: `matcha/data/text_mel_datamodule.py` L247-297

`TextMelBatchCollate.__call__()` は既にdurationを正しく処理している:

- L267: `durations = torch.zeros((B, x_max_length), dtype=torch.long)` で初期化
- L281-282: `if item["durations"] is not None: durations[i, : item["durations"].shape[-1]] = item["durations"]` でパディング
- L296: `"durations": durations if not torch.eq(durations, 0).all() else None` で全ゼロならNoneに変換

**注意点**: L296の `torch.eq(durations, 0).all()` チェックは、durationが全サンプルでNoneの場合（`load_durations=False`）にバッチのdurationをNoneにするための既存ロジック。`load_durations=True` 時はdurationが非ゼロ値を含むため、正しくテンソルとして返される。変更不要。

#### 5. データフローの確認（変更不要だが要確認）

**ファイル**: `matcha/models/baselightningmodule.py` L100-113

`get_losses()` は既に `batch["durations"]` を `self()` に渡している:

```python
dur_loss, prior_loss, diff_loss, *_ = self(
    x=x, x_lengths=x_lengths, y=y, y_lengths=y_lengths,
    spks=spks, out_size=self.out_size,
    durations=batch["durations"],
)
```

**ファイル**: `matcha/models/matcha_tts.py` L161, L193-194

`forward()` は `durations` を受け取り、`use_precomputed_durations=True` のとき:
```python
attn = generate_path(durations.squeeze(1), attn_mask.squeeze(1))
```

`TextMelBatchCollate` は `(B, T_text)` shapeで返し、`durations.squeeze(1)` は2D tensorに対してno-opとなるため、`generate_path()` に正しく `(B, T_text)` が渡される。

**ファイル**: `matcha/utils/model.py` L33-45

`generate_path(duration, mask)` は `duration` shape `(B, T_text)` と `mask` shape `(B, T_text, T_mel)` を受け取る。duration値は各音素のフレーム数（整数）。`torch.cumsum(duration, 1)` で累積和を計算し、`sequence_mask()` でバイナリアライメント行列を構築する。

#### 6. 設定ファイルの更新（T-M2-03で実施）

**ファイル**: `configs/data/jvs_precomputed.yaml` L14

現在: `load_durations: false`

T-M2-03で実データが揃った段階で `true` に変更する。T-M2-02のコードテストでは `load_durations: true` をHydra override経由で設定する。

### エージェントチームの役割と人数

- **実装担当**: 1名。`precomputed_datamodule.py` の修正、テスト作成
- **レビュー担当**: 1名。shape整合性の確認、エラーメッセージの適切性、後方互換性の確認

### 提供範囲とテスト項目

#### ユニットテスト

**ファイル**: `tests/test_precomputed_datamodule.py`（新規作成）

```python
# test_load_from_disk_with_durations
# - "durations" キーを含む .pt ファイルをtmpに作成
# - load_durations=True でデータセットを生成
# - _load_from_disk() の戻り値に "durations" が含まれ、型・shape が正しいことを確認

# test_load_from_disk_without_durations
# - "durations" キーを含まない .pt ファイルをtmpに作成
# - load_durations=False でデータセットを生成
# - _load_from_disk() の戻り値の "durations" が None であることを確認

# test_load_from_disk_missing_durations_key_error
# - "durations" キーを含まない .pt ファイルをtmpに作成
# - load_durations=True でデータセットを生成
# - _load_from_disk() が KeyError を発生させることを確認

# test_load_from_disk_duration_length_mismatch
# - text長と異なるdurationを含む .pt をtmpに作成
# - load_durations=True で _load_from_disk() が ValueError を発生させることを確認

# test_collate_with_durations
# - duration付きの複数サンプル辞書をリスト化
# - TextMelBatchCollate(n_spks=100) で collate
# - 戻り値の "durations" が (B, max_x_length) shapeの torch.long であることを確認
# - パディング部分がゼロであることを確認

# test_collate_without_durations
# - duration=None の複数サンプル辞書をリスト化
# - TextMelBatchCollate(n_spks=100) で collate
# - 戻り値の "durations" が None であることを確認
```

#### 結合テスト

```python
# test_dataloader_to_model_forward_with_durations
# - tmpに duration付き .pt ファイル (3-5サンプル) を作成
# - PrecomputedTextMelDataModule を load_durations=True で構築
# - setup("fit") 後に train_dataloader() からバッチ取得
# - バッチの "durations" が non-None の (B, max_T_text) テンソルであることを確認
# - duration.sum(dim=1) の各要素が対応する y_lengths 以下であることを確認
# - @pytest.mark.slow

# test_preload_to_memory_with_durations
# - duration付き .pt ファイルで preload_to_memory=True を設定
# - _cache にdurationが含まれることを確認
```

### 懸念事項とレビュー項目

1. **`torch.load(weights_only=True)` と `"durations"` キー**: PyTorch 2.x の `weights_only=True` はテンソルと基本型のみ許可するが、`torch.IntTensor` は許可される。問題ないことを確認済み。

2. **duration shapeの一貫性**: `.pt` 内のdurationは1D `(T_text,)` で保存される。`TextMelBatchCollate` は2D `(B, max_T_text)` にパディングする。`forward()` の `durations.squeeze(1)` は2D入力に対してno-opであり、`generate_path()` は `(B, T_text)` を受け取る。この全経路でshapeが一貫していることをテストで検証する。

3. **`preload_to_memory` との整合性**: `_preload()` は `_load_from_disk()` を呼ぶため、`load_durations=True` 時もキャッシュにdurationが含まれる。追加対応不要だが、テストで明示的に確認する。

4. **後方互換性**: `load_durations=False`（デフォルト）のとき、既存の `"durations"` キーなし `.pt` ファイルで正常動作すること。`data.get("durations", None)` は `load_durations=True` 時のみ実行されるため、`False` 時は `"durations": None` がハードコードされ、既存動作と同一。

5. **L296の全ゼロチェック**: `TextMelBatchCollate` L296 の `torch.eq(durations, 0).all()` は、理論上はdurationが全音素で0フレームのケースでNoneを返してしまう。実際にはJuliusアライメントで全音素が0フレームになることはあり得ないため問題ないが、edge caseとして認識しておく。

### 一から作り直すとしたら

`PrecomputedTextMelDataset` を白紙から設計する場合:

- **`.pt` フォーマットのバージョニング**: `.pt` 内に `"version": 2` のようなメタキーを含め、duration対応前（v1）と後（v2）を区別する仕組みを入れる。現状はキーの有無で判別しており、十分にシンプルだが、将来さらにキーが増える場合にバージョン管理が有効。ただし現時点では過剰設計であり、不採用が妥当。
- **collate関数の分離**: `TextMelBatchCollate` は `text_mel_datamodule.py` に属しており、`PrecomputedTextMelDataModule` から import して使っている。理想的にはcollate関数を共通モジュールに切り出すべきだが、現状で動作しており変更範囲を最小化する方が優先。

### 後続タスクへの連絡事項

- **T-M2-03へ**: `load_durations=True` を設定する場合、`.pt` ファイルに `"durations"` キーが必須。キーがなければ `KeyError` で即座に失敗する。全 `.pt` にdurationが埋め込まれていることを確認してから設定を変更すること。
- **M3（Duration PredictorのFiLM）へ**: M2完了後、`forward()` に正確なdurationが到達する。`dur_loss` はMASではなく外部アライナーのdurationをターゲットとして計算される。M3でDPのアーキテクチャを変更しても、データフロー自体はM2で完成している。
- **M4（学習設定変更）へ**: `configs/data/jvs_precomputed.yaml` の `load_durations: true` は T-M2-03 完了後に変更する。`configs/model/matcha.yaml` の `use_precomputed_durations: ${data.load_durations}` は既に設定済みで、追加変更不要。

---

## T-M2-03: .ptファイル再生成（duration付き）

### タスク目的とゴール

T-M2-01で修正した `precompute_dataset.py` とM1で生成されたduration `.npy` ファイルを用いて、train/valの全 `.pt` ファイルをduration付きで再生成する。再生成後、形式・shape・整合性を検証し、`/dev/shm/jvs_precomputed/` にデプロイする。

**ゴール**: `/dev/shm/jvs_precomputed/{train,val}/` 配下の全 `.pt` ファイルが `"durations"` キーを含み、`load_durations: true` での学習が即座に開始可能な状態になること。

### 実装する内容の詳細

#### 1. M1出力の確認

M1のJulius forced alignmentの出力ディレクトリを確認する。想定構造:

```
data/jvs_durations/
  jvs001_VOICEACTRESS100_001.npy
  jvs001_VOICEACTRESS100_002.npy
  ...
  jvs100_VOICEACTRESS100_100.npy
```

各 `.npy` は int配列で、長さは intersperse後のテキスト長（`2 * n_phonemes + 1`）。

#### 2. train分の再生成

```bash
uv run python scripts/precompute_dataset.py \
  --filelist data/jvs/train.txt \
  --output-dir data/jvs_precomputed_v2/train \
  --mel-mean -6.550095 --mel-std 2.383771 \
  --durations-dir data/jvs_durations \
  --num-workers 8
```

**注意**: 出力先は既存の `data/jvs_precomputed/train` ではなく、`data/jvs_precomputed_v2/train` とする。既存ファイルを上書きしないことで、問題発生時のロールバックを可能にする。

#### 3. val分の再生成

```bash
uv run python scripts/precompute_dataset.py \
  --filelist data/jvs/val.txt \
  --output-dir data/jvs_precomputed_v2/val \
  --mel-mean -6.550095 --mel-std 2.383771 \
  --durations-dir data/jvs_durations \
  --num-workers 8
```

#### 4. 検証スクリプトの実行

**ファイル**: `scripts/validate_precomputed_durations.py`（新規作成、T-M2-03専用の検証ツール）

```python
"""Validate precomputed .pt files contain correctly shaped durations.

Usage:
    python scripts/validate_precomputed_durations.py \
        --pt-dir data/jvs_precomputed_v2/train \
        --expect-durations
"""
import argparse
from pathlib import Path
import torch
from tqdm import tqdm


def validate(pt_dir: str, expect_durations: bool):
    pt_dir = Path(pt_dir)
    pt_files = sorted(pt_dir.glob("*.pt"))
    print(f"Found {len(pt_files)} .pt files in {pt_dir}")

    errors = []
    stats = {"total": 0, "with_dur": 0, "dur_sum_mismatch": 0}

    for pt_path in tqdm(pt_files, desc="Validating"):
        stats["total"] += 1
        data = torch.load(pt_path, weights_only=True)

        # 必須キーの確認
        for key in ["mel", "text", "spk", "cleaned_text"]:
            if key not in data:
                errors.append((pt_path.name, f"Missing key: {key}"))

        text = data["text"]
        mel = data["mel"]

        if expect_durations:
            if "durations" not in data:
                errors.append((pt_path.name, "Missing 'durations' key"))
                continue

            dur = data["durations"]
            stats["with_dur"] += 1

            # 長さ一致チェック
            if len(dur) != len(text):
                errors.append((pt_path.name,
                    f"Duration length ({len(dur)}) != text length ({len(text)})"))

            # duration合計値とmel長の比較
            dur_sum = dur.sum().item()
            mel_len = mel.shape[-1]
            if dur_sum != mel_len:
                stats["dur_sum_mismatch"] += 1
                # 許容範囲: +/- 2フレーム (端数丸めの影響)
                if abs(dur_sum - mel_len) > 2:
                    errors.append((pt_path.name,
                        f"Duration sum ({dur_sum}) far from mel length ({mel_len})"))

            # NaN/Inf チェック
            if torch.isnan(mel).any():
                errors.append((pt_path.name, "NaN in mel"))
            if (dur < 0).any():
                errors.append((pt_path.name, "Negative duration values"))

    # サマリ出力
    print(f"\n=== Validation Summary ===")
    print(f"Total files: {stats['total']}")
    if expect_durations:
        print(f"With durations: {stats['with_dur']}")
        print(f"Duration sum != mel length: {stats['dur_sum_mismatch']}")
    if errors:
        print(f"\nERRORS ({len(errors)}):")
        for name, msg in errors[:20]:  # 最大20件表示
            print(f"  {name}: {msg}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")
    else:
        print("No errors found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt-dir", required=True)
    parser.add_argument("--expect-durations", action="store_true")
    args = parser.parse_args()
    validate(args.pt_dir, args.expect_durations)
```

実行コマンド:
```bash
# train検証
uv run python scripts/validate_precomputed_durations.py \
  --pt-dir data/jvs_precomputed_v2/train --expect-durations

# val検証
uv run python scripts/validate_precomputed_durations.py \
  --pt-dir data/jvs_precomputed_v2/val --expect-durations
```

#### 5. /dev/shmへのデプロイ

検証完了後:

```bash
# 既存データのバックアップ
mv /dev/shm/jvs_precomputed /dev/shm/jvs_precomputed_backup

# 新データの配置
cp -r data/jvs_precomputed_v2 /dev/shm/jvs_precomputed

# 動作確認: 数バッチ読み込み
uv run python -c "
from matcha.data.precomputed_datamodule import PrecomputedTextMelDataset
ds = PrecomputedTextMelDataset('/dev/shm/jvs_precomputed/train', n_spks=100, load_durations=True)
sample = ds[0]
print(f'text shape: {sample[\"x\"].shape}')
print(f'mel shape: {sample[\"y\"].shape}')
print(f'durations shape: {sample[\"durations\"].shape}')
print(f'durations sum: {sample[\"durations\"].sum().item()}')
print(f'mel length: {sample[\"y\"].shape[-1]}')
"
```

#### 6. 設定ファイルの更新

**ファイル**: `configs/data/jvs_precomputed.yaml`

```yaml
load_durations: true
```

L14の `load_durations: false` を `true` に変更する。`configs/model/matcha.yaml` L16の `use_precomputed_durations: ${data.load_durations}` により、モデル側も自動的にMASバイパスモードに切り替わる。

### エージェントチームの役割と人数

- **実行担当**: 1名。前処理スクリプト実行、検証、デプロイ
- **確認担当**: 1名。検証結果の確認、サンプリングによるspot check

### 提供範囲とテスト項目

#### 実行時検証

```
# 検証項目チェックリスト

# [1] ファイル数の確認
# - train: 期待サンプル数（~9,000）と一致
# - val: 期待サンプル数（~1,000）と一致
# - スキップされたサンプルが0件であること

# [2] 全ファイルに "durations" キーが存在
# - validate_precomputed_durations.py の出力でエラー0件

# [3] duration長とtext長の一致
# - 全サンプルで len(durations) == len(text)

# [4] duration合計値とmel長の整合性
# - dur.sum() と mel.shape[-1] の差が2フレーム以内
# - 大幅な乖離がある場合はM1のアライメント品質を再確認

# [5] durationの統計値が妥当
# - 平均duration（blank除外）: 3-10フレーム程度
# - 最大duration: 50フレーム以下が大半
# - duration=0の音素比率: 5%以下（退化アライメントではない）

# [6] /dev/shmへのコピー後の動作確認
# - DataLoader から1バッチ読み込み成功
# - バッチの "durations" が non-None
```

#### 統合テスト

```bash
# 5ステップの学習ドライランで全パイプライン確認
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python matcha/train.py \
  experiment=jvs_fast compile_model=false \
  data.batch_size=4 data.num_workers=0 +data.preload_to_memory=false \
  data.load_durations=true \
  trainer.max_epochs=1 trainer.limit_train_batches=5 trainer.limit_val_batches=2 \
  test=false
```

確認事項:
- MASブロック（L196-208）がスキップされること（ログにMAS関連の出力がないこと）
- `dur_loss`, `prior_loss`, `diff_loss` が全てfinite値であること
- `dur_loss` がMAS使用時より小さいことが期待される（ターゲットが正確なため）

### 懸念事項とレビュー項目

1. **M1出力の完全性**: M1がtrain/valの全サンプルに対してduration `.npy` を生成していることが前提。欠損があれば `precompute_dataset.py` がスキップし、`.pt` にdurationが含まれない。スキップ0件であることを確認すること。

2. **duration合計値とmel長の不一致**: Julius forced alignmentの時間精度は10ms（hop_length=256/22050Hz=11.6msに近い）だが、完全一致は保証されない。M1のduration生成スクリプトがフレーム単位への変換時に端数をどう処理するかに依存する。合計値がmel長と一致しない場合、`generate_path()` が不正なアライメントを生成する可能性がある。M1側でduration合計をmel長に一致させる正規化処理が必要（M1チケットで対応）。

3. **ディスク容量**: `.pt` ファイルにdurationが追加されることによる容量増加は軽微（1サンプルあたり数百バイト増）。10,000サンプルで数MB程度であり、`/dev/shm` の容量に影響しない。

4. **既存チェックポイントとの非互換性**: `load_durations: true` に切り替えた後、既存のMASベースチェックポイントから再開すると、モデル重みは互換だがDuration Predictorのターゲットが変わるため、`dur_loss` が一時的に急増する可能性がある。M4で段階的学習を計画しているため、この影響はM4で管理する。

5. **/dev/shmの揮発性**: `/dev/shm` はRAMディスクであり再起動で消失する。`data/jvs_precomputed_v2/` にディスク上のコピーを維持し、再起動後に再コピーできるようにする。

### 一から作り直すとしたら

再生成プロセス自体はシンプルであり、大幅な設計変更の余地は少ない。改善点としては:

- **Makefile/シェルスクリプト化**: 検証・デプロイを含めた一連のパイプラインをMakefileのターゲットとして定義し、`make precompute-with-durations` で一発実行可能にする。現状はマニュアル手順で十分だが、再実行頻度が高ければ自動化の価値がある。
- **差分更新**: 全ファイル再生成ではなく、duration追加分のみを既存 `.pt` に注入する方式。`torch.load()` → durationキー追加 → `torch.save()` で実現可能だが、mel正規化パラメータの変更時に対応できないため、全再生成の方が安全。

### 後続タスクへの連絡事項

- **M3へ**: M2完了後、`data.load_durations=true` での学習が可能。M3のFiLM変更はM2と独立だが、M3テスト時にduration付き `.pt` を使用すること。
- **M4へ**: `/dev/shm/jvs_precomputed/` にduration付きデータが配置済み。`configs/data/jvs_precomputed.yaml` の `load_durations: true` が設定済み。学習開始時に `data.load_durations=true` のoverrideは不要（設定ファイルに反映済み）。MASベースの既存チェックポイントからの再開 vs 新規学習の判断はM4で行う。
- **M1へ**: duration `.npy` のフレーム合計値がmel長と一致することを保証すること。不一致がある場合、最後のblank（intersperse後の末尾要素）のdurationを調整して合計値を合わせる正規化処理をM1側で実装すること。
