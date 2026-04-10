# M5: 推論・評価・品質検証

## マイルストーン概要

M4（外部アライナーduration + FiLM DP + Blank zero-initによる学習）の完了後、学習済みモデルの品質を定量・定性の両面で検証するフェーズ。MAS退化問題（39-43%の退化率）が外部アライナーアプローチで完全に解消されたことを数値で確認し、日本語100話者TTSとして実用水準に達しているかを判定する。

### 依存関係

```
M1: Julius Alignment ──→ M2: DataModule対応 ──→ M4: 学習実行 ──→ [M5: 評価]（本マイルストーン）
                                                    ↑
M3: FiLM DP + Blank init ─────────────────────────┘
```

M5はM4の学習済みチェックポイントに完全に依存する。M4が完了しチェックポイントが`logs/train/jvs_fast/runs/<run_dir>/checkpoints/`に存在することが前提条件。

### MASベースライン数値（比較対象）

| 指標 | MAS 500ep | MAS 2500ep | M5目標 |
|------|-----------|------------|--------|
| 退化率（phoneme 80%以上が1フレーム以下） | 39.2% | 43.0% | **0%** |
| 全phonemeの1フレーム以下率 | - | 62.9% | **< 10%** |
| 全phonemeの2フレーム以下率 | - | 73.8% | **< 30%** |
| phoneme median duration（退化サンプル） | - | 1.0フレーム | N/A（退化なし） |
| blank[0] duration平均（退化サンプル） | - | 101フレーム | **< 10フレーム** |
| encoder mu_x norm（phoneme） | - | 2.06-2.26（退化時） | **> 5.0** |

### 完了条件

1. **退化率 = 0%**: 全評価サンプルでphoneme 80%以上が1フレーム以下となるサンプルが存在しない
2. **Duration精度**: Julius ground truthとの相関係数 > 0.85
3. **MCD**: 目標値は学習後に確定（MASベースラインとの相対改善を確認）
4. **話者類似度**: 同一話者の合成音声間でcosine similarity > 0.85
5. **主観評価**: ABテストでMASベースラインに対して有意な選好（p < 0.05）
6. **評価レポート**: 全メトリクスをまとめたレポートがJSONおよび可視化付きで生成される

### 想定期間

- T-M5-01（推論パイプライン更新・サンプル生成）: 2-3日
- T-M5-02（品質評価メトリクス・ABテスト）: 3-5日
- **合計: 5-8日**（主観評価の被験者確保による変動あり）

### 一から作り直すとしたらの思考

評価パイプライン全体を再構築する場合、以下の設計判断を行う:

1. **評価スクリプトの統合**: 現在バラバラに存在する推論コード（`matcha/cli.py`）、duration抽出コード（`matcha/utils/get_durations_from_trained_model.py`）、分析コードを統合し、`scripts/evaluate/`配下に一貫したCLIツール群として配置する
2. **再現性の保証**: 全評価パラメータ（n_timesteps、temperature、length_scale、話者ID一覧、テキスト一覧）をYAML設定ファイルで管理し、同一条件での再実行を保証する
3. **自動化**: サンプル生成からメトリクス計算、レポート出力までをMakefileまたはシェルスクリプト1本で完結させる
4. **MASベースラインの同時生成**: 比較対象のMASモデルからも同一テキスト・話者でサンプルを生成し、ABテストの入力を自動準備する
5. **speaker embeddingの外部モデル**: 話者類似度にはWeSpeaker/ECAPA-TDNNなど事前学習済みの話者検証モデルを使用し、モデル内部のspk_embではなく独立した話者表現で評価する

---

## T-M5-01: 推論パイプライン更新・音声サンプル生成 {#t-m5-01}

### 1. タスク目的とゴール

M4で学習した新モデル（FiLM DP + Julius duration + Blank zero-init）の推論パイプラインを更新し、100話者全員の音声サンプルを系統的に生成する。MASベースラインモデルからも同一条件でサンプルを生成し、T-M5-02での定量・定性評価の入力データを準備する。

**具体的なゴール**:
- 新モデルで推論が正常に動作することを確認
- boundary blank clamping（`matcha_tts.py` L126-129）の要否を判定し、不要であれば除去または条件付き無効化
- 100話者 x 評価テキスト10文 = 1,000サンプル以上の音声を生成
- MASベースラインからも同一条件で1,000サンプルを生成
- 全サンプルのメルスペクトログラム・波形・duration情報を構造化して保存

### 2. 実装する内容の詳細

#### 2.1 推論パイプラインの更新

**対象ファイル**: `matcha/models/matcha_tts.py` の `synthesise()` メソッド（L78-159）

現在の`synthesise()`はMASベースの推論を前提としており、Duration Predictorの出力にboundary blank clamping（L126-129）を適用している:

```python
# 現在のコード（L124-129）
w = torch.exp(logw) * x_mask
for b in range(w.shape[0]):
    seq_len = x_lengths[b].item()
    w[b, 0, 0] = w[b, 0, 0].clamp(max=3.0)           # first blank
    w[b, 0, seq_len - 1] = w[b, 0, seq_len - 1].clamp(max=3.0)  # last blank
```

FiLM DP + Julius durationで学習したモデルでは、Duration Predictorが正確なdurationターゲットで学習されているため、このclampingは不要になる可能性が高い。ただし安全のため以下の段階的アプローチを取る:

1. **clamping有無の両方でサンプル生成**: clamping有り/無しの2セットを生成し、比較する
2. **clampingの条件付き制御**: `synthesise()`に`clamp_boundary_blanks`パラメータを追加（デフォルト`True`で後方互換性を維持）

```python
def synthesise(self, x, x_lengths, n_timesteps, temperature=1.0,
               spks=None, length_scale=1.0, clamp_boundary_blanks=True):
```

3. **結果に基づき判定**: clamping無しで問題なければデフォルトを`False`に変更

**FiLM DPの推論パス確認**:

M3でDuration Predictorに追加されたFiLM層は、`self.encoder(x, x_lengths, spks)`経由で呼ばれる。`synthesise()`ではL121でencoderを呼んでおり、spks引数もL118で`self.spk_emb(spks.long())`変換後に渡されている。したがってFiLM DPの推論パスはコード変更なしで動作するはず。ただし以下を確認:
- `TextEncoder.forward()`がspks引数をDuration Predictorに渡しているか（M3の実装依存）
- spk_emb_dimの整合性（`matcha.yaml`の`spk_emb_dim: 64`がFiLM層と一致するか）

#### 2.2 評価テキストセットの選定

日本語評価用テキスト10文を選定する。以下の基準:

- **音素カバレッジ**: 日本語55シンボルの全音素を含む
- **文長のバリエーション**: 短文（5モーラ程度）から長文（50モーラ程度）まで
- **韻律パターン**: 平叙文、疑問文、強調文を含む
- **既知の難所**: 促音（っ）、撥音（ん）、長音（ー）、連続母音を含む

テキストリストを`eval/texts_ja.txt`として保存。JVS評価セットのテキストからの流用も検討する。

#### 2.3 サンプル生成スクリプト

`scripts/generate_eval_samples.py`を新規作成:

```
入力:
  --checkpoint: 学習済みモデルのチェックポイントパス
  --vocoder: HiFi-GANチェックポイントパス
  --text-file: 評価テキストリスト（eval/texts_ja.txt）
  --speakers: 話者ID範囲（デフォルト: 0-99）
  --output-dir: 出力ディレクトリ
  --n-timesteps: ODEステップ数（デフォルト: 5）
  --temperature: ノイズ分散（デフォルト: 0.667）
  --length-scale: 発話速度（デフォルト: 1.0）
  --clamp-boundary-blanks: boundary blank clampingの有無（デフォルト: auto）
  --baseline-checkpoint: MASベースラインモデルのチェックポイントパス（任意）

出力ディレクトリ構造:
  output-dir/
    julius_model/
      spk_000/
        text_00.wav
        text_00.npy          # メルスペクトログラム
        text_00_dur.json     # duration情報（predicted）
      spk_001/
        ...
      spk_099/
        ...
    mas_baseline/            # --baseline-checkpoint指定時
      spk_000/
        ...
    metadata.json            # 生成パラメータ・テキスト一覧
```

**実装の要点**:
- `matcha/cli.py`の`load_matcha()`, `load_vocoder()`, `to_waveform()`を再利用
- `process_text()`で日本語テキスト→音素変換を実行
- 各サンプルでDuration Predictorの出力`w`（`torch.exp(logw)`）も保存（T-M5-02でのduration精度評価に使用）
- `synthesise()`の戻り値`attn`からduration配列を抽出: `durations = attn.squeeze().sum(-1)` でphonemeごとのフレーム数を取得
- torch.compileは生成スクリプトでも適用（`--no-compile`で無効化可能に）
- GPU メモリ効率のため1話者ずつ処理（1話者10文はバッチ処理可能）
- 進捗表示にtqdmを使用（100話者 x 10文 = 1,000サンプル）

#### 2.4 duration情報の抽出・保存

各サンプルの`_dur.json`には以下の情報を記録:

```json
{
  "text": "こんにちは世界",
  "phonemes": ["^", "k", "o", "N", "n", "i", "ch", "i", "w", "a", "s", "e", "k", "a", "i", "$"],
  "phonemes_with_blanks": ["_", "^", "_", "k", "_", "o", "_", ...],
  "predicted_durations": [1, 3, 1, 5, 1, 4, 1, ...],
  "total_frames": 142,
  "speaker_id": 0,
  "n_timesteps": 5,
  "temperature": 0.667,
  "length_scale": 1.0,
  "rtf": 0.0123,
  "clamp_boundary_blanks": false
}
```

#### 2.5 サンプル品質の目視・目聴確認

生成完了後、以下の初期チェックを実施:
- 10話者 x 3文 = 30サンプルをランダム選択して聴取
- メルスペクトログラムの目視確認（blank位置の異常集中がないこと）
- duration分布のヒストグラム生成（全1,000サンプルのphoneme duration分布）
- blank[0] durationの統計（平均・中央値・最大値）

### 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当内容 |
|------|------|----------|
| TTS推論エンジニア | 1名 | `synthesise()`の更新、clamping制御パラメータ追加、FiLM DP推論パスの動作確認 |
| 評価パイプラインエンジニア | 1名 | `scripts/generate_eval_samples.py`の実装、出力構造設計、metadata管理 |
| 音声品質チェッカー | 1名 | 生成サンプルの初期聴取・目視確認、異常サンプルの報告 |

**合計: 3名**（TTS推論エンジニアと評価パイプラインエンジニアは兼任可能、最小2名）

### 4. 提供範囲とテスト項目

#### 提供範囲

- `matcha/models/matcha_tts.py`: `synthesise()`に`clamp_boundary_blanks`パラメータ追加
- `scripts/generate_eval_samples.py`: 評価サンプル一括生成スクリプト（新規）
- `eval/texts_ja.txt`: 日本語評価テキストリスト（新規）
- `eval/samples/`: 生成サンプル出力ディレクトリ（1,000+ wavファイル）

#### テスト項目

| # | テスト内容 | 確認方法 | 合格基準 |
|---|-----------|----------|----------|
| 1 | 新モデルの推論動作 | 1話者1文で`synthesise()`実行 | エラーなく.wavが生成される |
| 2 | FiLM DPの推論パス | spk引数有無でduration出力を比較 | 話者IDによりdurationが変化する |
| 3 | clamping有無の比較 | 同一テキスト・話者でclamping有/無 | clamping無しでblank[0]が妥当な値（< 10フレーム） |
| 4 | 100話者の生成完走 | 全100話者 x 10文を生成 | 1,000ファイルが欠損なく生成される |
| 5 | MASベースラインの生成 | ベースラインモデルで同一条件生成 | 1,000ファイルが欠損なく生成される |
| 6 | duration JSONの整合性 | phoneme列長とduration配列長の一致 | 全サンプルで`len(phonemes_with_blanks) == len(predicted_durations)` |
| 7 | RTFの妥当性 | 全サンプルのRTF統計 | RTF < 0.1（GPU推論時） |
| 8 | metadata.jsonの完全性 | 生成パラメータの記録確認 | 全パラメータが記録され再現可能 |
| 9 | 既存テストの非破壊 | `make test` | 既存256テストが全パス |

### 5. 懸念事項とレビュー項目

#### 懸念事項

1. **FiLM DPの推論時の話者条件付け**: M3の実装がTextEncoderのforward内部でDuration Predictorにspksを渡す方式の場合、`synthesise()`側の変更は不要。しかしM3が新しいAPIを導入している場合は合わせる必要がある。M3チケットの実装詳細を事前確認すること

2. **boundary blank clampingの除去判断**: Julius durationで学習してもDuration Predictorの予測精度が完全ではない可能性がある。clampingを即座に除去せず、まずclamping有無の比較データを収集してから判断する

3. **MASベースラインチェックポイントの可用性**: 比較用のMASベースラインモデルのチェックポイントが`logs/`配下に残存しているか確認が必要。存在しない場合、MASベースラインとの比較はT-M5-02のduration統計のみで行い、ABテストは新モデル単体の評価とする

4. **HiFi-GANボコーダの品質上限**: HiFi-GAN自体がボトルネックとなりMCDやMOSの上限を規定する可能性がある。ボコーダ品質の分離評価（ground truth mel → HiFi-GAN → 波形のMCD）も実施すべき

5. **torch.compileの互換性**: M3でDuration Predictorの構造が変わった場合、torch.compileのキャッシュが無効化され初回推論が遅くなる。`--no-compile`オプションを用意しておく

#### レビュー項目

- [ ] `synthesise()`の`clamp_boundary_blanks`パラメータが後方互換性を維持しているか（デフォルト`True`）
- [ ] 生成スクリプトが`matcha/cli.py`の既存関数を適切に再利用しているか（コード重複の最小化）
- [ ] duration JSON形式がT-M5-02の評価スクリプトが期待する入力形式と一致しているか
- [ ] 出力ディレクトリ構造がディスク容量の制約内か（1,000 wav x 約200KB = 約200MB + メル約500MB）
- [ ] GPU メモリリークがないか（1,000サンプル生成中にOOMが発生しないこと）

### 6. 一から作り直すとしたら

推論パイプラインを設計し直すなら:

1. **`synthesise()`のリファクタリング**: 現在のfor loopによるbatch内iterationは非効率。clampingをベクトル化するか、完全に除去できる設計にする。また`synthesise()`の戻り値にduration配列を明示的に含める（現在は`attn`から間接的に取得する必要がある）

2. **設定ベースの評価実行**: `eval/config.yaml`のような設定ファイルで全パラメータを管理し、`python scripts/evaluate.py --config eval/config.yaml`で全工程（サンプル生成 → メトリクス計算 → レポート出力）を一括実行できるようにする

3. **ストリーミング生成**: 1,000サンプルの一括生成ではなく、1サンプルずつ生成→即座にメトリクス計算のパイプラインにすれば、ディスク使用量を削減でき、中断・再開も容易になる

4. **Ground truthメルとの直接比較**: 推論時にground truthメル（JVS元音声から計算）を同時にロードし、サンプルごとのMCDをリアルタイムで算出できるようにする

### 7. 後続タスクへの連絡事項

**T-M5-02への引き継ぎ**:

1. **出力ディレクトリのパス**: `eval/samples/julius_model/`および`eval/samples/mas_baseline/`を前提とする
2. **duration JSONの形式**: 上記2.4で定義した形式に準拠。特に`predicted_durations`はintersperse後のblank含む全位置のフレーム数
3. **clamping比較の結果**: clamping有無でどの程度差があるかを報告。T-M5-02でのduration精度評価の解釈に影響する
4. **異常サンプルリスト**: 初期聴取で検出した異常サンプル（無音・ノイズ・不明瞭な発音）のリストを提供
5. **metadata.json**: 生成時の全パラメータが記録されている。再現実験に使用可能

**プロジェクト全体への連絡**:

- boundary blank clampingの除去判断結果は`CLAUDE.md`の「JVS日本語学習の重要な知見」セクションに追記すること
- 新モデルの推奨推論パラメータ（n_timesteps、temperature、length_scale）が確定したら`matcha/cli.py`のデフォルト値を更新すること

---

## T-M5-02: 品質評価メトリクス・ABテスト {#t-m5-02}

### 1. タスク目的とゴール

T-M5-01で生成したサンプルに対して、定量メトリクス（duration精度、退化率、MCD、話者類似度）と定性評価（ABテスト）を実施し、外部アライナーアプローチの有効性を数値で立証する。最終的な評価レポートを生成し、MAS退化問題が解決されたことを結論づける。

**具体的なゴール**:
- 退化率 = 0% を数値で確認
- Duration精度（Julius ground truthとの相関）を定量評価
- MCDによる合成音声品質の定量評価
- 100話者の話者同一性保持を確認
- ABテストでMASベースラインに対する有意な改善を確認
- 全メトリクスを統合した評価レポートを自動生成

### 2. 実装する内容の詳細

#### 2.1 Duration精度評価スクリプト

`scripts/eval_duration_accuracy.py`を新規作成:

**入力**:
- T-M5-01で生成したduration JSON（`eval/samples/julius_model/spk_*/text_*_dur.json`）
- Julius ground truth duration（M1で生成した`.lab`ファイルからフレーム単位に変換したもの、または`data/jvs_precomputed/`内の`.pt`ファイルに格納されたduration配列）

**評価指標**:

| 指標 | 定義 | 目標値 |
|------|------|--------|
| Pearson相関係数 | predicted duration vs ground truth durationの相関 | > 0.85 |
| RMSE | フレーム単位のRoot Mean Square Error | < 3.0フレーム |
| 相対誤差（%） | \|pred - gt\| / gt の平均 | < 30% |
| blank duration精度 | blank位置のduration予測精度 | blank[0]平均 < 10フレーム |

**実装の要点**:
- phoneme-levelでの比較: intersperse後のblank含む全位置で比較
- blank位置とphoneme位置を分離して統計を出力（blankの精度とphonemeの精度は分けて評価）
- 話者ごとの精度を出力（100話者間の分散を確認）
- 文長との相関を分析（短文 vs 長文での精度差）
- 可視化: predicted vs ground truthの散布図、話者別精度のbox plot

#### 2.2 退化率計算スクリプト

`scripts/eval_degeneration_rate.py`を新規作成:

**退化の定義**（CLAUDE.mdと同一基準）:
- **サンプルレベル退化**: あるサンプルにおいてphoneme（blank除く）の80%以上が1フレーム以下
- **全体退化率**: 全評価サンプル中の退化サンプルの割合

**計算手順**:
1. T-M5-01のduration JSONから各サンプルのphonemeごとのduration配列を読み込み
2. blank位置（偶数インデックス: 0, 2, 4, ...）を除外し、phoneme位置のdurationのみ抽出
3. phonemeのうちduration <= 1フレームの割合を計算
4. 80%以上が1フレーム以下ならそのサンプルを「退化」と判定
5. 全サンプルでの退化率を算出

**追加の退化指標**:

| 指標 | MASベースライン | 目標値 |
|------|----------------|--------|
| サンプルレベル退化率 | 39-43% | **0%** |
| 全phonemeの1フレーム以下率 | 62.9% | **< 10%** |
| 全phonemeの2フレーム以下率 | 73.8% | **< 30%** |
| phoneme median duration | 1.0（退化）/ 2.0（正常） | **> 3.0** |
| blank[0] duration平均 | 101（退化）/ 4（正常） | **< 10** |

**出力形式**:
```json
{
  "degeneration_rate": 0.0,
  "total_samples": 1000,
  "degenerate_samples": 0,
  "phoneme_le1_frame_rate": 0.05,
  "phoneme_le2_frame_rate": 0.18,
  "phoneme_median_duration": 4.2,
  "blank0_duration_mean": 2.1,
  "blank0_duration_max": 8,
  "per_speaker_stats": {
    "spk_000": {"degeneration_rate": 0.0, "phoneme_median_duration": 4.5},
    ...
  }
}
```

#### 2.3 MCD（Mel Cepstral Distortion）計算

`scripts/eval_mcd.py`を新規作成:

**MCDの定義**:
```
MCD [dB] = (10 * sqrt(2) / ln(10)) * mean(||MFCC_synth - MFCC_ref||_2)
```

ここでMFCCはメルスペクトログラムからDCTで変換した最初の13次元（0次除く）のケプストラム係数。

**入力**:
- 合成音声のメルスペクトログラム（`eval/samples/julius_model/spk_*/text_*.npy`）
- 参照音声のメルスペクトログラム（JVS元音声から同一パラメータで計算）

**実装の要点**:
- メルスペクトログラムのアライメント: DTW（Dynamic Time Warping）で合成メルと参照メルのフレームを対応付け
- DTWにはscipy.spatial.distance.cdistとscipy.sparse.csgraph.minimum_spanning_treeまたはfastdtw等を使用
- MCD計算にはlibrosa.feature.mfccまたはscipy.fft.dctを使用
- ボコーダ品質の分離: ground truthメル → HiFi-GAN → 波形 → メル再計算 → MCDを「ボコーダ品質の下限」として報告
- 話者ごとのMCD平均・標準偏差を出力

**参考MCD値の目安**:

| システム | MCD [dB] |
|----------|----------|
| HiFi-GAN再合成（品質上限） | 1.5-2.5 |
| 高品質TTS | 3.0-5.0 |
| 標準的TTS | 5.0-7.0 |
| 低品質TTS | 7.0+ |

#### 2.4 話者類似度評価

`scripts/eval_speaker_similarity.py`を新規作成:

**手法**: 事前学習済みの話者検証モデルで話者埋め込みを抽出し、cosine similarityで評価

**話者検証モデルの選択肢**:

| モデル | フレームワーク | 精度 | 推奨度 |
|--------|--------------|------|--------|
| WeSpeaker ECAPA-TDNN | wespeaker | EER 1.01% (VoxCeleb) | 高（日本語対応確認済み） |
| SpeechBrain ECAPA-TDNN | speechbrain | EER 1.12% (VoxCeleb) | 高 |
| Resemblyzer | resemblyzer | GE2E | 中（軽量だが精度劣る） |

**評価指標**:

| 指標 | 定義 | 目標値 |
|------|------|--------|
| 同一話者内cos sim | 同一話者の合成音声10文間のcosine similarity平均 | > 0.85 |
| 合成-参照cos sim | 合成音声とJVS元音声のcosine similarity | > 0.75 |
| 異話者間cos sim | 異なる話者の合成音声間のcosine similarity平均 | < 0.5 |
| EER | Equal Error Rate（話者検証タスクとしてのEER） | < 10% |

**実装の要点**:
- 各話者の参照音声として、JVS validation setから3発話のembeddingを平均して参照ベクトルとする
- 合成音声10文のembeddingを抽出し、参照ベクトルとのcosine similarityを計算
- 100話者の類似度行列（100x100）を可視化（ヒートマップ）: 対角線が高く、非対角が低いことを確認
- 外れ値話者（cosine similarity < 0.6）のリストを出力

#### 2.5 ABテスト設計・実施

**テスト設計**:

- **比較条件**: A = 新モデル（Julius duration + FiLM DP）、B = MASベースラインモデル
- **評価軸**: 自然性（どちらがより自然に聞こえるか）、明瞭性（どちらが聞き取りやすいか）
- **テストサンプル数**: 20文 x 5話者 = 100ペア（各話者はランダム選択）
- **評価者数**: 最低5名（日本語ネイティブ）
- **テスト形式**: 各ペアに対してA/B/同等の3択
- **ランダム化**: A/Bの提示順をランダム化（左右バイアス除去）

**AB テストツールの実装**:

`scripts/prepare_ab_test.py`を新規作成:
- 評価サンプルペアの自動選定（話者・テキストの分散を確保）
- 提示順のランダム化とマッピングファイル生成
- Webベースのリスニングインターフェース（Gradio使用、`matcha-tts-app`の拡張）は任意

`scripts/analyze_ab_test.py`を新規作成:
- 評価結果CSVの読み込み
- 各条件の選好率計算
- 二項検定（scipy.stats.binom_test）による有意差検定（p < 0.05）
- 95%信頼区間の算出
- 評価者間一致度（Fleiss' kappa）の計算

**ABテストが実施できない場合の代替**:

被験者確保が困難な場合、以下の代替指標で定性評価を補完:
- UTMOS（UTokyo-SaruLab MOS predictor）: 事前学習済みMOS予測モデルによる自動MOS推定
- PESQ/POLQA: 客観音声品質指標（参照音声が必要）
- 人手による10サンプル聴取レポート（最低限）

#### 2.6 統合評価レポート生成

`scripts/generate_eval_report.py`を新規作成:

**レポート内容**:

1. **エグゼクティブサマリー**: 退化率0%達成の可否、MASベースラインとの比較結論
2. **Duration精度**: 相関係数、RMSE、散布図
3. **退化率**: 新モデル vs MASベースライン、話者別統計
4. **MCD**: 全体・話者別、ボコーダ品質の下限との比較
5. **話者類似度**: 類似度行列ヒートマップ、外れ値話者の分析
6. **ABテスト結果**: 選好率、有意差検定結果
7. **推奨パラメータ**: 最適なn_timesteps、temperature、length_scaleの推奨値

**出力形式**:
- `eval/report/eval_report.json`: 全メトリクスの構造化データ
- `eval/report/figures/`: 全グラフ画像（PNG）
  - `duration_scatter.png`: predicted vs ground truth duration
  - `degeneration_comparison.png`: 新モデル vs MASの退化率比較
  - `mcd_per_speaker.png`: 話者別MCD box plot
  - `speaker_similarity_heatmap.png`: 100x100類似度行列
  - `ab_test_results.png`: ABテスト選好率

### 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当内容 |
|------|------|----------|
| 評価メトリクスエンジニア | 1名 | duration精度、退化率、MCDの実装 |
| 話者検証エンジニア | 1名 | 話者埋め込み抽出、類似度計算、ヒートマップ生成 |
| ABテスト設計者 | 1名 | テスト設計、サンプル選定、統計検定、レポート生成 |
| リスニングテスト評価者 | 5名 | ABテストへの参加（日本語ネイティブ） |

**合計: 3名（実装）+ 5名（評価者）**（評価メトリクスエンジニアとABテスト設計者は兼任可能、実装最小2名）

### 4. 提供範囲とテスト項目

#### 提供範囲

- `scripts/eval_duration_accuracy.py`: Duration精度評価スクリプト（新規）
- `scripts/eval_degeneration_rate.py`: 退化率計算スクリプト（新規）
- `scripts/eval_mcd.py`: MCD計算スクリプト（新規）
- `scripts/eval_speaker_similarity.py`: 話者類似度評価スクリプト（新規）
- `scripts/prepare_ab_test.py`: ABテストサンプル準備スクリプト（新規）
- `scripts/analyze_ab_test.py`: ABテスト結果分析スクリプト（新規）
- `scripts/generate_eval_report.py`: 統合レポート生成スクリプト（新規）
- `eval/report/`: 評価レポート出力ディレクトリ

#### テスト項目

| # | テスト内容 | 確認方法 | 合格基準 |
|---|-----------|----------|----------|
| 1 | 退化率 = 0% | `eval_degeneration_rate.py`実行 | `degeneration_rate == 0.0` |
| 2 | 全phonemeの1フレーム以下率 | 同上 | `phoneme_le1_frame_rate < 0.10` |
| 3 | 全phonemeの2フレーム以下率 | 同上 | `phoneme_le2_frame_rate < 0.30` |
| 4 | phoneme median duration | 同上 | `phoneme_median_duration > 3.0` |
| 5 | blank[0] duration平均 | 同上 | `blank0_duration_mean < 10.0` |
| 6 | Duration相関係数 | `eval_duration_accuracy.py`実行 | Pearson r > 0.85 |
| 7 | Duration RMSE | 同上 | RMSE < 3.0フレーム |
| 8 | MCD全体平均 | `eval_mcd.py`実行 | MCD < 7.0 dB（絶対値）、MASベースラインより改善 |
| 9 | 同一話者cos sim | `eval_speaker_similarity.py`実行 | 平均 > 0.85 |
| 10 | 合成-参照cos sim | 同上 | 平均 > 0.75 |
| 11 | 異話者間cos sim | 同上 | 平均 < 0.5 |
| 12 | ABテスト選好率 | `analyze_ab_test.py`実行 | 新モデル選好率 > 60%（p < 0.05） |
| 13 | レポート生成 | `generate_eval_report.py`実行 | JSON + 全グラフが出力される |
| 14 | MASベースラインとの退化率比較 | 退化率レポート内の比較 | 新モデル0% vs MAS 39-43%が明記される |
| 15 | 各スクリプトの独立実行 | 個別に実行テスト | 各スクリプトが単独で動作する（他スクリプトへの暗黙の依存なし） |

### 5. 懸念事項とレビュー項目

#### 懸念事項

1. **Julius ground truth durationの取得**: T-M5-02でduration精度を評価するには、評価テキストに対するJulius ground truthが必要。しかし評価テキストはT-M5-01で新規選定するため、M1で処理済みのJVS発話とは異なる。以下のいずれかで対処:
   - **方法A**: 評価テキストをJVS validation setのテキストから選定し、M1で既に生成済みのdurationを使用（推奨）
   - **方法B**: 評価テキストに対してJulius alignmentを追加実行
   - **方法C**: duration精度はJVS validation setの事前計算済みduration（`.pt`ファイル内）で評価し、評価テキストは退化率・MCD・話者類似度のみに使用

2. **MCDのDTWアライメント精度**: 合成音声と参照音声の長さが大きく異なる場合、DTWが不正確なアライメントを生成しMCDが過大評価される可能性。length_scale=1.0でduration精度が高ければこの問題は軽微だが、外れ値チェックを入れる

3. **話者検証モデルの日本語性能**: VoxCelebで学習された話者検証モデルは英語話者に最適化されており、日本語話者での性能が低下する可能性がある。日本語で微調整されたモデル（例: NII JTubeSpeech pretrained）の使用を検討

4. **ABテストの評価者バイアス**: 評価者が少数（5名）の場合、個人の嗜好がバイアスとなる。Fleiss' kappaで評価者間一致度を確認し、kappa < 0.4の場合はABテスト結果の信頼性に注意を記載

5. **MASベースラインチェックポイントの品質**: MASベースラインモデルが2500epまで学習されている場合、退化率43%とはいえ正常サンプル57%は比較的高品質の可能性がある。ABテストではMASベースラインの正常サンプルとの比較が最も厳しいテストとなる

6. **依存ライブラリの追加**: MCDにfastdtw/scipy、話者類似度にwespeaker/speechbrainが必要。`pyproject.toml`の`[project.optional-dependencies]`に`eval`グループを追加する必要がある

#### レビュー項目

- [ ] 退化率の判定基準がCLAUDE.mdの定義（phoneme 80%以上が1フレーム以下）と完全に一致しているか
- [ ] MCD計算でDCTの次元数（13次元、0次除く）が標準的な定義と一致しているか
- [ ] DTWアライメントでframeの端が切り捨てられていないか（先頭・末尾の無音領域の処理）
- [ ] 話者類似度のcosine similarityが-1から1の範囲で正規化されているか
- [ ] ABテストの統計検定が片側検定ではなく両側検定で実施されているか
- [ ] 評価レポートのJSON形式がマシンパーサブルか（後続の自動化パイプラインで使用可能か）
- [ ] 各評価スクリプトのCLIインターフェースが一貫しているか（共通のargparse規約）

### 6. 一から作り直すとしたら

品質評価パイプラインを再設計するなら:

1. **統合評価フレームワーク**: 個別スクリプト7本ではなく、`EvaluationPipeline`クラスを設計し、メトリクスをプラグインとして追加可能にする。`pipeline.add_metric("mcd", MCDMetric())`のようなAPIで拡張性を確保

2. **UTMOS/自動MOSの優先**: ABテストは被験者確保のコストが高い。UTMOS（https://github.com/sarulab-speech/UTMOS22）のような自動MOS予測モデルを第一の主観品質指標とし、ABテストは最終確認のみに限定する

3. **CI/CD統合**: 評価メトリクスの一部（退化率、duration精度）をGitHub Actions/CIに組み込み、チェックポイント更新時に自動で回帰テストを実行する仕組みにする

4. **可視化ダッシュボード**: 静的なPNG画像ではなく、Streamlit/Gradioベースのインタラクティブダッシュボードで評価結果を閲覧可能にする。話者・テキスト・メトリクスのフィルタリングが可能

5. **A/B/Xテスト**: A/B 2択ではなくA/B/X形式（Xは隠された参照）を採用し、評価者の識別能力をキャリブレーションする

6. **話者クラスタ分析**: 100話者を話者特性（F0レンジ、話速、音声品質）でクラスタリングし、クラスタごとの品質傾向を分析する。特定の話者タイプ（例: 高ピッチ女性、低ピッチ男性）で品質差がないかを確認

### 7. 後続タスクへの連絡事項

**プロジェクト全体への報告事項**:

1. **退化率0%の達成確認**: 達成した場合、CLAUDE.mdの「MASアライメント退化問題」セクションに解決済みの旨を追記。未達の場合は原因分析と追加対策を記載

2. **推奨推論パラメータ**: 評価結果に基づき確定した最適パラメータを以下に反映:
   - `matcha/cli.py`のデフォルト値（`temperature`, `steps`, `speaking_rate`）
   - `CLAUDE.md`の「推論」セクション
   - boundary blank clampingの要否

3. **HiFi-GANボコーダの品質限界**: ボコーダ品質の下限MCD値を報告。ボコーダがボトルネックであることが判明した場合、ボコーダの再学習または別ボコーダ（BigVGAN等）への移行を検討事項として提起

4. **話者品質の分散**: 100話者間の品質分散が大きい場合（例: 特定話者でMCDが著しく高い）、該当話者のJVS元データの品質調査が必要。学習データの問題か、モデルの問題かを切り分ける情報を提供

5. **評価レポートの保管**: `eval/report/eval_report.json`を成果物として保管し、今後のモデル更新時の比較基準（ベースライン）とする

6. **Style-BERT-VITS2との比較**: MOS 4.37が参考値として記載されているが、直接比較は音声合成条件（テキスト・話者・ボコーダ）が異なるため不適切。自動MOS予測（UTMOS）で同一条件の比較が可能であれば参考情報として報告

**CLAUDE.md更新項目**:

- 「過去に失敗した最適化」セクション: 外部アライナーアプローチの成功を追記（退化率、MCD、ABテスト結果）
- 「MASアライメント退化問題」セクション: 対策結果の追記
- 「主要パラメータ」セクション: 推奨推論パラメータの更新
- boundary blank clamping（L126-129）の要否の結論を追記
