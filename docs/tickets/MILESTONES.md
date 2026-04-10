# マイルストーン一覧 — MAS退化対策: 外部アライナーによるduration事前計算

## 概要

JVS 100話者学習においてMASアライメントが構造的に退化する問題（39-43%の退化率）を、
Julius forced alignerによるduration事前計算で解決する。併せてDuration PredictorへのFiLM話者条件付けを追加し、
多話者TTSとしての品質を根本的に改善する。

## 依存関係図

```
M1: Julius Alignment ──→ M2: DataModule対応 ──→ M4: 学習実行 ──→ M5: 評価
                                                    ↑
M3: FiLM DP + Blank init ─────────────────────────┘
```

## マイルストーン一覧

| ID | マイルストーン | 依存 | チケット数 | 状態 |
|----|--------------|------|-----------|------|
| M1 | [Julius forced alignmentパイプライン構築](M1_julius_alignment.md) | なし | 4 | 未着手 |
| M2 | [DataModule・前処理スクリプトのduration対応](M2_datamodule_duration.md) | M1 | 3 | 未着手 |
| M3 | [Duration PredictorのFiLM話者条件付け + Blank zero-init](M3_film_dp.md) | なし | 2 | 未着手 |
| M4 | [学習設定変更・段階的学習実行](M4_training.md) | M2, M3 | 2 | 未着手 |
| M5 | [推論・評価・品質検証](M5_evaluation.md) | M4 | 2 | 未着手 |

## チケット一覧（全13チケット）

| チケットID | マイルストーン | タイトル | 状態 |
|-----------|--------------|---------|------|
| T-M1-01 | M1 | [Julius segmentation-kit環境構築・JVSアライメント実行](M1_julius_alignment.md#t-m1-01) | 未着手 |
| T-M1-02 | M1 | [音素マッピングテーブル作成（Julius → pyopenjtalk 55シンボル）](M1_julius_alignment.md#t-m1-02) | 未着手 |
| T-M1-03 | M1 | [.lab → durationフレーム配列変換 + blank intersperse](M1_julius_alignment.md#t-m1-03) | 未着手 |
| T-M1-04 | M1 | [アライメント品質検証・統計分析](M1_julius_alignment.md#t-m1-04) | 未着手 |
| T-M2-01 | M2 | [precompute_dataset.pyへのduration埋め込み機能追加](M2_datamodule_duration.md#t-m2-01) | 未着手 |
| T-M2-02 | M2 | [PrecomputedDataModuleのduration読み込み対応](M2_datamodule_duration.md#t-m2-02) | 未着手 |
| T-M2-03 | M2 | [.ptファイル再生成（duration付き）](M2_datamodule_duration.md#t-m2-03) | 未着手 |
| T-M3-01 | M3 | [DurationPredictorへのFiLM話者条件付け追加](M3_film_dp.md#t-m3-01) | 未着手 |
| T-M3-02 | M3 | [Blank embedding zero-init](M3_film_dp.md#t-m3-02) | 未着手 |
| T-M4-01 | M4 | [学習設定ファイル変更・Hydra設定統合](M4_training.md#t-m4-01) | 未着手 |
| T-M4-02 | M4 | [段階的学習実行・モニタリング](M4_training.md#t-m4-02) | 未着手 |
| T-M5-01 | M5 | [推論パイプライン更新・音声サンプル生成](M5_evaluation.md#t-m5-01) | 未着手 |
| T-M5-02 | M5 | [品質評価メトリクス・ABテスト](M5_evaluation.md#t-m5-02) | 未着手 |

## 環境情報

- **GPU**: 4x Tesla T4 (16GB each)
- **学習フレームワーク**: PyTorch Lightning + Hydra
- **データセット**: JVS ver1 (100 speakers, ~30hr)
- **前回学習結果**:
  - FP32 2500ep: MAS退化率43.0%（学習量増加で改善せず）
  - FP32 500ep: MAS退化率39.2%
