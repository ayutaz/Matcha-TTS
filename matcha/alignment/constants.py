"""音声処理・アライメントの共通定数。

これらの定数は、メルスペクトログラム計算、Julius forced alignment、
およびduration配列生成で使用される。変更する場合は、全パイプラインへの
影響を確認すること。

設計方針:
- Hydra設定との二重管理を避けるため、パイプラインスクリプト内では
  このモジュールからインポートする
- モデル学習時の設定はconfigs/で管理（Hydra経由）
- この定数モジュールはスクリプト側の「デフォルト値」として機能
"""

# === メルスペクトログラム計算パラメータ ===
SAMPLE_RATE: int = 22050
"""Matcha-TTSのメルスペクトログラム計算サンプリングレート (Hz)"""

HOP_LENGTH: int = 256
"""メルスペクトログラムのホップ長 (samples)"""

N_FFT: int = 1024
"""FFTウィンドウサイズ (samples)"""

N_FEATS: int = 80
"""メルビン数"""

# === Julius forced aligner パラメータ ===
JULIUS_SAMPLE_RATE: int = 16000
"""Julius segmentation-kitの入力サンプリングレート (Hz)"""

JULIUS_TIME_UNIT: int = 10_000_000
"""Julius .labファイルの時間単位。1秒 = 10^7 (100ns単位)"""

# === Duration配列パラメータ ===
FRAME_ADJUSTMENT_WARN_THRESHOLD: int = 10
"""端数調整量がこの値以上の場合に警告を出す (frames)"""

# === JVSデータセット ===
JVS_MEL_MEAN: float = -6.550095
"""JVS（トリミング済み）のメル平均値"""

JVS_MEL_STD: float = 2.383771
"""JVS（トリミング済み）のメル標準偏差"""

# === 日本語音素 ===
N_VOCAB_JA: int = 55
"""日本語音素語彙サイズ（pyopenjtalk 55シンボル）"""

N_VOCAB_EN: int = 178
"""英語音素語彙サイズ"""
