"""Julius forced alignment パイプラインのコアモジュール。

このパッケージは、Julius forced alignerによるduration事前計算に関連する
モジュールを統合的にアクセスするためのエントリポイントを提供する。

主要コンポーネント:
- phoneme_mapping: Julius音素 → pyopenjtalk 55シンボルのマッピング
- metrics: アライメント品質指標（M1品質検証 + M5学習後評価で共有）
- constants: 音声処理の共通定数

使用例:
    from matcha.alignment import JULIUS_TO_PYOPENJTALK, map_julius_phoneme
    from matcha.alignment import is_degenerate, compute_corpus_stats
    from matcha.alignment.constants import SAMPLE_RATE, HOP_LENGTH
"""

# 音素マッピング（matcha/text/julius_to_pyopenjtalk.pyから再エクスポート）
from matcha.text.julius_to_pyopenjtalk import (
    JULIUS_TO_PYOPENJTALK,
    PROSODY_SYMBOLS,
    get_unmapped_phonemes,
    map_julius_phoneme,
    map_julius_sequence,
)

# アライメント品質指標（matcha/utils/alignment_metrics.pyから再エクスポート）
from matcha.utils.alignment_metrics import (
    compute_corpus_stats,
    compute_duration_stats,
    compute_phoneme_class_stats,
    is_degenerate,
)

__all__ = [
    # phoneme mapping
    "JULIUS_TO_PYOPENJTALK",
    "PROSODY_SYMBOLS",
    "map_julius_phoneme",
    "map_julius_sequence",
    "get_unmapped_phonemes",
    # metrics
    "is_degenerate",
    "compute_duration_stats",
    "compute_corpus_stats",
    "compute_phoneme_class_stats",
]
