"""Julius音素セットからpyopenjtalk 55シンボルへのマッピング。

Julius forced aligner (segmentation-kit) が出力する音素ラベルを、
Matcha-TTS の日本語語彙テーブル (symbols_ja, 55シンボル) に変換する。

主な差異:
- Julius の silB/silE → pyopenjtalk の "sil"
- Julius の sp → pyopenjtalk の "pau"
- 無声化母音 (A,I,U,E,O) は Julius では区別されない (T-M1-03で対応)
- 韻律記号 (^,$,?,_,#,[,]) は Julius では生成されない
"""

from matcha.text.symbols import symbols_ja

_VALID_PYOPENJTALK_SYMBOLS = set(symbols_ja)

# Julius音素 → pyopenjtalk音素のマッピング辞書
JULIUS_TO_PYOPENJTALK: dict[str, str] = {
    # === 母音 (5) ===
    "a": "a",
    "i": "i",
    "u": "u",
    "e": "e",
    "o": "o",
    # === 基本子音 (16) ===
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
    # === 拗音・複合子音 (17) ===
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
    # === 特殊音素 (3) ===
    "N": "N",   # 撥音
    "cl": "cl",  # 促音
    "pau": "pau",  # ポーズ
    # === Julius固有の無音・ポーズ (4) ===
    "silB": "sil",  # 発話先頭の無音
    "silE": "sil",  # 発話末尾の無音
    "sp": "pau",    # 短いポーズ
    "sil": "sil",   # 一般的な無音
}

# 韻律記号（pyopenjtalkにあるがJuliusにはない）
PROSODY_SYMBOLS = {"^", "$", "?", "_", "#", "[", "]"}

# マッピングテーブルの整合性検証: 全バリューが symbols_ja に含まれること
for _julius_ph, _pyopenjtalk_ph in JULIUS_TO_PYOPENJTALK.items():
    assert _pyopenjtalk_ph in _VALID_PYOPENJTALK_SYMBOLS, (
        f"Mapping value '{_pyopenjtalk_ph}' (from Julius '{_julius_ph}') "
        f"is not in symbols_ja"
    )


def map_julius_phoneme(julius_phoneme: str) -> str:
    """1つのJulius音素をpyopenjtalk音素に変換。

    Args:
        julius_phoneme: Julius forced aligner が出力した音素ラベル。

    Returns:
        pyopenjtalk 55シンボル体系の音素。

    Raises:
        KeyError: マッピングに存在しない音素の場合。
    """
    try:
        return JULIUS_TO_PYOPENJTALK[julius_phoneme]
    except KeyError:
        raise KeyError(
            f"Unknown Julius phoneme: '{julius_phoneme}'. "
            f"Known phonemes: {sorted(JULIUS_TO_PYOPENJTALK.keys())}"
        ) from None


def map_julius_sequence(julius_phonemes: list[str]) -> list[str]:
    """Julius音素列をpyopenjtalk音素列に変換。

    Args:
        julius_phonemes: Julius音素のリスト。

    Returns:
        pyopenjtalk音素のリスト。

    Raises:
        KeyError: マッピングに存在しない音素が含まれる場合。
    """
    return [map_julius_phoneme(ph) for ph in julius_phonemes]


def get_unmapped_phonemes(julius_phonemes: list[str]) -> set[str]:
    """マッピングされていないJulius音素を返す（デバッグ用）。

    Args:
        julius_phonemes: Julius音素のリスト。

    Returns:
        JULIUS_TO_PYOPENJTALK に存在しない音素の集合。
        全音素がマッピング済みなら空集合を返す。
    """
    return {ph for ph in julius_phonemes if ph not in JULIUS_TO_PYOPENJTALK}
