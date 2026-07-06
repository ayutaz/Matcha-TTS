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
    # === 特殊音素 (4) ===
    "N": "N",  # 撥音
    "cl": "cl",  # 促音
    "q": "cl",  # 促音（Juliusは"q"を使用、pyopenjtalkは"cl"）
    "pau": "pau",  # ポーズ
    # === 長母音（Juliusは":"付きで出力する場合がある） (5) ===
    "a:": "a",
    "i:": "i",
    "u:": "u",
    "e:": "e",
    "o:": "o",
    # === Julius固有の無音・ポーズ (4) ===
    "silB": "sil",  # 発話先頭の無音
    "silE": "sil",  # 発話末尾の無音
    "sp": "pau",  # 短いポーズ
    "sil": "sil",  # 一般的な無音
}

# 韻律記号（pyopenjtalkにあるがJuliusにはない）
PROSODY_SYMBOLS = {"^", "$", "?", "_", "#", "[", "]"}

# マッピングテーブルの整合性検証: 全バリューが symbols_ja に含まれること
for _julius_ph, _pyopenjtalk_ph in JULIUS_TO_PYOPENJTALK.items():
    assert _pyopenjtalk_ph in _VALID_PYOPENJTALK_SYMBOLS, (
        f"Mapping value '{_pyopenjtalk_ph}' (from Julius '{_julius_ph}') is not in symbols_ja"
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
            f"Unknown Julius phoneme: '{julius_phoneme}'. Known phonemes: {sorted(JULIUS_TO_PYOPENJTALK.keys())}"
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


# ヴ行カタカナ → バ行の置換テーブル。
# Julius segmentation-kit の yomi2voca.pl は「ゔ」を変換できず（音響モデルにも
# v 音素が無い）、pyopenjtalk-plus 自身も ヴ を b 音素で出力するため、
# Julius 入力のかな生成時にバ行へ正規化して両者を一致させる。
# 2文字の組み合わせを先に置換すること（「ヴ」単体が最後）。
_VU_TO_BA: tuple[tuple[str, str], ...] = (
    ("ヴァ", "バ"),
    ("ヴィ", "ビ"),
    ("ヴャ", "ビャ"),
    ("ヴュ", "ビュ"),
    ("ヴョ", "ビョ"),
    ("ヴェ", "ベ"),
    ("ヴォ", "ボ"),
    ("ヴ", "ブ"),
)


def normalize_vu_kana(kana: str) -> str:
    """カタカナ列のヴ行をバ行へ置換する（Julius forced alignment 入力用）。"""
    for src, dst in _VU_TO_BA:
        kana = kana.replace(src, dst)
    return kana


def get_unmapped_phonemes(julius_phonemes: list[str]) -> set[str]:
    """マッピングされていないJulius音素を返す（デバッグ用）。

    Args:
        julius_phonemes: Julius音素のリスト。

    Returns:
        JULIUS_TO_PYOPENJTALK に存在しない音素の集合。
        全音素がマッピング済みなら空集合を返す。
    """
    return {ph for ph in julius_phonemes if ph not in JULIUS_TO_PYOPENJTALK}
