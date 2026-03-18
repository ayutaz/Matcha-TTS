"""from https://github.com/keithito/tacotron

Defines the set of symbols used in text input to the model.
"""

_pad = "_"
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_letters_ipa = (
    "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
)


# Export all symbols:
symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)

# Special symbol ids
SPACE_ID = symbols.index(" ")


# === Japanese symbols (ttslearn-compatible, 52 symbols) ===
_pad_ja = "~"
_extra_symbols_ja = ["^", "$", "?", "_", "#", "[", "]"]  # 韻律記号 (7)
_phonemes_ja = [
    "A", "E", "I", "N", "O", "U",                          # 無声化母音+撥音 (6)
    "a", "b", "by", "ch", "cl", "d", "dy", "e", "f",      # (9)
    "g", "gy", "h", "hy", "i", "j", "k", "ky",            # (8)
    "m", "my", "n", "ny", "o", "p", "py",                  # (7)
    "r", "ry", "s", "sh", "t", "ts", "ty",                 # (7)
    "u", "v", "w", "y", "z",                               # (5)
    "pau", "sil",                                           # (2)
]  # 44音素
symbols_ja = [_pad_ja] + _extra_symbols_ja + _phonemes_ja  # 52
