"""from https://github.com/keithito/tacotron

Cleaners are transformations that run over the input text at both training and eval time.

Cleaners can be selected by passing a comma-delimited list of cleaner names as the "cleaners"
hyperparameter. Some cleaners are English-specific. You'll typically want to use:
  1. "english_cleaners" for English text
  2. "transliteration_cleaners" for non-English text that can be transliterated to ASCII using
     the Unidecode library (https://pypi.python.org/pypi/Unidecode)
  3. "basic_cleaners" if you do not want to transliterate (in this case, you should also update
     the symbols in symbols.py to match your data).
"""

import logging
import re

from unidecode import unidecode

# Lazy-initialized espeak phonemizer (only created when english_cleaners2 is called)
_global_phonemizer = None


def _get_phonemizer():
    global _global_phonemizer
    if _global_phonemizer is None:
        import phonemizer

        critical_logger = logging.getLogger("phonemizer")
        critical_logger.setLevel(logging.CRITICAL)
        _global_phonemizer = phonemizer.backend.EspeakBackend(
            language="en-us",
            preserve_punctuation=True,
            with_stress=True,
            language_switch="remove-flags",
            logger=critical_logger,
        )
    return _global_phonemizer


# Regular expression matching whitespace:
_whitespace_re = re.compile(r"\s+")

# Remove brackets
_brackets_re = re.compile(r"[\[\]\(\)\{\}]")

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [
    (re.compile(f"\\b{x[0]}\\.", re.IGNORECASE), x[1])
    for x in [
        ("mrs", "misess"),
        ("mr", "mister"),
        ("dr", "doctor"),
        ("st", "saint"),
        ("co", "company"),
        ("jr", "junior"),
        ("maj", "major"),
        ("gen", "general"),
        ("drs", "doctors"),
        ("rev", "reverend"),
        ("lt", "lieutenant"),
        ("hon", "honorable"),
        ("sgt", "sergeant"),
        ("capt", "captain"),
        ("esq", "esquire"),
        ("ltd", "limited"),
        ("col", "colonel"),
        ("ft", "fort"),
    ]
]


def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


def lowercase(text):
    return text.lower()


def remove_brackets(text):
    return re.sub(_brackets_re, "", text)


def collapse_whitespace(text):
    return re.sub(_whitespace_re, " ", text)


def convert_to_ascii(text):
    return unidecode(text)


def basic_cleaners(text):
    """Basic pipeline that lowercases and collapses whitespace without transliteration."""
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def transliteration_cleaners(text):
    """Pipeline for non-English text that transliterates to ASCII."""
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def english_cleaners2(text):
    """Pipeline for English text, including abbreviation expansion. + punctuation + stress"""
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_abbreviations(text)
    phonemes = _get_phonemizer().phonemize([text], strip=True, njobs=1)[0]
    # Added in some cases espeak is not removing brackets
    phonemes = remove_brackets(phonemes)
    phonemes = collapse_whitespace(phonemes)
    return phonemes


def ipa_simplifier(text):
    replacements = [
        ("ɐ", "ə"),
        ("ˈə", "ə"),
        ("ʤ", "dʒ"),
        ("ʧ", "tʃ"),
        ("ᵻ", "ɪ"),
    ]
    for replacement in replacements:
        text = text.replace(replacement[0], replacement[1])
    phonemes = collapse_whitespace(text)
    return phonemes


# I am removing this due to incompatibility with several version of python
# However, if you want to use it, you can uncomment it
# and install piper-phonemize with the following command:
# pip install piper-phonemize

# import piper_phonemize
# def english_cleaners_piper(text):
#     """Pipeline for English text, including abbreviation expansion. + punctuation + stress"""
#     text = convert_to_ascii(text)
#     text = lowercase(text)
#     text = expand_abbreviations(text)
#     phonemes = "".join(piper_phonemize.phonemize_espeak(text=text, voice="en-US")[0])
#     phonemes = collapse_whitespace(phonemes)
#     return phonemes


# === Japanese cleaners (pyopenjtalk + ttslearn-compatible prosody) ===

_PHONEME_RE = re.compile(r"-([^+]+)\+")
_A1_RE = re.compile(r"/A:([0-9-]+)\+")
_A2_RE = re.compile(r"\+(\d+)\+")
_F1_RE = re.compile(r"/F:(\d+)_")


def _fullcontext_to_prosody(labels):
    """Convert HTS full-context labels to prosody-annotated phoneme sequence.

    This follows the ttslearn ``pp_symbols`` convention:
      - ``^`` / ``$`` / ``?`` : utterance start / end / pause-end
      - ``_`` : pause (pau)
      - ``#`` : accent phrase boundary
      - ``[`` / ``]`` : pitch rise / fall
    """
    phonemes = []
    for i, label in enumerate(labels):
        # Extract phoneme (p3 field)
        m = _PHONEME_RE.search(label)
        if m is None:
            continue
        ph = m.group(1)

        # Handle silence / pause
        if ph == "sil":
            # First sil -> ^, last sil -> $, others -> ? (shouldn't appear normally)
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

        # Extract accent features
        a1_m = _A1_RE.search(label)
        a2_m = _A2_RE.search(label)
        f1_m = _F1_RE.search(label)
        if a1_m is None or a2_m is None or f1_m is None:
            phonemes.append(ph)
            continue
        a1 = int(a1_m.group(1))  # mora position in accent phrase
        a2 = int(a2_m.group(1))  # accent nucleus position
        f1 = int(f1_m.group(1))  # mora count in accent phrase

        # Accent phrase boundary
        a2_next_m = _A2_RE.search(labels[i + 1]) if i + 1 < len(labels) else None
        a2_next = int(a2_next_m.group(1)) if a2_next_m is not None else -1
        if a1 == 1 and a2_next == 1:
            phonemes.append("#")

        # Pitch rise / fall markers
        if a2 == 1 and a1 == 1:
            phonemes.append("[")
        elif a1 == a2 + 1 and a2 != f1:
            phonemes.append("]")

        phonemes.append(ph)

    return phonemes


def japanese_cleaners(text):
    """Pipeline for Japanese text using pyopenjtalk (lazy-imported).

    Returns a **space-separated** phoneme string with prosody markers,
    compatible with ``symbols_ja``.
    """
    import pyopenjtalk  # lazy import — only needed when language="ja"

    labels = pyopenjtalk.extract_fullcontext(text)
    phonemes = _fullcontext_to_prosody(labels)
    return " ".join(phonemes)
