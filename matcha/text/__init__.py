"""from https://github.com/keithito/tacotron"""

from functools import lru_cache

from matcha.text import cleaners
from matcha.text.symbols import symbols, symbols_ja

# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}  # pylint: disable=unnecessary-comprehension

# Language-specific symbol maps (lazily populated)
_symbol_maps = {}


def _get_symbol_map(language):
    """Return (symbol_to_id, id_to_symbol) for the given language."""
    if language not in _symbol_maps:
        if language == "en":
            _symbol_maps[language] = (_symbol_to_id, _id_to_symbol)
        elif language == "ja":
            s2i = {s: i for i, s in enumerate(symbols_ja)}
            i2s = {i: s for i, s in enumerate(symbols_ja)}
            _symbol_maps[language] = (s2i, i2s)
        else:
            raise ValueError(f"Unsupported language: {language}")
    return _symbol_maps[language]


class UnknownCleanerException(Exception):
    pass


@lru_cache(maxsize=16384)
def _cached_clean_text(text, cleaners_tuple, language):
    """Cached version of text cleaning for repeated texts."""
    return _clean_text(text, list(cleaners_tuple))


def text_to_sequence(text, cleaner_names, *, language="en"):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through
      language: "en" or "ja"
    Returns:
      List of integers corresponding to the symbols in the text
    """
    sequence = []
    clean_text = _cached_clean_text(text, tuple(cleaner_names), language)

    if language == "ja":
        sym2id, _ = _get_symbol_map("ja")
        for symbol in clean_text.split():
            sequence.append(sym2id[symbol])
    else:
        for symbol in clean_text:
            symbol_id = _symbol_to_id[symbol]
            sequence += [symbol_id]

    return sequence, clean_text


def cleaned_text_to_sequence(cleaned_text, *, language="en"):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
      language: "en" or "ja"
    Returns:
      List of integers corresponding to the symbols in the text
    """
    if language == "ja":
        sym2id, _ = _get_symbol_map("ja")
        return [sym2id[symbol] for symbol in cleaned_text.split()]
    return [_symbol_to_id[symbol] for symbol in cleaned_text]


def sequence_to_text(sequence, *, language="en"):
    """Converts a sequence of IDs back to a string"""
    if language == "ja":
        _, id2sym = _get_symbol_map("ja")
        return " ".join(id2sym[symbol_id] for symbol_id in sequence)
    result = ""
    for symbol_id in sequence:
        s = _id_to_symbol[symbol_id]
        result += s
    return result


def _clean_text(text, cleaner_names):
    for name in cleaner_names:
        cleaner = getattr(cleaners, name)
        if not cleaner:
            raise UnknownCleanerException(f"Unknown cleaner: {name}")
        text = cleaner(text)
    return text
