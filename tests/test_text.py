"""Tests for the Matcha-TTS text processing pipeline.

Covers:
  - Symbol list completeness
  - text_to_sequence / sequence_to_text roundtrip
  - intersperse utility function
  - Individual cleaner helpers (that do not require phonemizer)
"""

import sys
import types

import pytest

# ---------------------------------------------------------------------------
# Mock phonemizer before any matcha.text import, because matcha/text/cleaners.py
# executes `phonemizer.backend.EspeakBackend(...)` at module level and would
# fail without espeak-ng installed.
# ---------------------------------------------------------------------------
_fake_phonemizer = types.ModuleType("phonemizer")
_fake_backend = types.ModuleType("phonemizer.backend")


class _FakeEspeakBackend:
    """Minimal stand-in so cleaners.py can be imported."""

    def __init__(self, **kwargs):
        pass

    def phonemize(self, text_list, strip=True, njobs=1):
        # Return the input unchanged; real phonemisation is not tested here.
        return text_list


_fake_backend.EspeakBackend = _FakeEspeakBackend
_fake_phonemizer.backend = _fake_backend

# Also mock the espeak submodules that the real phonemizer may try to load
_fake_espeak = types.ModuleType("phonemizer.backend.espeak")
_fake_espeak_espeak = types.ModuleType("phonemizer.backend.espeak.espeak")
_fake_backend.espeak = _fake_espeak
_fake_espeak.espeak = _fake_espeak_espeak

# Force-insert mocks so they override any already-imported real modules.
# setdefault would silently keep the real package when it is installed,
# causing failures in environments where phonemizer is present but
# espeak-ng is not.
sys.modules["phonemizer"] = _fake_phonemizer
sys.modules["phonemizer.backend"] = _fake_backend
sys.modules["phonemizer.backend.espeak"] = _fake_espeak
sys.modules["phonemizer.backend.espeak.espeak"] = _fake_espeak_espeak

# Now it is safe to import the text modules ---------------------------------
from matcha.text import (  # noqa: E402
    _id_to_symbol,
    _symbol_to_id,
    cleaned_text_to_sequence,
    sequence_to_text,
    text_to_sequence,
)
from matcha.text.symbols import SPACE_ID, _pad, symbols  # noqa: E402
from matcha.utils.utils import intersperse  # noqa: E402

# ---------------------------------------------------------------------------
# 1. Symbol list completeness
# ---------------------------------------------------------------------------


class TestSymbols:
    """Verify structural properties of the symbol table."""

    def test_symbol_count(self):
        """The symbol list should contain exactly 178 entries."""
        assert len(symbols) == 178

    def test_pad_is_first(self):
        """The padding symbol '_' must be the first symbol (index 0)."""
        assert symbols[0] == _pad
        assert _pad == "_"

    def test_pad_in_symbols(self):
        assert _pad in symbols

    def test_space_in_symbols(self):
        assert " " in symbols

    def test_space_id_correct(self):
        """SPACE_ID must match the actual position of ' ' in the list."""
        assert symbols[SPACE_ID] == " "

    def test_no_duplicate_symbols(self):
        """Symbols should be unique (except _pad which may duplicate another char)."""
        # _pad ('_') may appear both as pad and in punctuation, so mapping has one fewer entry
        unique_count = len(set(symbols))
        assert unique_count >= len(symbols) - 1

    def test_symbol_to_id_mapping_size(self):
        """Forward mapping covers unique symbols, reverse mapping covers all positions."""
        assert len(_symbol_to_id) == len(set(symbols))
        assert len(_id_to_symbol) == len(symbols)

    def test_symbol_to_id_roundtrip(self):
        """symbol -> id -> symbol must be the identity for every symbol."""
        for sym in symbols:
            assert _id_to_symbol[_symbol_to_id[sym]] == sym


# ---------------------------------------------------------------------------
# 2. text_to_sequence / sequence_to_text roundtrip
# ---------------------------------------------------------------------------


class TestTextSequenceConversion:
    """Test encoding and decoding of text through the symbol table."""

    def test_basic_cleaners_roundtrip(self):
        """Lowercase ASCII text should survive a full roundtrip with basic_cleaners."""
        original = "hello world"
        seq, clean = text_to_sequence(original, ["basic_cleaners"])
        decoded = sequence_to_text(seq)
        assert decoded == original

    def test_basic_cleaners_lowercases(self):
        """basic_cleaners should lowercase the input."""
        seq, clean = text_to_sequence("HELLO", ["basic_cleaners"])
        decoded = sequence_to_text(seq)
        assert decoded == "hello"

    def test_basic_cleaners_collapses_whitespace(self):
        seq, clean = text_to_sequence("a  b   c", ["basic_cleaners"])
        decoded = sequence_to_text(seq)
        assert decoded == "a b c"

    def test_transliteration_cleaners_roundtrip(self):
        """transliteration_cleaners converts to ASCII, lowercases, collapses whitespace."""
        seq, clean = text_to_sequence("Café", ["transliteration_cleaners"])
        decoded = sequence_to_text(seq)
        assert decoded == "cafe"

    def test_cleaned_text_to_sequence(self):
        """cleaned_text_to_sequence should agree with text_to_sequence for already-clean text."""
        text = "hello"
        seq_via_clean, _ = text_to_sequence(text, ["basic_cleaners"])
        seq_direct = cleaned_text_to_sequence(text)
        assert seq_via_clean == seq_direct

    def test_sequence_is_list_of_ints(self):
        seq, _ = text_to_sequence("test", ["basic_cleaners"])
        assert isinstance(seq, list)
        assert all(isinstance(i, int) for i in seq)

    def test_empty_string(self):
        seq, clean = text_to_sequence("", ["basic_cleaners"])
        assert seq == []
        assert sequence_to_text(seq) == ""

    def test_punctuation_preserved(self):
        """Punctuation symbols that are in the symbol table should survive the roundtrip."""
        text = "hello, world!"
        seq, clean = text_to_sequence(text, ["basic_cleaners"])
        decoded = sequence_to_text(seq)
        assert decoded == text

    def test_unknown_symbol_raises(self):
        """Characters outside the symbol table should raise a KeyError."""
        # The null character is not in the symbol table.
        with pytest.raises(KeyError):
            cleaned_text_to_sequence("\x00")


# ---------------------------------------------------------------------------
# 3. intersperse utility
# ---------------------------------------------------------------------------


class TestIntersperse:
    """Test the intersperse helper used to insert blank tokens."""

    def test_basic(self):
        result = intersperse([1, 2, 3], 0)
        assert result == [0, 1, 0, 2, 0, 3, 0]

    def test_single_element(self):
        result = intersperse([42], 0)
        assert result == [0, 42, 0]

    def test_empty_list(self):
        result = intersperse([], 0)
        assert result == [0]

    def test_length_relation(self):
        """Result length must be 2*n + 1 where n is the input length."""
        for n in range(10):
            lst = list(range(n))
            result = intersperse(lst, -1)
            assert len(result) == 2 * n + 1

    def test_original_items_at_odd_indices(self):
        """The original items should sit at odd indices (1, 3, 5, ...)."""
        lst = [10, 20, 30]
        result = intersperse(lst, 0)
        assert result[1::2] == lst

    def test_blanks_at_even_indices(self):
        """The inserted blank should sit at even indices (0, 2, 4, ...)."""
        lst = [10, 20, 30]
        blank = 0
        result = intersperse(lst, blank)
        assert all(result[i] == blank for i in range(0, len(result), 2))

    def test_with_text_sequence(self):
        """Intersperse integrates with the text pipeline: insert blank (0) between IDs."""
        seq, _ = text_to_sequence("hi", ["basic_cleaners"])
        result = intersperse(seq, 0)
        # 'h' and 'i' yield 2 IDs, result should have 5 elements
        assert len(result) == 5
        assert result[1::2] == seq
