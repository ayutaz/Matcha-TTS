"""Tests for the Matcha-TTS Japanese text processing pipeline.

Covers:
  - Japanese symbol list (symbols_ja) completeness
  - text_to_sequence / sequence_to_text roundtrip with language="ja"
  - cleaned_text_to_sequence with language="ja"
  - japanese_cleaners (requires pyopenjtalk, skipped if not installed)
"""

import sys
import types

import pytest

# ---------------------------------------------------------------------------
# Module-level sys.modules mocking (intentional).
#
# matcha/text/cleaners.py instantiates ``phonemizer.backend.EspeakBackend(...)``
# at *import time* (module scope), so the mock **must** be installed before any
# ``matcha.text`` import occurs.  This rules out ``monkeypatch`` (function- or
# session-scoped) and ``conftest.py`` autouse fixtures, because Python will have
# already executed the top-level code in cleaners.py by the time a fixture runs.
#
# The mutation is confined to this test module and is harmless to other tests:
# if phonemizer is genuinely installed the real module will already be in
# ``sys.modules`` and this block simply overwrites it with a compatible stub for
# the duration of the process.
# ---------------------------------------------------------------------------
_fake_phonemizer = types.ModuleType("phonemizer")
_fake_backend = types.ModuleType("phonemizer.backend")


class _FakeEspeakBackend:
    """Minimal stand-in so cleaners.py can be imported."""

    def __init__(self, **kwargs):
        pass

    def phonemize(self, text_list, strip=True, njobs=1):
        return text_list


_fake_backend.EspeakBackend = _FakeEspeakBackend
_fake_phonemizer.backend = _fake_backend

_fake_espeak = types.ModuleType("phonemizer.backend.espeak")
_fake_espeak_espeak = types.ModuleType("phonemizer.backend.espeak.espeak")
_fake_backend.espeak = _fake_espeak
_fake_espeak.espeak = _fake_espeak_espeak

sys.modules["phonemizer"] = _fake_phonemizer
sys.modules["phonemizer.backend"] = _fake_backend
sys.modules["phonemizer.backend.espeak"] = _fake_espeak
sys.modules["phonemizer.backend.espeak.espeak"] = _fake_espeak_espeak

# Now it is safe to import the text modules ---------------------------------
from matcha.text import (  # noqa: E402
    cleaned_text_to_sequence,
    sequence_to_text,
    text_to_sequence,
)
from matcha.text.symbols import symbols_ja  # noqa: E402
from matcha.utils.utils import intersperse  # noqa: E402

# ---------------------------------------------------------------------------
# 1. Japanese symbol list completeness
# ---------------------------------------------------------------------------


class TestSymbolsJa:
    """Verify structural properties of the Japanese symbol table."""

    def test_symbol_count(self):
        """The Japanese symbol list should contain exactly 55 entries (including fy, gw, kw)."""
        assert len(symbols_ja) == 55

    def test_pad_is_first(self):
        """The padding symbol '~' must be the first symbol (index 0)."""
        assert symbols_ja[0] == "~"

    def test_no_duplicate_symbols(self):
        """All Japanese symbols should be unique."""
        assert len(set(symbols_ja)) == len(symbols_ja)

    def test_contains_prosody_markers(self):
        """Prosody markers should be present."""
        for marker in ["^", "$", "?", "_", "#", "[", "]"]:
            assert marker in symbols_ja

    def test_contains_key_phonemes(self):
        """Key Japanese phonemes should be in the symbol list."""
        for ph in ["a", "i", "u", "e", "o", "N", "cl", "pau", "sil", "sh", "ch", "ts"]:
            assert ph in symbols_ja


# ---------------------------------------------------------------------------
# 2. text_to_sequence / sequence_to_text roundtrip (language="ja")
# ---------------------------------------------------------------------------


class TestJapaneseTextSequenceConversion:
    """Test encoding and decoding of Japanese phoneme sequences."""

    def test_simple_phoneme_roundtrip(self):
        """Space-separated phonemes should survive a full roundtrip."""
        # Use cleaned_text_to_sequence to avoid basic_cleaners lowercasing N (moraic nasal)
        phonemes = "^ k o N n i ch i w a $"
        seq = cleaned_text_to_sequence(phonemes, language="ja")
        decoded = sequence_to_text(seq, language="ja")
        assert decoded == phonemes

    def test_multi_char_phonemes(self):
        """Multi-character phonemes like 'sh', 'ch', 'ts' should be handled correctly."""
        phonemes = "sh i ts u r e i sh i m a sh i t a"
        seq, clean = text_to_sequence(phonemes, ["basic_cleaners"], language="ja")
        # Each space-separated token should map to exactly one ID
        assert len(seq) == len(phonemes.split())
        decoded = sequence_to_text(seq, language="ja")
        assert decoded == phonemes

    def test_prosody_markers(self):
        """Prosody markers should be encoded and decoded correctly."""
        phonemes = "^ [ k o N n i ch i w a ] $"
        seq = cleaned_text_to_sequence(phonemes, language="ja")
        decoded = sequence_to_text(seq, language="ja")
        assert decoded == phonemes

    def test_sequence_is_list_of_ints(self):
        seq, _ = text_to_sequence("a i u", ["basic_cleaners"], language="ja")
        assert isinstance(seq, list)
        assert all(isinstance(i, int) for i in seq)

    def test_cleaned_text_to_sequence_ja(self):
        """cleaned_text_to_sequence should work with language='ja'."""
        phonemes = "a i u e o"
        seq_via_clean, _ = text_to_sequence(phonemes, ["basic_cleaners"], language="ja")
        seq_direct = cleaned_text_to_sequence(phonemes, language="ja")
        assert seq_via_clean == seq_direct

    def test_pad_symbol_id(self):
        """The pad symbol '~' should have ID 0."""
        seq = cleaned_text_to_sequence("~", language="ja")
        assert seq == [0]

    def test_with_intersperse(self):
        """Intersperse should work with Japanese sequences."""
        seq, _ = text_to_sequence("a i u", ["basic_cleaners"], language="ja")
        result = intersperse(seq, 0)
        assert len(result) == 2 * len(seq) + 1
        assert result[1::2] == seq


# ---------------------------------------------------------------------------
# 3. japanese_cleaners (requires pyopenjtalk)
# ---------------------------------------------------------------------------


class TestJapaneseCleaners:
    """Test japanese_cleaners (requires pyopenjtalk to be installed)."""

    @pytest.fixture(autouse=True)
    def _skip_without_pyopenjtalk(self):
        pytest.importorskip("pyopenjtalk")

    def test_basic_japanese_text(self):
        from matcha.text.cleaners import japanese_cleaners

        result = japanese_cleaners("こんにちは")
        # Should return a space-separated string of phonemes/prosody markers
        assert isinstance(result, str)
        tokens = result.split()
        assert len(tokens) > 0
        # Must start with ^ (utterance start) and end with $ (utterance end)
        assert tokens[0] == "^"
        assert tokens[-1] == "$"

    def test_output_symbols_in_table(self):
        from matcha.text.cleaners import japanese_cleaners

        result = japanese_cleaners("東京は日本の首都です")
        tokens = result.split()
        for token in tokens:
            assert token in symbols_ja, f"Token '{token}' not in symbols_ja"


# ---------------------------------------------------------------------------
# 4. Backward compatibility — English still works
# ---------------------------------------------------------------------------


class TestEnglishBackwardCompatibility:
    """Ensure language='en' (default) still works as before."""

    def test_english_default(self):
        """Calling without language arg should behave identically to before."""
        seq1, clean1 = text_to_sequence("hello", ["basic_cleaners"])
        seq2, clean2 = text_to_sequence("hello", ["basic_cleaners"], language="en")
        assert seq1 == seq2
        assert clean1 == clean2

    def test_sequence_to_text_default(self):
        seq, _ = text_to_sequence("hello", ["basic_cleaners"])
        result1 = sequence_to_text(seq)
        result2 = sequence_to_text(seq, language="en")
        assert result1 == result2

    def test_cleaned_text_default(self):
        seq1 = cleaned_text_to_sequence("hello")
        seq2 = cleaned_text_to_sequence("hello", language="en")
        assert seq1 == seq2
