"""Tests for Julius-to-pyopenjtalk phoneme mapping module.

Covers:
  - Basic vowel/consonant/compound mapping
  - Special phonemes (N, cl, pau)
  - Julius-specific silence labels (silB, silE, sp, sil)
  - Unknown phoneme handling
  - Sequence mapping
  - Mapping table integrity (all values in symbols_ja)
  - Unmapped phoneme detection
  - Prosody symbol exclusion from Julius mapping
  - Full Julius phoneme coverage
"""

import pytest

from matcha.text.julius_to_pyopenjtalk import (
    JULIUS_TO_PYOPENJTALK,
    PROSODY_SYMBOLS,
    get_unmapped_phonemes,
    map_julius_phoneme,
    map_julius_sequence,
)
from matcha.text.symbols import symbols_ja

# ---------------------------------------------------------------------------
# 1. Basic vowel mapping
# ---------------------------------------------------------------------------


class TestBasicVowels:
    def test_map_basic_vowels(self):
        """a, i, u, e, o should map to themselves."""
        for vowel in ["a", "i", "u", "e", "o"]:
            assert map_julius_phoneme(vowel) == vowel


# ---------------------------------------------------------------------------
# 2. Basic consonant mapping
# ---------------------------------------------------------------------------


class TestBasicConsonants:
    def test_map_basic_consonants(self):
        """k, s, t, n, h, m, y, r, w, g, z, d, b, p should map to themselves."""
        consonants = ["k", "s", "t", "n", "h", "m", "y", "r", "w", "g", "z", "d", "b", "p"]
        for c in consonants:
            assert map_julius_phoneme(c) == c


# ---------------------------------------------------------------------------
# 3. Compound consonant mapping
# ---------------------------------------------------------------------------


class TestCompoundConsonants:
    def test_map_compound_consonants(self):
        """ky, sh, ch, ts, ty, ny, hy, ry, gy, by, py, my, dy, fy should map to themselves."""
        compounds = [
            "ky", "sh", "ch", "ts", "ty", "ny", "hy", "ry",
            "gy", "by", "py", "my", "dy", "fy",
        ]
        for c in compounds:
            assert map_julius_phoneme(c) == c


# ---------------------------------------------------------------------------
# 4. Special phonemes
# ---------------------------------------------------------------------------


class TestSpecialPhonemes:
    def test_map_special_phonemes(self):
        """N, cl, pau should map to themselves."""
        assert map_julius_phoneme("N") == "N"
        assert map_julius_phoneme("cl") == "cl"
        assert map_julius_phoneme("pau") == "pau"

    def test_map_q_to_cl(self):
        """Julius geminate 'q' should map to pyopenjtalk 'cl'."""
        assert map_julius_phoneme("q") == "cl"

    def test_map_long_vowels(self):
        """Julius long vowels (a:, i:, u:, e:, o:) should map to short vowels."""
        assert map_julius_phoneme("a:") == "a"
        assert map_julius_phoneme("i:") == "i"
        assert map_julius_phoneme("u:") == "u"
        assert map_julius_phoneme("e:") == "e"
        assert map_julius_phoneme("o:") == "o"


# ---------------------------------------------------------------------------
# 5. Silence mapping
# ---------------------------------------------------------------------------


class TestSilenceMapping:
    def test_map_silence_silB_to_sil(self):
        """silB (utterance start silence) should map to sil."""
        assert map_julius_phoneme("silB") == "sil"

    def test_map_silence_silE_to_sil(self):
        """silE (utterance end silence) should map to sil."""
        assert map_julius_phoneme("silE") == "sil"

    def test_map_sp_to_pau(self):
        """sp (short pause) should map to pau."""
        assert map_julius_phoneme("sp") == "pau"

    def test_map_sil_to_sil(self):
        """sil should map to sil."""
        assert map_julius_phoneme("sil") == "sil"


# ---------------------------------------------------------------------------
# 6. Unknown phoneme handling
# ---------------------------------------------------------------------------


class TestUnknownPhoneme:
    def test_map_unknown_phoneme_raises_keyerror(self):
        """Unknown phonemes should raise KeyError."""
        with pytest.raises(KeyError, match="Unknown Julius phoneme"):
            map_julius_phoneme("xx")

    def test_map_unknown_phoneme_empty_string(self):
        """Empty string should raise KeyError."""
        with pytest.raises(KeyError):
            map_julius_phoneme("")


# ---------------------------------------------------------------------------
# 7. Sequence mapping
# ---------------------------------------------------------------------------


class TestSequenceMapping:
    def test_map_sequence_basic(self):
        """silB k o N n i ch i w a silE -> sil k o N n i ch i w a sil."""
        julius_seq = ["silB", "k", "o", "N", "n", "i", "ch", "i", "w", "a", "silE"]
        expected = ["sil", "k", "o", "N", "n", "i", "ch", "i", "w", "a", "sil"]
        assert map_julius_sequence(julius_seq) == expected

    def test_map_sequence_with_sp(self):
        """Sequence with sp should correctly map to pau."""
        julius_seq = ["silB", "a", "sp", "i", "silE"]
        expected = ["sil", "a", "pau", "i", "sil"]
        assert map_julius_sequence(julius_seq) == expected

    def test_map_sequence_empty(self):
        """Empty sequence should return empty list."""
        assert map_julius_sequence([]) == []

    def test_map_sequence_with_q(self):
        """Sequence with q (geminate) should map q to cl."""
        julius_seq = ["silB", "i", "q", "k", "a", "i", "silE"]
        expected = ["sil", "i", "cl", "k", "a", "i", "sil"]
        assert map_julius_sequence(julius_seq) == expected

    def test_map_sequence_with_long_vowels(self):
        """Sequence with long vowels should map to short vowels."""
        julius_seq = ["silB", "o:", "k", "i:", "i", "silE"]
        expected = ["sil", "o", "k", "i", "i", "sil"]
        assert map_julius_sequence(julius_seq) == expected

    def test_map_sequence_unknown_raises_keyerror(self):
        """Sequence with unknown phoneme should raise KeyError."""
        with pytest.raises(KeyError):
            map_julius_sequence(["silB", "UNKNOWN", "silE"])


# ---------------------------------------------------------------------------
# 8. Mapping table integrity
# ---------------------------------------------------------------------------


class TestMappingIntegrity:
    def test_all_mapped_values_are_valid_symbols(self):
        """All values in JULIUS_TO_PYOPENJTALK must be in symbols_ja."""
        valid_symbols = set(symbols_ja)
        for julius_ph, pyopenjtalk_ph in JULIUS_TO_PYOPENJTALK.items():
            assert pyopenjtalk_ph in valid_symbols, (
                f"Mapping {julius_ph!r} -> {pyopenjtalk_ph!r} is not in symbols_ja"
            )

    def test_mapping_is_not_empty(self):
        """The mapping dictionary should not be empty."""
        assert len(JULIUS_TO_PYOPENJTALK) > 0

    def test_mapping_keys_are_strings(self):
        """All keys should be non-empty strings."""
        for key in JULIUS_TO_PYOPENJTALK:
            assert isinstance(key, str)
            assert len(key) > 0

    def test_mapping_values_are_strings(self):
        """All values should be non-empty strings."""
        for value in JULIUS_TO_PYOPENJTALK.values():
            assert isinstance(value, str)
            assert len(value) > 0


# ---------------------------------------------------------------------------
# 9. Unmapped phoneme detection
# ---------------------------------------------------------------------------


class TestGetUnmappedPhonemes:
    def test_get_unmapped_phonemes_empty_for_known(self):
        """Known phonemes should return empty set."""
        known = ["a", "i", "u", "silB", "silE", "N", "cl"]
        assert get_unmapped_phonemes(known) == set()

    def test_get_unmapped_phonemes_detects_unknown(self):
        """Unknown phonemes should be detected."""
        phonemes = ["a", "UNKNOWN1", "i", "UNKNOWN2"]
        result = get_unmapped_phonemes(phonemes)
        assert result == {"UNKNOWN1", "UNKNOWN2"}

    def test_get_unmapped_phonemes_empty_input(self):
        """Empty input should return empty set."""
        assert get_unmapped_phonemes([]) == set()

    def test_get_unmapped_phonemes_all_unknown(self):
        """All unknown phonemes should be returned."""
        phonemes = ["XX", "YY", "ZZ"]
        result = get_unmapped_phonemes(phonemes)
        assert result == {"XX", "YY", "ZZ"}


# ---------------------------------------------------------------------------
# 10. Prosody symbols exclusion
# ---------------------------------------------------------------------------


class TestProsodySymbols:
    def test_prosody_symbols_not_in_julius_mapping(self):
        """Prosody symbols (^, $, ?, _, #, [, ]) should not be in JULIUS_TO_PYOPENJTALK keys."""
        julius_keys = set(JULIUS_TO_PYOPENJTALK.keys())
        for symbol in PROSODY_SYMBOLS:
            assert symbol not in julius_keys, (
                f"Prosody symbol {symbol!r} should not be a Julius mapping key"
            )

    def test_prosody_symbols_are_in_symbols_ja(self):
        """Prosody symbols should be valid symbols_ja entries."""
        valid_symbols = set(symbols_ja)
        for symbol in PROSODY_SYMBOLS:
            assert symbol in valid_symbols

    def test_prosody_symbols_complete(self):
        """PROSODY_SYMBOLS should contain exactly 7 symbols."""
        assert len(PROSODY_SYMBOLS) == 7
        assert {"^", "$", "?", "_", "#", "[", "]"} == PROSODY_SYMBOLS


# ---------------------------------------------------------------------------
# 11. Full Julius phoneme coverage
# ---------------------------------------------------------------------------


class TestJuliusPhoneCoverage:
    def test_mapping_covers_all_julius_phonemes(self):
        """JULIUS_TO_PYOPENJTALK should cover all phonemes that Julius can output.

        Julius segmentation-kit for Japanese outputs these phoneme categories:
        - 5 vowels: a, i, u, e, o
        - 16 basic consonants: k, s, t, n, h, m, y, r, w, g, z, d, b, p, f, v
        - 17 compound consonants: ky, sh, ch, ts, ty, ny, hy, ry, gy, by, py, my, dy, fy, j, kw, gw
        - 4 special: N (moraic nasal), cl (geminate), q (geminate variant), pau (pause)
        - 5 long vowels: a:, i:, u:, e:, o:
        - 4 silence: silB, silE, sp, sil
        """
        # All expected Julius output phonemes
        expected_julius_phonemes = {
            # Vowels
            "a", "i", "u", "e", "o",
            # Basic consonants
            "k", "s", "t", "n", "h", "m", "y", "r", "w",
            "g", "z", "d", "b", "p", "f", "v",
            # Compound consonants
            "ky", "sh", "ch", "ts", "ty", "ny", "hy", "ry",
            "gy", "by", "py", "my", "dy", "fy", "j", "kw", "gw",
            # Special
            "N", "cl", "q", "pau",
            # Long vowels
            "a:", "i:", "u:", "e:", "o:",
            # Silence
            "silB", "silE", "sp", "sil",
        }
        julius_keys = set(JULIUS_TO_PYOPENJTALK.keys())

        missing = expected_julius_phonemes - julius_keys
        assert not missing, f"Missing Julius phonemes in mapping: {sorted(missing)}"

    def test_no_extra_keys_beyond_julius(self):
        """All mapping keys should be recognized Julius phonemes (no typos or extras)."""
        expected_julius_phonemes = {
            "a", "i", "u", "e", "o",
            "k", "s", "t", "n", "h", "m", "y", "r", "w",
            "g", "z", "d", "b", "p", "f", "v",
            "ky", "sh", "ch", "ts", "ty", "ny", "hy", "ry",
            "gy", "by", "py", "my", "dy", "fy", "j", "kw", "gw",
            "N", "cl", "q", "pau",
            "a:", "i:", "u:", "e:", "o:",
            "silB", "silE", "sp", "sil",
        }
        julius_keys = set(JULIUS_TO_PYOPENJTALK.keys())

        extra = julius_keys - expected_julius_phonemes
        assert not extra, f"Unexpected keys in mapping: {sorted(extra)}"
