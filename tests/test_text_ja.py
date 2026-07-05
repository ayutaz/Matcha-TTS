"""Tests for the Matcha-TTS Japanese text processing pipeline.

Covers:
  - Japanese symbol list (symbols_ja) completeness
  - text_to_sequence / sequence_to_text roundtrip with language="ja"
  - cleaned_text_to_sequence with language="ja"
  - japanese_cleaners (requires pyopenjtalk, skipped if not installed)
  - _fullcontext_to_prosody with synthetic HTS full-context labels
  - Frozen index snapshot of symbols_ja (checkpoint compatibility guard)
  - _cached_clean_text LRU cache keying (language / cleaner names)
  - Exhaustive 55-symbol encode/decode roundtrip
  - Error paths (_get_symbol_map, unknown tokens/IDs, unsupported language)
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
import matcha.text  # noqa: E402
from matcha.text import (  # noqa: E402
    _cached_clean_text,
    cleaned_text_to_sequence,
    sequence_to_text,
    text_to_sequence,
)
from matcha.text.cleaners import _fullcontext_to_prosody  # noqa: E402
from matcha.text.symbols import symbols, symbols_ja  # noqa: E402
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


# ---------------------------------------------------------------------------
# 5. _fullcontext_to_prosody with synthetic HTS labels (pure function)
# ---------------------------------------------------------------------------


def _make_label(ph, a1="xx", a2="xx", f1="xx"):
    """Build a minimal synthetic HTS full-context label.

    Only the fields that ``_fullcontext_to_prosody`` actually parses are
    populated: the ``-p3+`` phoneme slot, ``/A:a1+a2+…`` and ``/F:f1_…``.
    Passing the default ``"xx"`` makes the corresponding regex fail to match,
    exactly as in real pyopenjtalk labels for silence segments.
    """
    return f"xx^xx-{ph}+xx=xx/A:{a1}+{a2}+xx/F:{f1}_xx"


class TestFullcontextToProsody:
    """Characterization tests for _fullcontext_to_prosody on synthetic labels.

    These pin the CURRENT behaviour of the branch (including known quirks)
    so that any refactor of the label-parsing regexes is caught.
    """

    def test_sil_positions(self):
        """First sil -> '^', last sil -> '$', middle sil -> '?'."""
        labels = [_make_label("sil"), _make_label("sil"), _make_label("sil")]
        assert _fullcontext_to_prosody(labels) == ["^", "?", "$"]

    def test_single_sil_label_treated_as_start(self):
        """A lone sil label is both first and last; the i == 0 branch wins."""
        assert _fullcontext_to_prosody([_make_label("sil")]) == ["^"]

    def test_pau_maps_to_underscore(self):
        assert _fullcontext_to_prosody([_make_label("pau")]) == ["_"]

    def test_accent_phrase_boundary(self):
        """'#' is emitted when a1 == 1 and the NEXT label has a2 == 1."""
        labels = [
            _make_label("k", a1=1, a2=3, f1=5),  # a1==1, next a2==1 -> '#'
            _make_label("a", a1=1, a2=1, f1=5),  # last label: rise marker only
        ]
        assert _fullcontext_to_prosody(labels) == ["#", "k", "[", "a"]

    def test_pitch_rise(self):
        """'[' is emitted when a2 == 1 and a1 == 1 (next a2 != 1 avoids '#')."""
        labels = [
            _make_label("k", a1=1, a2=1, f1=5),
            _make_label("a", a1=2, a2=2, f1=5),
        ]
        assert _fullcontext_to_prosody(labels) == ["[", "k", "a"]

    def test_pitch_fall(self):
        """']' is emitted when a1 == a2 + 1 and a2 != f1."""
        assert _fullcontext_to_prosody([_make_label("o", a1=2, a2=1, f1=3)]) == ["]", "o"]

    def test_no_fall_when_accent_on_last_mora(self):
        """No ']' when a2 == f1 even though a1 == a2 + 1."""
        assert _fullcontext_to_prosody([_make_label("o", a1=2, a2=1, f1=1)]) == ["o"]

    def test_missing_accent_fields_yield_plain_phoneme(self):
        """'xx' in the /A: and /F: fields -> plain phoneme, no prosody markers."""
        assert _fullcontext_to_prosody([_make_label("a")]) == ["a"]

    def test_label_without_accent_blocks_yields_plain_phoneme(self):
        """A label lacking /A: and /F: blocks entirely is still a plain phoneme."""
        assert _fullcontext_to_prosody(["xx^xx-a+xx=xx"]) == ["a"]

    def test_label_without_phoneme_pattern_is_skipped(self):
        """Lines that do not match the '-ph+' pattern are silently dropped."""
        labels = ["no phoneme pattern here", _make_label("a")]
        assert _fullcontext_to_prosody(labels) == ["a"]

    def test_last_label_a2_next_defaults_to_minus_one(self):
        """For the final label a2_next defaults to -1, so a1 == 1 emits no '#'."""
        assert _fullcontext_to_prosody([_make_label("a", a1=1, a2=2, f1=5)]) == ["a"]

    def test_negative_a1_emits_no_markers(self):
        """Negative a1 (mora before the accent nucleus) parses and emits no marker."""
        assert _fullcontext_to_prosody([_make_label("k", a1=-4, a2=1, f1=5)]) == ["k"]

    def test_empty_label_list(self):
        assert _fullcontext_to_prosody([]) == []

    def test_probe_characterization_full_utterance(self):
        """Characterization: pin the full output for a synthetic utterance.

        NOTE: this pins current-branch behaviour, including the quirk that
        every phoneme whose mora satisfies a1 == 1 and a2 == 1 receives its
        own '[' marker (duplicate '['-per-mora), rather than one '[' per
        accent phrase as in ttslearn's pp_symbols.
        """
        labels = [
            "xx^xx-sil+k=o/A:xx+xx+xx/F:xx_xx",
            "xx^sil-k+a=a/A:1+1+5/F:5_1",
            "sil^k-a+sil=xx/A:1+1+5/F:5_1",
            "k^a-sil+xx=xx/A:xx+xx+xx/F:xx_xx",
        ]
        assert _fullcontext_to_prosody(labels) == ["^", "#", "[", "k", "[", "a", "$"]


# ---------------------------------------------------------------------------
# 6. Frozen index snapshot of symbols_ja (checkpoint compatibility guard)
# ---------------------------------------------------------------------------


class TestSymbolsJaFrozenSnapshot:
    """Guard the exact symbol-to-index mapping.

    Trained Japanese checkpoints embed these indices; reordering or
    inserting symbols silently breaks every existing model, so the full
    table is hardcoded here as a frozen snapshot.
    """

    # Copied verbatim from matcha/text/symbols.py (55 symbols).
    EXPECTED_SYMBOLS_JA = [
        "~",  # 0: pad
        "^", "$", "?", "_", "#", "[", "]",  # 1-7: prosody markers
        "A", "E", "I", "N", "O", "U",  # 8-13: devoiced vowels + moraic nasal
        "a", "b", "by", "ch", "cl", "d", "dy", "e", "f", "fy",  # 14-23
        "g", "gw", "gy", "h", "hy", "i", "j", "k", "kw", "ky",  # 24-33
        "m", "my", "n", "ny", "o", "p", "py",  # 34-40
        "r", "ry", "s", "sh", "t", "ts", "ty",  # 41-47
        "u", "v", "w", "y", "z",  # 48-52
        "pau", "sil",  # 53-54
    ]  # fmt: skip

    def test_full_frozen_snapshot(self):
        """symbols_ja must match the hardcoded snapshot element-wise."""
        assert symbols_ja == self.EXPECTED_SYMBOLS_JA

    @pytest.mark.parametrize(
        ("symbol", "index"),
        [
            ("~", 0),
            ("^", 1),
            ("$", 2),
            ("?", 3),
            ("_", 4),
            ("#", 5),
            ("[", 6),
            ("]", 7),
            ("N", 11),
            ("cl", 18),
            ("pau", 53),
            ("sil", 54),
        ],
    )
    def test_individual_symbol_indices(self, symbol, index):
        assert symbols_ja[index] == symbol
        assert symbols_ja.index(symbol) == index


# ---------------------------------------------------------------------------
# 7. Shared cleaner cache — language must be part of the LRU cache key
# ---------------------------------------------------------------------------


class TestSharedCleanerCache:
    """_cached_clean_text is shared across languages; verify key separation."""

    def test_en_and_ja_yield_distinct_correct_sequences(self):
        """The same text yields per-character IDs (en) vs per-token IDs (ja)."""
        _cached_clean_text.cache_clear()
        seq_en, clean_en = text_to_sequence("a i u", ["basic_cleaners"], language="en")
        seq_ja, clean_ja = text_to_sequence("a i u", ["basic_cleaners"], language="ja")
        assert clean_en == clean_ja == "a i u"
        # English path: one ID per character (including spaces) -> 5 IDs
        assert seq_en == [symbols.index(c) for c in "a i u"]
        assert len(seq_en) == 5
        # Japanese path: one ID per whitespace-separated token -> 3 IDs
        assert seq_ja == [symbols_ja.index(t) for t in ["a", "i", "u"]]
        assert len(seq_ja) == 3
        assert seq_en != seq_ja

    def test_language_is_part_of_cache_key(self):
        """Same text + cleaners with different languages must be two cache entries."""
        _cached_clean_text.cache_clear()
        text_to_sequence("a i u", ["basic_cleaners"], language="en")
        text_to_sequence("a i u", ["basic_cleaners"], language="ja")
        info = _cached_clean_text.cache_info()
        assert info.misses == 2
        assert info.currsize == 2
        assert info.hits == 0

    def test_repeat_calls_return_equal_results_via_cache_hit(self):
        _cached_clean_text.cache_clear()
        first, clean_first = text_to_sequence("a i u", ["basic_cleaners"], language="ja")
        second, clean_second = text_to_sequence("a i u", ["basic_cleaners"], language="ja")
        assert first == second
        assert clean_first == clean_second
        info = _cached_clean_text.cache_info()
        assert info.misses == 1
        assert info.hits == 1

    def test_cleaner_names_not_cross_contaminated(self):
        """Different cleaner_names on the same text must not share cache entries."""
        _cached_clean_text.cache_clear()
        _, clean_basic = text_to_sequence("A  B", ["basic_cleaners"], language="en")
        _, clean_collapse = text_to_sequence("A  B", ["collapse_whitespace"], language="en")
        assert clean_basic == "a b"  # lowercased + whitespace collapsed
        assert clean_collapse == "A B"  # only whitespace collapsed
        info = _cached_clean_text.cache_info()
        assert info.misses == 2
        assert info.currsize == 2


# ---------------------------------------------------------------------------
# 8. Exhaustive 55-symbol roundtrip
# ---------------------------------------------------------------------------


class TestFullVocabRoundtrip:
    """Encode and decode every symbol in the Japanese vocabulary."""

    def test_all_55_symbols_encode_to_their_indices(self):
        text = " ".join(symbols_ja)
        ids = cleaned_text_to_sequence(text, language="ja")
        assert ids == list(range(55))

    def test_full_vocab_roundtrip_to_identical_string(self):
        text = " ".join(symbols_ja)
        ids = cleaned_text_to_sequence(text, language="ja")
        assert sequence_to_text(ids, language="ja") == text


# ---------------------------------------------------------------------------
# 9. Error paths
# ---------------------------------------------------------------------------


class TestErrorPaths:
    """Unknown tokens, unknown IDs and unsupported languages."""

    def test_unknown_ja_token_raises_keyerror(self):
        with pytest.raises(KeyError, match="zz"):
            cleaned_text_to_sequence("zz", language="ja")

    def test_unsupported_language_raises_valueerror(self):
        with pytest.raises(ValueError, match="Unsupported language"):
            matcha.text._get_symbol_map("fr")

    def test_out_of_range_id_raises_keyerror(self):
        with pytest.raises(KeyError, match="999"):
            sequence_to_text([999], language="ja")

    def test_unknown_language_falls_through_to_english(self):
        """Characterization: text_to_sequence only special-cases 'ja', so any
        other language string (e.g. 'xx') silently takes the English path
        today instead of raising. Pinning current behaviour."""
        seq_xx, clean_xx = text_to_sequence("hello", ["basic_cleaners"], language="xx")
        seq_en, clean_en = text_to_sequence("hello", ["basic_cleaners"], language="en")
        assert seq_xx == seq_en
        assert clean_xx == clean_en


# ---------------------------------------------------------------------------
# 10. japanese_cleaners — broader end-to-end coverage (requires pyopenjtalk)
# ---------------------------------------------------------------------------


class TestJapaneseCleanersExtended:
    """Broader japanese_cleaners coverage (requires pyopenjtalk)."""

    @pytest.fixture(autouse=True)
    def _skip_without_pyopenjtalk(self):
        pytest.importorskip("pyopenjtalk")

    def test_midsentence_comma_produces_pause(self):
        """A mid-sentence '、' should be rendered as a pause marker '_'."""
        from matcha.text.cleaners import japanese_cleaners

        tokens = japanese_cleaners("今日は、雨です").split()
        assert "_" in tokens
        # The pause must be strictly inside the utterance brackets
        assert 0 < tokens.index("_") < len(tokens) - 1

    @pytest.mark.parametrize("text", ["2024年です", "ABCです"])
    def test_non_kana_text_stays_in_vocab(self, text):
        """Digits and Latin letters must map into symbols_ja and encode cleanly."""
        from matcha.text.cleaners import japanese_cleaners

        result = japanese_cleaners(text)
        tokens = result.split()
        assert len(tokens) > 0
        for token in tokens:
            assert token in symbols_ja, f"Token '{token}' not in symbols_ja"
        # Must encode without KeyError
        seq = cleaned_text_to_sequence(result, language="ja")
        assert len(seq) == len(tokens)

    def test_accented_text_produces_pitch_rise(self):
        """A sentence containing a heiban accent phrase (東京) yields '['."""
        from matcha.text.cleaners import japanese_cleaners

        tokens = japanese_cleaners("東京は日本の首都です").split()
        assert "[" in tokens

    @pytest.mark.parametrize(
        "text",
        [
            "こんにちは",
            "ありがとうございます",
            "今日はいい天気ですね",
            "音声合成のテストです",
        ],
    )
    def test_utterance_starts_and_ends_with_brackets(self, text):
        """Every utterance must start with '^' and end with '$'."""
        from matcha.text.cleaners import japanese_cleaners

        tokens = japanese_cleaners(text).split()
        assert tokens[0] == "^"
        assert tokens[-1] == "$"
