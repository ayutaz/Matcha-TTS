"""Edge case tests for Julius alignment pipeline and alignment metrics.

Covers:
  - Corrupted .lab files (non-integer timestamps, inverted times, etc.)
  - Large utterances (1000+ phonemes) for alignment and duration arrays
  - Abnormal prosody symbol patterns (all 7 symbols, consecutive, missing)
  - Numerical safety (very large values, all zeros, boundary cases)
  - alignment_metrics edge cases (mixed durations, single phoneme, large corpus)

All tests use synthetic data and do not require JVS corpus files or GPU.
"""

import sys
import time
import types

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Mock phonemizer before importing matcha.text modules (same pattern as
# test_convert_julius_to_durations.py). The cleaners module creates an
# EspeakBackend at import time.
# ---------------------------------------------------------------------------
_fake_phonemizer = types.ModuleType("phonemizer")
_fake_backend = types.ModuleType("phonemizer.backend")


class _FakeEspeakBackend:
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

sys.modules.setdefault("phonemizer", _fake_phonemizer)
sys.modules.setdefault("phonemizer.backend", _fake_backend)
sys.modules.setdefault("phonemizer.backend.espeak", _fake_espeak)
sys.modules.setdefault("phonemizer.backend.espeak.espeak", _fake_espeak_espeak)

# Now safe to import -------------------------------------------------------
from matcha.utils.alignment_metrics import (  # noqa: E402
    compute_corpus_stats,
    compute_duration_stats,
    is_degenerate,
)
from scripts.convert_julius_to_durations import (  # noqa: E402
    align_julius_with_pyopenjtalk,
    align_julius_with_pyopenjtalk_dtw,
    build_duration_array_with_blanks,
    parse_lab_file,
    time_to_frames,
)


# ===========================================================================
# 1. Corrupted .lab files
# ===========================================================================


class TestCorruptedLabFiles:
    def test_non_integer_timestamp(self, tmp_path):
        """Non-numeric timestamp line is skipped; valid lines are returned."""
        lab = tmp_path / "corrupt.lab"
        lab.write_text("0 abc silB\n0 5000000 k\n5000000 10000000 silE\n")
        segments = parse_lab_file(lab)
        # First line skipped (abc is not a valid integer), 2 valid lines remain
        assert len(segments) == 2
        assert segments[0][2] == "k"
        assert segments[1][2] == "silE"

    def test_inverted_timestamps(self, tmp_path):
        """end < start timestamps are parsed but time_to_frames clamps to 0."""
        lab = tmp_path / "inverted.lab"
        lab.write_text("0 5000000 silB\n10000000 5000000 k\n5000000 15000000 silE\n")
        segments = parse_lab_file(lab)
        # All three lines parse (integers are valid)
        assert len(segments) == 3
        # Inverted segment: start=1.0, end=0.5
        assert segments[1][0] == pytest.approx(1.0, abs=1e-9)
        assert segments[1][1] == pytest.approx(0.5, abs=1e-9)
        # time_to_frames should clamp to 0 for inverted segment
        frames = time_to_frames(segments[1][0], segments[1][1])
        assert frames == 0

    def test_empty_phoneme_label(self, tmp_path):
        """Line with trailing space and no phoneme text is skipped (< 3 fields)."""
        lab = tmp_path / "empty_ph.lab"
        lab.write_text("0 5000000 \n5000000 10000000 k\n")
        segments = parse_lab_file(lab)
        # "0 5000000 ".strip().split() -> ["0", "5000000"] (2 parts, skipped)
        assert len(segments) == 1
        assert segments[0][2] == "k"

    def test_extra_columns(self, tmp_path):
        """Lines with 4+ columns: extra columns are ignored, phoneme is column 3."""
        lab = tmp_path / "extra.lab"
        lab.write_text("0 5000000 silB extra_info\n5000000 10000000 k comment\n")
        segments = parse_lab_file(lab)
        assert len(segments) == 2
        assert segments[0][2] == "silB"
        assert segments[1][2] == "k"

    def test_very_large_timestamps(self, tmp_path):
        """Very large timestamps (10 minutes of audio) are parsed correctly."""
        lab = tmp_path / "long.lab"
        # 600 seconds = 6,000,000,000 in 100ns units
        lab.write_text("0 6000000000 silB\n6000000000 6000100000 silE\n")
        segments = parse_lab_file(lab)
        assert len(segments) == 2
        assert abs(segments[0][1] - 600.0) < 0.001
        assert abs(segments[1][0] - 600.0) < 0.001

    def test_single_field_lines_skipped(self, tmp_path):
        """Lines with fewer than 3 fields are silently skipped."""
        lab = tmp_path / "short.lab"
        lab.write_text("header\n0 5000000\n0 5000000 silB\n")
        segments = parse_lab_file(lab)
        # Only the last line has 3+ fields
        assert len(segments) == 1
        assert segments[0][2] == "silB"


# ===========================================================================
# 2. Large utterances (1000+ phonemes)
# ===========================================================================


class TestLargeUtterances:
    def test_1000_phonemes_sequential_alignment(self):
        """1000-phoneme sequential alignment completes and returns correct length."""
        phonemes = ["a", "k", "i"] * 333 + ["a"]  # 1000 phonemes
        julius_ph = ["sil"] + phonemes + ["sil"]
        julius_dur = [5] + [3] * 1000 + [5]
        pyopenjtalk_ph = ["^"] + phonemes + ["$"]

        result = align_julius_with_pyopenjtalk(
            julius_ph, pyopenjtalk_ph, julius_dur, align_mode="sequential"
        )
        assert len(result) == len(pyopenjtalk_ph)
        # ^ should get first sil duration
        assert result[0] == 5
        # $ should get last sil duration
        assert result[-1] == 5

    def test_1000_phonemes_dtw(self):
        """1000-phoneme DTW completes within 60 seconds."""
        phonemes = ["a", "k", "i"] * 333 + ["a"]
        julius_ph = ["sil"] + phonemes + ["sil"]
        julius_dur = [5] + [3] * 1000 + [5]
        pyopenjtalk_ph = ["^"] + phonemes + ["$"]

        start = time.time()
        result = align_julius_with_pyopenjtalk_dtw(
            julius_ph, pyopenjtalk_ph, julius_dur
        )
        elapsed = time.time() - start

        assert len(result) == len(pyopenjtalk_ph)
        assert elapsed < 60, f"DTW took {elapsed:.1f}s for 1000 phonemes"

    def test_build_duration_array_1000_phonemes(self):
        """1000-phoneme duration array has correct size and sum."""
        durations = [3] * 1000
        total = sum(durations)
        result = build_duration_array_with_blanks(durations, total)
        assert len(result) == 2001  # 2*1000+1
        assert result.sum() == total
        assert result.dtype == np.int64

    def test_1000_phonemes_all_durations_assigned(self):
        """All 1000 phoneme positions receive their duration value."""
        phonemes = ["a", "k", "i"] * 333 + ["a"]
        julius_ph = ["sil"] + phonemes + ["sil"]
        julius_dur = [5] + [3] * 1000 + [5]
        pyopenjtalk_ph = ["^"] + phonemes + ["$"]

        result = align_julius_with_pyopenjtalk(
            julius_ph, pyopenjtalk_ph, julius_dur, align_mode="sequential"
        )
        # All non-boundary phonemes should have dur=3
        for i in range(1, len(result) - 1):
            assert result[i] == 3, f"Phoneme at index {i} has dur={result[i]}, expected 3"


# ===========================================================================
# 3. Abnormal prosody symbol patterns
# ===========================================================================


class TestAbnormalProsodyPatterns:
    def test_all_prosody_symbols(self):
        """All 7 prosody symbols (^, $, ?, _, #, [, ]) are handled correctly."""
        julius_ph = ["sil", "k", "a", "pau", "k", "a", "sil"]
        julius_dur = [10, 5, 5, 3, 5, 5, 10]
        pyopenjtalk_ph = ["^", "[", "k", "a", "_", "#", "k", "]", "a", "?", "$"]

        result = align_julius_with_pyopenjtalk(julius_ph, pyopenjtalk_ph, julius_dur)
        assert len(result) == len(pyopenjtalk_ph)
        # ^ -> silB duration
        assert result[0] == 10
        # [ -> 0
        assert result[1] == 0
        # k -> 5
        assert result[2] == 5
        # a -> 5
        assert result[3] == 5
        # _ -> pau duration
        assert result[4] == 3
        # # -> 0
        assert result[5] == 0
        # k -> 5
        assert result[6] == 5
        # ] -> 0
        assert result[7] == 0
        # a -> 5
        assert result[8] == 5
        # ? -> 0
        assert result[9] == 0
        # $ -> silE duration
        assert result[10] == 10

    def test_triple_consecutive_prosody(self):
        """Three consecutive prosody symbols (] # [) all get duration=0."""
        julius_ph = ["sil", "k", "a", "k", "a", "sil"]
        julius_dur = [10, 5, 5, 5, 5, 10]
        pyopenjtalk_ph = ["^", "k", "a", "]", "#", "[", "k", "a", "$"]

        result = align_julius_with_pyopenjtalk(julius_ph, pyopenjtalk_ph, julius_dur)
        assert len(result) == len(pyopenjtalk_ph)
        assert result[3] == 0  # ]
        assert result[4] == 0  # #
        assert result[5] == 0  # [
        # Surrounding phonemes still matched correctly
        assert result[2] == 5  # a
        assert result[6] == 5  # k

    def test_prosody_at_start_after_hat(self):
        """[ immediately after ^ gets duration=0; ^ gets silB duration."""
        julius_ph = ["sil", "k", "a", "sil"]
        julius_dur = [10, 5, 5, 10]
        pyopenjtalk_ph = ["^", "[", "k", "a", "$"]

        result = align_julius_with_pyopenjtalk(julius_ph, pyopenjtalk_ph, julius_dur)
        assert result[0] > 0   # ^ gets silB duration
        assert result[0] == 10
        assert result[1] == 0  # [ gets 0

    def test_no_prosody_symbols(self):
        """Sequence with no prosody symbols: durations passed through directly."""
        julius_ph = ["sil", "k", "a", "sil"]
        julius_dur = [10, 5, 5, 10]
        # No ^, $, [, ], #, ?, _ -- direct phoneme names
        pyopenjtalk_ph = ["sil", "k", "a", "sil"]

        result = align_julius_with_pyopenjtalk(julius_ph, pyopenjtalk_ph, julius_dur)
        assert result == [10, 5, 5, 10]

    def test_question_mark_at_end(self):
        """? at utterance end (question intonation) gets duration=0."""
        julius_ph = ["sil", "k", "a", "sil"]
        julius_dur = [10, 5, 5, 10]
        pyopenjtalk_ph = ["^", "k", "a", "?"]

        result = align_julius_with_pyopenjtalk(julius_ph, pyopenjtalk_ph, julius_dur)
        assert len(result) == 4
        assert result[3] == 0  # ? always 0

    def test_prosody_only_utterance(self):
        """Utterance with only ^ and $ (no real phonemes)."""
        julius_ph = ["sil", "sil"]
        julius_dur = [50, 50]
        pyopenjtalk_ph = ["^", "$"]

        result = align_julius_with_pyopenjtalk(julius_ph, pyopenjtalk_ph, julius_dur)
        assert len(result) == 2
        # ^ gets first sil, $ gets last sil
        assert result[0] == 50
        assert result[1] == 50


# ===========================================================================
# 4. Numerical safety / integer boundary cases
# ===========================================================================


class TestNumericalSafety:
    def test_very_large_duration_values(self):
        """Duration array handles very large frame counts (10 min audio)."""
        # 10 minutes at 22050 Hz / 256 hop ~ 51680 frames
        total_frames = 51680
        durations = [total_frames]
        result = build_duration_array_with_blanks(durations, total_frames)
        assert result.sum() == total_frames
        assert result.dtype == np.int64

    def test_all_zero_durations(self):
        """All-zero durations: diff is placed in last phoneme slot."""
        durations = [0, 0, 0, 0, 0]
        total_mel = 100
        result = build_duration_array_with_blanks(durations, total_mel)
        assert result.sum() == total_mel
        # All zero initially, so diff=100 goes to last phoneme slot (index 2*5-1=9)
        assert result[9] == total_mel
        assert result.dtype == np.int64

    def test_single_frame_utterance(self):
        """1-frame utterance: single phoneme gets that 1 frame."""
        durations = [1]
        result = build_duration_array_with_blanks(durations, 1)
        assert len(result) == 3  # [0, 1, 0]
        assert result[0] == 0
        assert result[1] == 1
        assert result[2] == 0
        assert result.sum() == 1

    def test_duration_sum_larger_than_mel(self):
        """Duration sum > mel frames: last non-zero phoneme is reduced."""
        durations = [50, 50, 50]  # sum=150
        total_mel = 100  # need -50
        result = build_duration_array_with_blanks(durations, total_mel)
        assert result.sum() == total_mel
        # Last phoneme: max(0, 50 + (-50)) = 0
        assert result[5] == 0  # 50 - 50 = 0
        # First two unchanged
        assert result[1] == 50
        assert result[3] == 50
        assert all(result >= 0)

    def test_duration_sum_much_larger_than_mel_clamps_to_zero(self):
        """Massive negative adjustment clamps last phoneme to 0."""
        durations = [10, 10, 10]  # sum=30
        total_mel = 5  # need -25, last phoneme is only 10
        result = build_duration_array_with_blanks(durations, total_mel)
        # Last phoneme: max(0, 10 + (-25)) = 0, so sum = 10+10+0 = 20, not 5
        # This is a documented degenerate case -- verify no negatives
        assert all(result >= 0)

    def test_time_to_frames_very_large_audio(self):
        """time_to_frames handles 10-minute audio correctly."""
        frames = time_to_frames(0.0, 600.0)
        expected = round(600.0 * 22050 / 256)
        assert frames == expected
        assert frames > 50000  # sanity check

    def test_time_to_frames_sub_millisecond(self):
        """Sub-millisecond segment may round to 0 frames."""
        # 0.1ms = 0.0001s -> round(0.0001 * 22050 / 256) = round(0.0086) = 0
        frames = time_to_frames(0.0, 0.0001)
        assert frames == 0

    def test_build_empty_phoneme_list(self):
        """Empty phoneme list produces array of length 1 (single blank)."""
        result = build_duration_array_with_blanks([], 0)
        assert len(result) == 1  # 2*0+1 = 1
        assert result[0] == 0


# ===========================================================================
# 5. alignment_metrics edge cases
# ===========================================================================


class TestMetricsEdgeCases:
    def test_is_degenerate_with_mixed_zeros_and_large(self):
        """90% of phonemes <= 1 with one very large value -> degenerate."""
        # 21-element array = 10 phoneme positions (odd indices)
        # Phoneme values: [0, 1000, 0, 0, 0, 0, 0, 0, 0, 0]
        # 9 out of 10 are <= 1 -> 90% >= 80% threshold -> degenerate
        durations = np.array(
            [0, 0, 0, 1000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            dtype=np.int64,
        )
        assert is_degenerate(durations) is True

    def test_is_degenerate_all_large_values(self):
        """All phonemes have large durations -> not degenerate."""
        durations = np.array([0, 100, 0, 200, 0, 300, 0], dtype=np.int64)
        assert is_degenerate(durations) is False

    def test_is_degenerate_empty_array(self):
        """Single-element (no phonemes) array -> degenerate."""
        durations = np.array([0], dtype=np.int64)
        assert is_degenerate(durations) is True

    def test_compute_duration_stats_single_phoneme(self):
        """Single phoneme: mean and median both equal the single value."""
        durations = np.array([0, 42, 0], dtype=np.int64)
        stats = compute_duration_stats(durations)
        assert stats["mean"] == 42.0
        assert stats["median"] == 42.0
        assert stats["std"] == 0.0
        assert stats["pct_le1"] == 0.0
        assert stats["pct_le2"] == 0.0
        assert stats["blank0_duration"] == 0
        assert stats["is_degenerate"] is False

    def test_compute_duration_stats_all_same(self):
        """All phonemes have the same duration: std=0."""
        durations = np.array([0, 5, 0, 5, 0, 5, 0, 5, 0], dtype=np.int64)
        stats = compute_duration_stats(durations)
        assert stats["mean"] == 5.0
        assert stats["median"] == 5.0
        assert stats["std"] == 0.0

    def test_compute_duration_stats_no_phonemes(self):
        """Empty duration array (single blank): returns zero stats, degenerate=True."""
        durations = np.array([0], dtype=np.int64)
        stats = compute_duration_stats(durations)
        assert stats["mean"] == 0.0
        assert stats["is_degenerate"] is True
        assert stats["blank0_duration"] == 0

    def test_compute_corpus_stats_large_corpus(self):
        """1000-utterance corpus statistics are computed correctly."""
        corpus = [
            np.array([0, i + 1, 0, i + 2, 0], dtype=np.int64) for i in range(1000)
        ]
        stats = compute_corpus_stats(corpus)
        assert stats["total_samples"] == 1000
        # All utterances have phoneme durations >= 1 and >= 2 (i+1 >= 1, i+2 >= 2)
        # so none are degenerate (both phonemes > 1 for i >= 1, and for i=0: [1,2])
        assert stats["degenerate_count"] == 0
        assert stats["degenerate_rate"] == 0.0

    def test_compute_corpus_stats_empty_corpus(self):
        """Empty corpus returns zero stats without error."""
        stats = compute_corpus_stats([])
        assert stats["total_samples"] == 0
        assert stats["degenerate_count"] == 0
        assert stats["degenerate_rate"] == 0.0

    def test_compute_corpus_stats_all_degenerate(self):
        """Corpus where every utterance is degenerate."""
        # Each utterance has 5 phonemes all with duration=0 -> 100% <= 1 -> degenerate
        corpus = [np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int64) for _ in range(50)]
        stats = compute_corpus_stats(corpus)
        assert stats["total_samples"] == 50
        assert stats["degenerate_count"] == 50
        assert stats["degenerate_rate"] == 1.0

    def test_compute_corpus_stats_mixed(self):
        """Corpus with mix of healthy and degenerate utterances."""
        healthy = np.array([0, 10, 0, 8, 0, 12, 0], dtype=np.int64)
        degenerate = np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.int64)
        corpus = [healthy] * 7 + [degenerate] * 3  # 30% degenerate
        stats = compute_corpus_stats(corpus)
        assert stats["total_samples"] == 10
        assert stats["degenerate_count"] == 3
        assert abs(stats["degenerate_rate"] - 0.3) < 1e-9
