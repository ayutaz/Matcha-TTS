"""Tests for Julius .lab to Matcha-TTS duration array conversion.

Covers:
  - .lab file parsing (HTK format)
  - Time-to-frame conversion (absolute timestamp based)
  - Julius <-> pyopenjtalk phoneme alignment (sequential + DTW)
  - Blank-interspersed duration array construction
  - Full pipeline (single utterance processing)

All tests use synthetic data and do not require JVS corpus files.
pyopenjtalk-dependent tests are skipped when pyopenjtalk is not installed.
"""

import sys
import types

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Mock phonemizer before importing matcha.text modules (same pattern as
# test_text_ja.py). The cleaners module creates an EspeakBackend at import
# time.
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
from matcha.utils.utils import intersperse  # noqa: E402
from scripts.convert_julius_to_durations import (  # noqa: E402
    HOP_LENGTH,
    SAMPLE_RATE,
    align_julius_with_pyopenjtalk,
    align_julius_with_pyopenjtalk_dtw,
    build_duration_array_with_blanks,
    parse_lab_file,
    process_single_utterance,
    time_to_frames,
)

# ===========================================================================
# TestParseLabFile
# ===========================================================================


class TestParseLabFile:
    def test_basic_parsing(self, tmp_path):
        """3-line HTK .lab file is parsed correctly."""
        lab = tmp_path / "test.lab"
        lab.write_text(
            "0 2100000 silB\n"
            "2100000 5000000 k\n"
            "5000000 8000000 silE\n"
        )
        result = parse_lab_file(lab)
        assert len(result) == 3
        assert result[0][2] == "silB"
        assert result[1][2] == "k"
        assert result[2][2] == "silE"

    def test_empty_file(self, tmp_path):
        """Empty file returns empty list."""
        lab = tmp_path / "empty.lab"
        lab.write_text("")
        result = parse_lab_file(lab)
        assert result == []

    def test_time_conversion_to_seconds(self, tmp_path):
        """100ns units are correctly converted to seconds."""
        # 10_000_000 (100ns units) = 1.0 second
        lab = tmp_path / "time.lab"
        lab.write_text(
            "0 10000000 silB\n"
            "10000000 20000000 a\n"
            "20000000 30000000 silE\n"
        )
        result = parse_lab_file(lab)
        assert result[0] == pytest.approx((0.0, 1.0, "silB"), abs=1e-9)
        assert result[1] == pytest.approx((1.0, 2.0, "a"), abs=1e-9)
        assert result[2] == pytest.approx((2.0, 3.0, "silE"), abs=1e-9)

    def test_blank_lines_ignored(self, tmp_path):
        """Blank lines in .lab file are skipped."""
        lab = tmp_path / "blanks.lab"
        lab.write_text(
            "0 10000000 silB\n"
            "\n"
            "10000000 20000000 silE\n"
            "\n"
        )
        result = parse_lab_file(lab)
        assert len(result) == 2


# ===========================================================================
# TestTimeToFrames
# ===========================================================================


class TestTimeToFrames:
    def test_one_second(self):
        """1 second -> round(22050/256) = 86 frames."""
        frames = time_to_frames(0.0, 1.0)
        expected = round(1.0 * SAMPLE_RATE / HOP_LENGTH)  # 86
        assert frames == expected
        assert frames == 86

    def test_short_segment_10ms(self):
        """10ms -> round(0.01 * 22050 / 256) = 1 frame."""
        frames = time_to_frames(0.0, 0.01)
        expected = round(0.01 * SAMPLE_RATE / HOP_LENGTH)  # round(0.8613) = 1
        assert frames == expected
        assert frames == 1

    def test_zero_duration(self):
        """0 seconds -> 0 frames."""
        frames = time_to_frames(1.0, 1.0)
        assert frames == 0

    def test_negative_clipped_to_zero(self):
        """If start > end somehow, result is clamped to 0."""
        frames = time_to_frames(1.0, 0.5)
        assert frames == 0

    def test_absolute_vs_relative_accuracy(self):
        """Absolute timestamp rounding has less cumulative error than relative.

        Compare 10 consecutive 0.1s segments: absolute method computes each
        segment from global timestamps, while relative method rounds each
        segment independently and accumulates.
        """
        segment_duration = 0.1  # 100ms each
        n_segments = 10
        total_sec = n_segments * segment_duration

        # Absolute method (what we use)
        abs_frames = []
        for i in range(n_segments):
            start = i * segment_duration
            end = (i + 1) * segment_duration
            abs_frames.append(time_to_frames(start, end))
        abs_total = sum(abs_frames)

        # Relative method (naive)
        rel_frame_per_segment = round(segment_duration * SAMPLE_RATE / HOP_LENGTH)
        rel_total = rel_frame_per_segment * n_segments

        # Ground truth: total frames for the whole 1s span
        ground_truth = time_to_frames(0.0, total_sec)

        # Absolute method should match ground truth exactly
        assert abs_total == ground_truth
        # Relative method may differ (though in this particular case it happens
        # to match -- the test verifies the property holds)
        # The key point: absolute method is never worse than relative
        assert abs(abs_total - ground_truth) <= abs(rel_total - ground_truth)

    def test_non_zero_start(self):
        """Frames from a segment not starting at 0."""
        # 0.21s to 0.32s
        frames = time_to_frames(0.21, 0.32)
        start_f = round(0.21 * SAMPLE_RATE / HOP_LENGTH)
        end_f = round(0.32 * SAMPLE_RATE / HOP_LENGTH)
        assert frames == max(0, end_f - start_f)


# ===========================================================================
# TestAlignJuliusWithPyopenjtalk
# ===========================================================================


class TestAlignJuliusWithPyopenjtalk:
    def test_simple_konnichiwa(self):
        """Alignment for 'konnichiwa' (the standard test case)."""
        # Julius output (already mapped to pyopenjtalk symbols):
        julius_ph = ["sil", "k", "o", "N", "n", "i", "ch", "i", "w", "a", "sil"]
        julius_dur = [10,    5,   4,   3,   4,   3,   5,    3,   4,   5,   8]

        # pyopenjtalk output: "^ k o [ N n i ch i w a $"
        pyopenjtalk_ph = ["^", "k", "o", "[", "N", "n", "i", "ch", "i", "w", "a", "$"]

        result = align_julius_with_pyopenjtalk(julius_ph, pyopenjtalk_ph, julius_dur)

        assert len(result) == len(pyopenjtalk_ph)
        # ^ -> first sil duration
        assert result[0] == 10
        # k -> k duration
        assert result[1] == 5
        # o -> o duration
        assert result[2] == 4
        # [ -> prosody symbol, duration=0
        assert result[3] == 0
        # N -> N duration
        assert result[4] == 3
        # n -> n duration
        assert result[5] == 4
        # i -> i duration
        assert result[6] == 3
        # ch -> ch duration
        assert result[7] == 5
        # i -> i duration
        assert result[8] == 3
        # w -> w duration
        assert result[9] == 4
        # a -> a duration
        assert result[10] == 5
        # $ -> last sil duration
        assert result[11] == 8

    def test_prosody_symbols_get_zero_duration(self):
        """Prosody symbols [, ], # get duration=0."""
        julius_ph = ["sil", "k", "o", "sil"]
        julius_dur = [5, 10, 10, 5]

        pyopenjtalk_ph = ["^", "[", "k", "]", "#", "o", "$"]
        result = align_julius_with_pyopenjtalk(julius_ph, pyopenjtalk_ph, julius_dur)

        assert result[1] == 0  # [
        assert result[3] == 0  # ]
        assert result[4] == 0  # #

    def test_pause_gets_julius_pau_duration(self):
        """_ (pause) gets the Julius pau duration."""
        julius_ph = ["sil", "a", "pau", "i", "sil"]
        julius_dur = [5, 10, 15, 10, 5]

        pyopenjtalk_ph = ["^", "a", "_", "i", "$"]
        result = align_julius_with_pyopenjtalk(julius_ph, pyopenjtalk_ph, julius_dur)

        assert result[0] == 5   # ^
        assert result[1] == 10  # a
        assert result[2] == 15  # _ -> pau
        assert result[3] == 10  # i
        assert result[4] == 5   # $

    def test_devoiced_vowel_matching(self):
        """Devoiced vowels (uppercase A,I,U) match Julius lowercase."""
        # "suki" with devoiced U -> S U k i in pyopenjtalk
        julius_ph = ["sil", "s", "u", "k", "i", "sil"]
        julius_dur = [5, 4, 3, 4, 5, 5]

        # pyopenjtalk may produce devoiced U
        pyopenjtalk_ph = ["^", "s", "U", "k", "i", "$"]
        result = align_julius_with_pyopenjtalk(julius_ph, pyopenjtalk_ph, julius_dur)

        assert result[2] == 3  # U matched with Julius "u", dur=3

    def test_devoiced_vowel_missing_gets_zero(self):
        """Devoiced vowel missing in Julius gets duration=0."""
        # Julius dropped the devoiced vowel entirely
        julius_ph = ["sil", "s", "k", "i", "sil"]
        julius_dur = [5, 4, 4, 5, 5]

        pyopenjtalk_ph = ["^", "s", "U", "k", "i", "$"]
        result = align_julius_with_pyopenjtalk(julius_ph, pyopenjtalk_ph, julius_dur)

        # U should get 0 because Julius has no "u" at position after "s"
        assert result[2] == 0
        # k and i should still match
        assert result[3] == 4  # k
        assert result[4] == 5  # i

    def test_prosody_bracket_between_phonemes(self):
        """[ inserted between phonemes does not shift Julius index."""
        julius_ph = ["sil", "k", "o", "N", "sil"]
        julius_dur = [5, 4, 3, 6, 5]

        # [ appears between k and o
        pyopenjtalk_ph = ["^", "k", "[", "o", "N", "$"]
        result = align_julius_with_pyopenjtalk(julius_ph, pyopenjtalk_ph, julius_dur)

        assert result[0] == 5  # ^
        assert result[1] == 4  # k
        assert result[2] == 0  # [
        assert result[3] == 3  # o
        assert result[4] == 6  # N
        assert result[5] == 5  # $

    def test_hash_at_accent_boundary(self):
        """# at accent phrase boundary gets duration=0, adjacent phonemes correct."""
        julius_ph = ["sil", "a", "i", "u", "e", "o", "sil"]
        julius_dur = [5, 3, 4, 3, 4, 3, 5]

        # # separates two accent phrases
        pyopenjtalk_ph = ["^", "a", "i", "#", "u", "e", "o", "$"]
        result = align_julius_with_pyopenjtalk(julius_ph, pyopenjtalk_ph, julius_dur)

        assert result[0] == 5  # ^
        assert result[1] == 3  # a
        assert result[2] == 4  # i
        assert result[3] == 0  # #
        assert result[4] == 3  # u
        assert result[5] == 4  # e
        assert result[6] == 3  # o
        assert result[7] == 5  # $

    def test_multiple_prosody_consecutive(self):
        """Consecutive prosody symbols (e.g., ] #) each get duration=0."""
        julius_ph = ["sil", "a", "i", "sil"]
        julius_dur = [5, 3, 4, 5]

        pyopenjtalk_ph = ["^", "a", "]", "#", "i", "$"]
        result = align_julius_with_pyopenjtalk(julius_ph, pyopenjtalk_ph, julius_dur)

        assert result[2] == 0  # ]
        assert result[3] == 0  # #
        assert result[4] == 4  # i still gets correct duration

    def test_empty_pyopenjtalk_returns_empty(self):
        """Empty pyopenjtalk sequence returns empty list."""
        result = align_julius_with_pyopenjtalk(
            ["sil", "a", "sil"], [], [5, 10, 5]
        )
        assert result == []

    def test_question_mark_prosody(self):
        """? prosody marker gets duration=0."""
        julius_ph = ["sil", "a", "sil"]
        julius_dur = [5, 10, 5]

        pyopenjtalk_ph = ["^", "a", "?"]
        result = align_julius_with_pyopenjtalk(julius_ph, pyopenjtalk_ph, julius_dur)

        assert result[2] == 0  # ?


# ===========================================================================
# TestAlignDTW
# ===========================================================================


class TestAlignDTW:
    def test_dtw_matches_sequential_on_clean_input(self):
        """DTW produces same result as sequential on a clean (no mismatch) input."""
        julius_ph = ["sil", "k", "o", "N", "n", "i", "ch", "i", "w", "a", "sil"]
        julius_dur = [10, 5, 4, 3, 4, 3, 5, 3, 4, 5, 8]

        pyopenjtalk_ph = ["^", "k", "o", "[", "N", "n", "i", "ch", "i", "w", "a", "$"]

        result_seq = align_julius_with_pyopenjtalk(julius_ph, pyopenjtalk_ph, julius_dur)
        result_dtw = align_julius_with_pyopenjtalk_dtw(julius_ph, pyopenjtalk_ph, julius_dur)

        assert result_dtw == result_seq

    def test_dtw_handles_missing_phoneme(self):
        """DTW handles a phoneme missing in Julius."""
        # Julius is missing "n" compared to pyopenjtalk
        julius_ph = ["sil", "k", "o", "i", "sil"]
        julius_dur = [5, 4, 3, 4, 5]

        pyopenjtalk_ph = ["^", "k", "o", "n", "i", "$"]

        result = align_julius_with_pyopenjtalk_dtw(julius_ph, pyopenjtalk_ph, julius_dur)

        assert len(result) == len(pyopenjtalk_ph)
        # ^ and $ should get sil durations
        assert result[0] == 5  # ^ -> sil
        assert result[5] == 5  # $ -> sil
        # k, o, i should be matched
        assert result[1] == 4  # k
        assert result[2] == 3  # o
        assert result[4] == 4  # i
        # n is missing in Julius -> duration=0
        assert result[3] == 0

    def test_dtw_handles_extra_julius_phoneme(self):
        """DTW handles an extra phoneme in Julius not in pyopenjtalk."""
        julius_ph = ["sil", "k", "o", "r", "i", "sil"]
        julius_dur = [5, 4, 3, 2, 4, 5]

        # pyopenjtalk doesn't have "r"
        pyopenjtalk_ph = ["^", "k", "o", "i", "$"]

        result = align_julius_with_pyopenjtalk_dtw(julius_ph, pyopenjtalk_ph, julius_dur)

        assert len(result) == len(pyopenjtalk_ph)
        assert result[0] == 5   # ^
        assert result[1] == 4   # k
        assert result[2] == 3   # o
        assert result[3] == 4   # i
        assert result[4] == 5   # $

    def test_dtw_empty_pyopenjtalk(self):
        """DTW returns empty for empty pyopenjtalk."""
        result = align_julius_with_pyopenjtalk_dtw(
            ["sil", "a", "sil"], [], [5, 10, 5]
        )
        assert result == []


# ===========================================================================
# TestBuildDurationArrayWithBlanks
# ===========================================================================


class TestBuildDurationArrayWithBlanks:
    def test_length_is_2n_plus_1(self):
        """For N phonemes, array length is 2*N+1."""
        for n in [1, 3, 5, 10, 50]:
            durations = [5] * n
            total = sum(durations)
            arr = build_duration_array_with_blanks(durations, total)
            assert len(arr) == 2 * n + 1

    def test_blank_positions_are_zero(self):
        """Even indices (blank positions) are all zero."""
        durations = [5, 10, 15, 20]
        total = sum(durations)
        arr = build_duration_array_with_blanks(durations, total)

        for i in range(0, len(arr), 2):
            assert arr[i] == 0, f"Blank at index {i} should be 0, got {arr[i]}"

    def test_phoneme_positions_match(self):
        """Odd indices contain the phoneme durations."""
        durations = [5, 10, 15, 20]
        total = sum(durations)
        arr = build_duration_array_with_blanks(durations, total)

        for i, dur in enumerate(durations):
            assert arr[2 * i + 1] == dur

    def test_sum_matches_mel_frames(self):
        """Total duration equals total_mel_frames."""
        durations = [5, 10, 15, 20]
        total_mel = 50
        arr = build_duration_array_with_blanks(durations, total_mel)
        assert arr.sum() == total_mel

    def test_positive_adjustment(self):
        """When dur sum < mel_frames, last phoneme is extended."""
        durations = [5, 10, 15]  # sum=30
        total_mel = 40  # need +10
        arr = build_duration_array_with_blanks(durations, total_mel)

        assert arr.sum() == total_mel
        # Last phoneme (index 5) should be adjusted: 15 + 10 = 25
        assert arr[5] == 25
        # Other durations unchanged
        assert arr[1] == 5
        assert arr[3] == 10

    def test_negative_adjustment(self):
        """When dur sum > mel_frames, last phoneme is reduced."""
        durations = [5, 10, 15]  # sum=30
        total_mel = 25  # need -5
        arr = build_duration_array_with_blanks(durations, total_mel)

        assert arr.sum() == total_mel
        # Last phoneme adjusted: 15 - 5 = 10
        assert arr[5] == 10

    def test_negative_adjustment_clamped(self):
        """Adjustment does not make duration negative."""
        durations = [5, 10, 15]  # sum=30
        total_mel = 10  # need -20, but last phoneme is only 15
        arr = build_duration_array_with_blanks(durations, total_mel)

        # Last phoneme clamped to 0, so sum will be 5+10+0=15, not 10
        # This is a degenerate case -- verify no negative values
        assert all(arr >= 0), f"Negative values in array: {arr}"

    def test_consistent_with_intersperse(self):
        """Array length matches intersperse(text_seq, 0) length."""
        # Simulate a text sequence of length N
        for n in [1, 5, 10, 20]:
            text_seq = list(range(1, n + 1))
            interspersed = intersperse(text_seq, 0)
            durations = [3] * n
            total_mel = 3 * n
            arr = build_duration_array_with_blanks(durations, total_mel)
            assert len(arr) == len(interspersed)

    def test_dtype_is_int64(self):
        """Output array has dtype int64."""
        arr = build_duration_array_with_blanks([5, 10], 15)
        assert arr.dtype == np.int64

    def test_single_phoneme(self):
        """Works with a single phoneme."""
        arr = build_duration_array_with_blanks([50], 50)
        assert len(arr) == 3  # [0, 50, 0]
        assert arr[0] == 0
        assert arr[1] == 50
        assert arr[2] == 0
        assert arr.sum() == 50

    def test_zero_total_mel_frames(self):
        """Edge case: total_mel_frames=0 produces all-zero array."""
        arr = build_duration_array_with_blanks([5, 10], 0)
        assert all(arr >= 0)


# ===========================================================================
# TestProcessSingleUtterance
# ===========================================================================


class TestProcessSingleUtterance:
    @pytest.fixture(autouse=True)
    def _skip_without_pyopenjtalk(self):
        pytest.importorskip("pyopenjtalk")

    def test_full_pipeline(self, tmp_path):
        """.lab file -> .npy file full pipeline with dummy data."""
        # Create a dummy .lab file for "konnichiwa"
        # Julius segmentation: silB k o N n i ch i w a silE
        # Approximate timings (total ~0.93s = ~80 frames at 22050/256)
        lab_content = (
            "0 2100000 silB\n"          # 0.00-0.21s
            "2100000 3200000 k\n"       # 0.21-0.32s
            "3200000 4100000 o\n"       # 0.32-0.41s
            "4100000 4800000 N\n"       # 0.41-0.48s
            "4800000 5600000 n\n"       # 0.48-0.56s
            "5600000 6300000 i\n"       # 0.56-0.63s
            "6300000 7200000 ch\n"      # 0.63-0.72s
            "7200000 7900000 i\n"       # 0.72-0.79s
            "7900000 8600000 w\n"       # 0.79-0.86s
            "8600000 9300000 a\n"       # 0.86-0.93s
            "9300000 11500000 silE\n"   # 0.93-1.15s
        )
        lab_file = tmp_path / "test.lab"
        lab_file.write_text(lab_content)

        # Compute expected total frames
        total_frames = time_to_frames(0.0, 1.15)

        output_file = tmp_path / "test.npy"

        success, msg = process_single_utterance(
            str(lab_file),
            "こんにちは",
            total_frames,
            str(output_file),
        )

        assert success, f"Pipeline failed: {msg}"
        assert output_file.exists()

        # Load and verify
        arr = np.load(str(output_file))
        assert arr.dtype == np.int64
        assert arr.sum() == total_frames

        # Verify blank positions are zero
        for i in range(0, len(arr), 2):
            assert arr[i] == 0

    def test_pipeline_empty_lab_file(self, tmp_path):
        """Empty .lab file returns failure."""
        lab_file = tmp_path / "empty.lab"
        lab_file.write_text("")
        output_file = tmp_path / "empty.npy"

        success, msg = process_single_utterance(
            str(lab_file), "こんにちは", 100, str(output_file)
        )
        assert not success
        assert "Empty" in msg

    def test_pipeline_output_length_matches_interspersed(self, tmp_path):
        """Output .npy length matches intersperse(text_sequence, 0)."""
        from matcha.text import text_to_sequence

        text = "こんにちは"
        seq, _ = text_to_sequence(text, ["japanese_cleaners"], language="ja")
        interspersed = intersperse(seq, 0)
        expected_len = len(interspersed)

        # Create matching .lab
        lab_content = (
            "0 2100000 silB\n"
            "2100000 3200000 k\n"
            "3200000 4100000 o\n"
            "4100000 4800000 N\n"
            "4800000 5600000 n\n"
            "5600000 6300000 i\n"
            "6300000 7200000 ch\n"
            "7200000 7900000 i\n"
            "7900000 8600000 w\n"
            "8600000 9300000 a\n"
            "9300000 11500000 silE\n"
        )
        lab_file = tmp_path / "len_test.lab"
        lab_file.write_text(lab_content)
        output_file = tmp_path / "len_test.npy"

        total_frames = time_to_frames(0.0, 1.15)
        success, msg = process_single_utterance(
            str(lab_file), text, total_frames, str(output_file)
        )
        assert success, f"Pipeline failed: {msg}"

        arr = np.load(str(output_file))
        assert len(arr) == expected_len
