"""Tests for matcha.utils.alignment_metrics.

All tests use synthetic data and do not require JVS corpus files or GPU.
"""

import numpy as np
import pytest

from matcha.utils.alignment_metrics import (
    compute_corpus_stats,
    compute_corpus_stats_streaming,
    compute_duration_stats,
    compute_phoneme_class_stats,
    is_degenerate,
)

# ===========================================================================
# TestIsDegenerate
# ===========================================================================


class TestIsDegenerate:
    def test_all_short_is_degenerate(self):
        """All phonemes <= 1 frame -> degenerate."""
        # blank-interspersed: [0, 1, 0, 0, 0, 1, 0, 1, 0]
        durations = np.array([0, 1, 0, 0, 0, 1, 0, 1, 0], dtype=np.int64)
        assert is_degenerate(durations) is True

    def test_healthy_not_degenerate(self):
        """Phonemes with sufficient duration -> not degenerate."""
        # phoneme durations: 10, 8, 12, 6 -- all well above 1
        durations = np.array([0, 10, 0, 8, 0, 12, 0, 6, 0], dtype=np.int64)
        assert is_degenerate(durations) is False

    def test_threshold_boundary_79_percent(self):
        """79% short (below threshold 80%) -> not degenerate."""
        # 10 phonemes: 7 short (<= 1), 3 long -- 70% short
        # Make 100 phonemes for finer granularity: 79 short, 21 long
        n = 100
        length = 2 * n + 1
        arr = np.zeros(length, dtype=np.int64)
        for i in range(n):
            if i < 79:
                arr[2 * i + 1] = 1  # short
            else:
                arr[2 * i + 1] = 10  # long
        assert is_degenerate(arr) is False

    def test_threshold_boundary_80_percent(self):
        """Exactly 80% short -> degenerate (>= threshold)."""
        n = 100
        length = 2 * n + 1
        arr = np.zeros(length, dtype=np.int64)
        for i in range(n):
            if i < 80:
                arr[2 * i + 1] = 1  # short
            else:
                arr[2 * i + 1] = 10  # long
        assert is_degenerate(arr) is True

    def test_empty_durations_is_degenerate(self):
        """Empty array -> degenerate."""
        # Array with only blank (length 1): no phoneme positions
        durations = np.array([0], dtype=np.int64)
        assert is_degenerate(durations) is True

    def test_single_phoneme_long(self):
        """Single phoneme with long duration -> not degenerate."""
        durations = np.array([0, 50, 0], dtype=np.int64)
        assert is_degenerate(durations) is False

    def test_single_phoneme_short(self):
        """Single phoneme with duration=1 -> degenerate (100% <= 1)."""
        durations = np.array([0, 1, 0], dtype=np.int64)
        assert is_degenerate(durations) is True

    def test_custom_threshold(self):
        """Custom threshold_ratio changes the boundary."""
        # 5 phonemes: 3 short (60%), 2 long
        arr = np.array([0, 1, 0, 1, 0, 1, 0, 10, 0, 10, 0], dtype=np.int64)
        # 60% short: degenerate at threshold=0.5, not at threshold=0.8
        assert is_degenerate(arr, threshold_ratio=0.5) is True
        assert is_degenerate(arr, threshold_ratio=0.8) is False

    def test_zero_duration_phonemes_count_as_short(self):
        """Phonemes with duration=0 count as <= 1."""
        durations = np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.int64)
        assert is_degenerate(durations) is True


# ===========================================================================
# TestComputeDurationStats
# ===========================================================================


class TestComputeDurationStats:
    def test_basic_stats(self):
        """Mean, median, std, pct_le1, pct_le2 are computed correctly."""
        # phoneme durations: 1, 2, 3, 10, 20
        durations = np.array([0, 1, 0, 2, 0, 3, 0, 10, 0, 20, 0], dtype=np.int64)
        stats = compute_duration_stats(durations)

        ph = stats["phoneme_durations"]
        assert list(ph) == [1, 2, 3, 10, 20]
        assert stats["mean"] == pytest.approx(7.2)
        assert stats["median"] == pytest.approx(3.0)
        assert stats["std"] == pytest.approx(np.std([1, 2, 3, 10, 20]))

        # pct_le1: 1 out of 5 = 0.2
        assert stats["pct_le1"] == pytest.approx(0.2)
        # pct_le2: 2 out of 5 = 0.4
        assert stats["pct_le2"] == pytest.approx(0.4)

    def test_blank_positions_excluded(self):
        """Blank-position durations do not affect phoneme statistics."""
        # Blanks have large values, but phoneme stats should ignore them
        durations = np.array([100, 5, 100, 10, 100], dtype=np.int64)
        stats = compute_duration_stats(durations)

        assert list(stats["phoneme_durations"]) == [5, 10]
        assert stats["mean"] == pytest.approx(7.5)
        # Blanks captured separately
        assert list(stats["blank_durations"]) == [100, 100, 100]

    def test_blank0_duration(self):
        """First blank duration is correctly extracted."""
        durations = np.array([42, 5, 0, 10, 0], dtype=np.int64)
        stats = compute_duration_stats(durations)
        assert stats["blank0_duration"] == 42

    def test_blank0_zero(self):
        """First blank=0 is reported correctly."""
        durations = np.array([0, 5, 0, 10, 0], dtype=np.int64)
        stats = compute_duration_stats(durations)
        assert stats["blank0_duration"] == 0

    def test_is_degenerate_flag_true(self):
        """Degenerate flag is True for short durations."""
        durations = np.array([0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0], dtype=np.int64)
        stats = compute_duration_stats(durations)
        assert stats["is_degenerate"] is True

    def test_is_degenerate_flag_false(self):
        """Degenerate flag is False for healthy durations."""
        durations = np.array([0, 10, 0, 8, 0, 12, 0], dtype=np.int64)
        stats = compute_duration_stats(durations)
        assert stats["is_degenerate"] is False

    def test_no_phonemes(self):
        """Single-blank array (no phoneme positions) returns safe defaults."""
        durations = np.array([5], dtype=np.int64)
        stats = compute_duration_stats(durations)
        assert len(stats["phoneme_durations"]) == 0
        assert stats["mean"] == 0.0
        assert stats["is_degenerate"] is True
        assert stats["blank0_duration"] == 5


# ===========================================================================
# TestComputeCorpusStats
# ===========================================================================


class TestComputeCorpusStats:
    def test_multiple_utterances(self):
        """Aggregation over multiple utterances is correct."""
        dur1 = np.array([0, 5, 0, 10, 0], dtype=np.int64)   # healthy
        dur2 = np.array([0, 8, 0, 12, 0, 6, 0], dtype=np.int64)  # healthy
        stats = compute_corpus_stats([dur1, dur2])

        assert stats["total_samples"] == 2
        assert stats["degenerate_count"] == 0
        assert stats["degenerate_rate"] == 0.0

        # All phoneme durations: [5, 10, 8, 12, 6]
        ph = stats["phoneme_duration_stats"]
        assert ph["mean"] == pytest.approx(np.mean([5, 10, 8, 12, 6]))
        assert ph["median"] == pytest.approx(np.median([5, 10, 8, 12, 6]))
        assert ph["pct_le1"] == pytest.approx(0.0)  # none <= 1
        assert ph["pct_le2"] == pytest.approx(0.0)  # none <= 2

    def test_degenerate_rate_calculation(self):
        """Degenerate rate = degenerate_count / total."""
        healthy = np.array([0, 10, 0, 8, 0], dtype=np.int64)
        degenerate = np.array([0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0], dtype=np.int64)
        stats = compute_corpus_stats([healthy, degenerate])

        assert stats["total_samples"] == 2
        assert stats["degenerate_count"] == 1
        assert stats["degenerate_rate"] == pytest.approx(0.5)

    def test_empty_corpus(self):
        """Empty corpus returns zero-filled stats without error."""
        stats = compute_corpus_stats([])
        assert stats["total_samples"] == 0
        assert stats["degenerate_count"] == 0
        assert stats["degenerate_rate"] == 0.0
        assert stats["phoneme_duration_stats"]["mean"] == 0.0
        assert stats["blank_stats"]["mean_blank0"] == 0.0

    def test_blank_all_zero_rate(self):
        """all_zero_rate tracks fraction of utterances with all-zero blanks."""
        # Both have all-zero blanks
        d1 = np.array([0, 5, 0, 10, 0], dtype=np.int64)
        d2 = np.array([0, 8, 0], dtype=np.int64)
        stats = compute_corpus_stats([d1, d2])
        assert stats["blank_stats"]["all_zero_rate"] == pytest.approx(1.0)

        # One with non-zero blank
        d3 = np.array([3, 5, 0, 10, 0], dtype=np.int64)
        stats2 = compute_corpus_stats([d1, d3])
        assert stats2["blank_stats"]["all_zero_rate"] == pytest.approx(0.5)

    def test_mean_blank0(self):
        """mean_blank0 averages blank[0] across utterances."""
        d1 = np.array([0, 5, 0], dtype=np.int64)
        d2 = np.array([10, 5, 0], dtype=np.int64)
        stats = compute_corpus_stats([d1, d2])
        assert stats["blank_stats"]["mean_blank0"] == pytest.approx(5.0)

    def test_single_utterance_corpus(self):
        """Single utterance corpus computes correctly."""
        dur = np.array([0, 7, 0, 3, 0, 12, 0], dtype=np.int64)
        stats = compute_corpus_stats([dur])
        assert stats["total_samples"] == 1
        assert stats["degenerate_count"] == 0
        ph = stats["phoneme_duration_stats"]
        assert ph["mean"] == pytest.approx(np.mean([7, 3, 12]))


# ===========================================================================
# TestComputeCorpusStatsStreaming
# ===========================================================================


class TestComputeCorpusStatsStreaming:
    def test_matches_batch_on_basic_input(self, tmp_path):
        """Streaming stats match batch stats for mean, std, pct_le1, pct_le2."""
        dur1 = np.array([0, 5, 0, 10, 0], dtype=np.int64)
        dur2 = np.array([0, 8, 0, 12, 0, 6, 0], dtype=np.int64)

        p1 = tmp_path / "d1.npy"
        p2 = tmp_path / "d2.npy"
        np.save(str(p1), dur1)
        np.save(str(p2), dur2)

        batch = compute_corpus_stats([dur1, dur2])
        streaming = compute_corpus_stats_streaming([p1, p2])

        assert streaming["total_samples"] == batch["total_samples"]
        assert streaming["degenerate_count"] == batch["degenerate_count"]
        assert streaming["degenerate_rate"] == pytest.approx(batch["degenerate_rate"])

        # Phoneme stats: mean and std should match (Welford vs batch numpy)
        b_ph = batch["phoneme_duration_stats"]
        s_ph = streaming["phoneme_duration_stats"]
        assert s_ph["mean"] == pytest.approx(b_ph["mean"], abs=1e-6)
        assert s_ph["std"] == pytest.approx(b_ph["std"], abs=1e-6)
        assert s_ph["pct_le1"] == pytest.approx(b_ph["pct_le1"])
        assert s_ph["pct_le2"] == pytest.approx(b_ph["pct_le2"])

        # Blank stats
        assert streaming["blank_stats"]["mean_blank0"] == pytest.approx(
            batch["blank_stats"]["mean_blank0"]
        )
        assert streaming["blank_stats"]["all_zero_rate"] == pytest.approx(
            batch["blank_stats"]["all_zero_rate"]
        )

    def test_degenerate_detection(self, tmp_path):
        """Streaming correctly identifies degenerate utterances."""
        healthy = np.array([0, 10, 0, 8, 0], dtype=np.int64)
        degenerate = np.array([0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0], dtype=np.int64)

        p1 = tmp_path / "h.npy"
        p2 = tmp_path / "d.npy"
        np.save(str(p1), healthy)
        np.save(str(p2), degenerate)

        stats = compute_corpus_stats_streaming([p1, p2])
        assert stats["degenerate_count"] == 1
        assert stats["degenerate_rate"] == pytest.approx(0.5)

    def test_empty_input(self):
        """Empty list returns zero-filled stats."""
        stats = compute_corpus_stats_streaming([])
        assert stats["total_samples"] == 0
        assert stats["degenerate_count"] == 0
        assert stats["phoneme_duration_stats"]["mean"] == 0.0

    def test_single_utterance(self, tmp_path):
        """Single-file streaming produces correct results."""
        dur = np.array([0, 7, 0, 3, 0, 12, 0], dtype=np.int64)
        p = tmp_path / "single.npy"
        np.save(str(p), dur)

        stats = compute_corpus_stats_streaming([p])
        assert stats["total_samples"] == 1
        assert stats["degenerate_count"] == 0
        assert stats["phoneme_duration_stats"]["mean"] == pytest.approx(
            np.mean([7, 3, 12]), abs=1e-6
        )

    def test_no_median_key(self, tmp_path):
        """Streaming stats omit median (cannot be computed in streaming)."""
        dur = np.array([0, 5, 0, 10, 0], dtype=np.int64)
        p = tmp_path / "d.npy"
        np.save(str(p), dur)

        stats = compute_corpus_stats_streaming([p])
        assert "median" not in stats["phoneme_duration_stats"]


# ===========================================================================
# TestComputePhonemeClassStats
# ===========================================================================


class TestComputePhonemeClassStats:
    def test_basic_class_grouping(self):
        """Durations are grouped by phoneme symbol."""
        # Simulated symbols_ja mapping (subset):
        #   0 -> "~" (pad), 1 -> "a", 2 -> "k", 3 -> "i"
        id_to_sym = {0: "~", 1: "a", 2: "k", 3: "i"}
        # 3 phonemes: a, k, i with durations 5, 10, 8
        durations = np.array([0, 5, 0, 10, 0, 8, 0], dtype=np.int64)
        phoneme_ids = [1, 2, 3]

        result = compute_phoneme_class_stats(durations, phoneme_ids, id_to_sym)

        assert result["a"] == [5]
        assert result["k"] == [10]
        assert result["i"] == [8]

    def test_repeated_phoneme_collects_all(self):
        """Multiple occurrences of the same phoneme are collected together."""
        id_to_sym = {1: "a", 2: "k"}
        # phonemes: a, k, a -> durations 5, 10, 7
        durations = np.array([0, 5, 0, 10, 0, 7, 0], dtype=np.int64)
        phoneme_ids = [1, 2, 1]

        result = compute_phoneme_class_stats(durations, phoneme_ids, id_to_sym)

        assert sorted(result["a"]) == [5, 7]
        assert result["k"] == [10]

    def test_unknown_id_labeled(self):
        """Phoneme IDs not in id_to_symbol get a fallback label."""
        id_to_sym = {1: "a"}
        durations = np.array([0, 5, 0, 10, 0], dtype=np.int64)
        phoneme_ids = [1, 99]  # 99 not in map

        result = compute_phoneme_class_stats(durations, phoneme_ids, id_to_sym)
        assert "a" in result
        assert "<99>" in result

    def test_empty_phoneme_ids(self):
        """Empty phoneme_ids with a blank-only array returns empty dict."""
        durations = np.array([0], dtype=np.int64)
        result = compute_phoneme_class_stats(durations, [], {})
        assert result == {}

    def test_overflow_handled(self):
        """If phoneme_ids shorter than phoneme positions, overflow is labeled."""
        id_to_sym = {1: "a"}
        # 2 phoneme positions but only 1 id
        durations = np.array([0, 5, 0, 10, 0], dtype=np.int64)
        phoneme_ids = [1]

        result = compute_phoneme_class_stats(durations, phoneme_ids, id_to_sym)
        assert "a" in result
        assert "<overflow>" in result
