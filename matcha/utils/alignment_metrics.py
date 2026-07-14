"""Alignment quality metrics for duration arrays.

Shared library for M1 (Julius alignment quality verification) and
M5 (post-training evaluation). Computes degenerate alignment rates and
phoneme duration statistics using the same criteria documented in CLAUDE.md.

All functions are pure numpy -- no torch dependency.
"""

from __future__ import annotations

import numpy as np


def is_degenerate(durations: np.ndarray, threshold_ratio: float = 0.8) -> bool:
    """Determine whether a duration array represents a degenerate alignment.

    An alignment is considered degenerate when ``threshold_ratio`` or more of
    the phoneme positions (odd indices in the blank-interspersed array) have
    duration <= 1 frame.  This matches the MAS degenerate criterion from
    CLAUDE.md.

    Args:
        durations: Blank-interspersed duration array of shape ``(2*N+1,)``.
        threshold_ratio: Fraction of short phonemes that triggers degenerate
            classification (default 0.8).

    Returns:
        ``True`` if the alignment is degenerate.
    """
    phoneme_durations = durations[1::2]  # odd indices = phoneme positions
    if len(phoneme_durations) == 0:
        return True
    n_short = int(np.sum(phoneme_durations <= 1))
    return (n_short / len(phoneme_durations)) >= threshold_ratio


def compute_duration_stats(durations: np.ndarray) -> dict:
    """Compute per-utterance statistics from a blank-interspersed duration array.

    Args:
        durations: Blank-interspersed duration array of shape ``(2*N+1,)``.

    Returns:
        Dictionary with keys:

        - ``phoneme_durations``: 1-D array of phoneme-position durations.
        - ``blank_durations``: 1-D array of blank-position durations.
        - ``mean``: Mean phoneme duration (float).
        - ``median``: Median phoneme duration (float).
        - ``std``: Standard deviation of phoneme durations (float).
        - ``pct_le1``: Fraction of phonemes with duration <= 1 frame.
        - ``pct_le2``: Fraction of phonemes with duration <= 2 frames.
        - ``blank0_duration``: Duration of the first blank (index 0).
        - ``is_degenerate``: Boolean degenerate flag.
    """
    phoneme_durations = durations[1::2].copy()
    blank_durations = durations[0::2].copy()

    n_ph = len(phoneme_durations)
    if n_ph == 0:
        return {
            "phoneme_durations": phoneme_durations,
            "blank_durations": blank_durations,
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "pct_le1": 1.0,
            "pct_le2": 1.0,
            "blank0_duration": int(durations[0]) if len(durations) > 0 else 0,
            "is_degenerate": True,
        }

    return {
        "phoneme_durations": phoneme_durations,
        "blank_durations": blank_durations,
        "mean": float(np.mean(phoneme_durations)),
        "median": float(np.median(phoneme_durations)),
        "std": float(np.std(phoneme_durations)),
        "pct_le1": float(np.sum(phoneme_durations <= 1) / n_ph),
        "pct_le2": float(np.sum(phoneme_durations <= 2) / n_ph),
        "blank0_duration": int(durations[0]),
        "is_degenerate": is_degenerate(durations),
    }


def compute_corpus_stats(all_durations: list[np.ndarray]) -> dict:
    """Aggregate duration statistics across a whole corpus.

    Args:
        all_durations: List of blank-interspersed duration arrays, one per
            utterance.

    Returns:
        Dictionary with keys:

        - ``total_samples``: Number of utterances.
        - ``degenerate_count``: Number of degenerate utterances.
        - ``degenerate_rate``: Fraction of degenerate utterances.
        - ``phoneme_duration_stats``: ``{mean, median, std, pct_le1, pct_le2}``
          computed over *all* phoneme positions across the corpus.
        - ``blank_stats``: ``{mean_blank0, all_zero_rate}`` where
          ``all_zero_rate`` is the fraction of utterances whose blank
          positions are entirely zero.
    """
    total = len(all_durations)
    if total == 0:
        return {
            "total_samples": 0,
            "degenerate_count": 0,
            "degenerate_rate": 0.0,
            "phoneme_duration_stats": {
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
                "pct_le1": 0.0,
                "pct_le2": 0.0,
            },
            "blank_stats": {
                "mean_blank0": 0.0,
                "all_zero_rate": 0.0,
            },
        }

    degenerate_count = 0
    all_phoneme_durs: list[np.ndarray] = []
    blank0_values: list[int] = []
    all_blank_zero_count = 0

    for dur in all_durations:
        if is_degenerate(dur):
            degenerate_count += 1

        ph_durs = dur[1::2]
        if len(ph_durs) > 0:
            all_phoneme_durs.append(ph_durs)

        blank0_values.append(int(dur[0]) if len(dur) > 0 else 0)

        blank_durs = dur[0::2]
        if len(blank_durs) > 0 and np.all(blank_durs == 0):
            all_blank_zero_count += 1

    if all_phoneme_durs:
        concat_ph = np.concatenate(all_phoneme_durs)
        n_ph = len(concat_ph)
        ph_stats = {
            "mean": float(np.mean(concat_ph)),
            "median": float(np.median(concat_ph)),
            "std": float(np.std(concat_ph)),
            "pct_le1": float(np.sum(concat_ph <= 1) / n_ph),
            "pct_le2": float(np.sum(concat_ph <= 2) / n_ph),
        }
    else:
        ph_stats = {
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "pct_le1": 0.0,
            "pct_le2": 0.0,
        }

    return {
        "total_samples": total,
        "degenerate_count": degenerate_count,
        "degenerate_rate": degenerate_count / total,
        "phoneme_duration_stats": ph_stats,
        "blank_stats": {
            "mean_blank0": float(np.mean(blank0_values)),
            "all_zero_rate": all_blank_zero_count / total,
        },
    }


def compute_corpus_stats_streaming(duration_paths: list) -> dict:
    """Memory-efficient streaming corpus statistics using Welford's algorithm.

    Instead of loading all duration arrays into memory at once, processes
    each file one at a time and computes running statistics.

    Args:
        duration_paths: List of ``Path`` objects pointing to ``.npy`` duration
            files.

    Returns:
        Dictionary with the same keys as :func:`compute_corpus_stats`:

        - ``total_samples``: Number of utterances processed.
        - ``degenerate_count``: Number of degenerate utterances.
        - ``degenerate_rate``: Fraction of degenerate utterances.
        - ``phoneme_duration_stats``: ``{mean, std, pct_le1, pct_le2}``
          computed over all phoneme positions across the corpus (median is
          omitted since it cannot be computed in a streaming fashion).
        - ``blank_stats``: ``{mean_blank0, all_zero_rate}``.
    """
    total = 0
    degenerate_count = 0

    # Welford's online algorithm for mean/variance of phoneme durations
    ph_count = 0  # total number of phoneme positions
    ph_mean = 0.0
    ph_m2 = 0.0  # sum of squared deviations
    ph_le1_count = 0
    ph_le2_count = 0

    blank0_sum = 0.0
    all_blank_zero_count = 0

    for path in duration_paths:
        dur = np.load(str(path))
        total += 1

        if is_degenerate(dur):
            degenerate_count += 1

        ph_durs = dur[1::2]
        for d in ph_durs:
            ph_count += 1
            delta = float(d) - ph_mean
            ph_mean += delta / ph_count
            delta2 = float(d) - ph_mean
            ph_m2 += delta * delta2

            if d <= 1:
                ph_le1_count += 1
            if d <= 2:
                ph_le2_count += 1

        blank0_sum += int(dur[0]) if len(dur) > 0 else 0

        blank_durs = dur[0::2]
        if len(blank_durs) > 0 and np.all(blank_durs == 0):
            all_blank_zero_count += 1

    if total == 0:
        return {
            "total_samples": 0,
            "degenerate_count": 0,
            "degenerate_rate": 0.0,
            "phoneme_duration_stats": {
                "mean": 0.0,
                "std": 0.0,
                "pct_le1": 0.0,
                "pct_le2": 0.0,
            },
            "blank_stats": {
                "mean_blank0": 0.0,
                "all_zero_rate": 0.0,
            },
        }

    ph_std = (ph_m2 / ph_count) ** 0.5 if ph_count > 0 else 0.0
    ph_stats = {
        "mean": ph_mean if ph_count > 0 else 0.0,
        "std": ph_std,
        "pct_le1": ph_le1_count / ph_count if ph_count > 0 else 0.0,
        "pct_le2": ph_le2_count / ph_count if ph_count > 0 else 0.0,
    }

    return {
        "total_samples": total,
        "degenerate_count": degenerate_count,
        "degenerate_rate": degenerate_count / total,
        "phoneme_duration_stats": ph_stats,
        "blank_stats": {
            "mean_blank0": blank0_sum / total,
            "all_zero_rate": all_blank_zero_count / total,
        },
    }


def compute_phoneme_class_stats(
    durations: np.ndarray,
    phoneme_ids: list[int],
    id_to_symbol: dict[int, str],
) -> dict[str, list[int]]:
    """Collect duration values grouped by phoneme symbol.

    Args:
        durations: Blank-interspersed duration array of shape ``(2*N+1,)``.
        phoneme_ids: Phoneme ID sequence **before** intersperse (length N).
        id_to_symbol: Mapping from phoneme ID to symbol string.

    Returns:
        ``{symbol: [dur_frame, ...], ...}`` collecting every occurrence of
        each symbol across the utterance.
    """
    phoneme_durs = durations[1::2]  # length N
    result: dict[str, list[int]] = {}
    for idx, dur in enumerate(phoneme_durs):
        if idx < len(phoneme_ids):
            sym = id_to_symbol.get(phoneme_ids[idx], f"<{phoneme_ids[idx]}>")
        else:
            sym = "<overflow>"
        result.setdefault(sym, []).append(int(dur))
    return result
