"""Integration tests for scripts/verify_alignment_quality.py.

Creates temporary duration files and filelists, then verifies that the
report generation and CLI work correctly.

pyopenjtalk is required for text_to_sequence (skipped if not installed).
"""

import json
import sys
import types

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Mock phonemizer before any matcha.text import (same pattern as other tests)
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

# Now safe to import ---------------------------------------------------------
from matcha.text import text_to_sequence  # noqa: E402
from matcha.utils.utils import intersperse  # noqa: E402
from scripts.verify_alignment_quality import (  # noqa: E402
    classify_phoneme,
    generate_report,
    main,
    make_output_name,
    parse_filelist,
)

# All tests require pyopenjtalk for text_to_sequence
pytestmark = pytest.mark.skipif(
    not pytest.importorskip("pyopenjtalk", reason="pyopenjtalk not installed"),
    reason="pyopenjtalk not installed",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_duration_for_text(text: str, *, healthy: bool = True) -> np.ndarray:
    """Create a duration array matching the interspersed length of *text*.

    If healthy=True, phoneme positions get duration=7.
    If healthy=False, phoneme positions get duration=1 (degenerate).
    """
    seq, _ = text_to_sequence(text, ["japanese_cleaners"], language="ja")
    interspersed = intersperse(seq, 0)
    n_total = len(interspersed)
    arr = np.zeros(n_total, dtype=np.int64)
    dur_val = 7 if healthy else 1
    # Odd indices = phoneme positions
    arr[1::2] = dur_val
    return arr


def _create_test_data(tmp_path, texts_and_health, *, speaker="jvs001"):
    """Create filelist + .npy files for testing.

    Args:
        texts_and_health: list of (text, healthy_bool) tuples.

    Returns:
        (filelist_path, duration_dir)
    """
    dur_dir = tmp_path / "durations"
    dur_dir.mkdir()
    filelist_path = tmp_path / "test.txt"

    lines = []
    for i, (text, healthy) in enumerate(texts_and_health):
        utt_id = f"UTT{i:04d}"
        wav_path = f"/data/wavs/{speaker}/{utt_id}.wav"
        spk_id = "0"
        lines.append(f"{wav_path}|{spk_id}|{text}")

        name = make_output_name(wav_path)
        dur = _make_duration_for_text(text, healthy=healthy)
        np.save(str(dur_dir / f"{name}.npy"), dur)

    filelist_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(filelist_path), dur_dir


# ===========================================================================
# TestParseFilelist
# ===========================================================================


class TestParseFilelist:
    def test_basic(self, tmp_path):
        fl = tmp_path / "fl.txt"
        fl.write_text("/a/b/c.wav|0|hello\n/d/e/f.wav|1|world\n")
        entries = parse_filelist(str(fl))
        assert len(entries) == 2
        assert entries[0] == ("/a/b/c.wav", "0", "hello")

    def test_empty_lines_skipped(self, tmp_path):
        fl = tmp_path / "fl.txt"
        fl.write_text("\n/a/b.wav|0|text\n\n")
        entries = parse_filelist(str(fl))
        assert len(entries) == 1


# ===========================================================================
# TestClassifyPhoneme
# ===========================================================================


class TestClassifyPhoneme:
    def test_vowels(self):
        for v in ["a", "i", "u", "e", "o", "A", "I", "U", "E", "O"]:
            assert classify_phoneme(v) == "vowel"

    def test_hatsuon(self):
        assert classify_phoneme("N") == "hatsuon"

    def test_sokuon(self):
        assert classify_phoneme("cl") == "sokuon"

    def test_pause(self):
        assert classify_phoneme("pau") == "pause"
        assert classify_phoneme("_") == "pause"

    def test_silence(self):
        assert classify_phoneme("sil") == "silence"
        assert classify_phoneme("^") == "silence"
        assert classify_phoneme("$") == "silence"
        # ? = interrogative-final sil (carries duration, unlike prosody markers)
        assert classify_phoneme("?") == "silence"

    def test_prosody(self):
        for p in ["#", "[", "]"]:
            assert classify_phoneme(p) == "prosody"

    def test_consonant(self):
        for c in ["k", "s", "t", "n", "h", "m", "r", "w", "y", "ch", "sh", "ts"]:
            assert classify_phoneme(c) == "consonant"


# ===========================================================================
# TestVerifyAlignmentQuality (integration)
# ===========================================================================


class TestVerifyAlignmentQuality:
    def test_report_generation(self, tmp_path):
        """Healthy duration files produce a correct report."""
        fl_path, dur_dir = _create_test_data(
            tmp_path,
            [
                ("こんにちは", True),
                ("ありがとう", True),
            ],
        )
        report = generate_report(dur_dir, fl_path)

        assert report["total_in_filelist"] == 2
        assert report["total_loaded"] == 2
        assert report["missing_files"] == 0
        assert report["corpus_stats"]["degenerate_count"] == 0
        assert report["corpus_stats"]["degenerate_rate"] == 0.0
        assert len(report["length_mismatches"]) == 0
        assert len(report["degenerate_samples"]) == 0

        # Phoneme duration mean should be 7.0 (all phonemes set to 7)
        assert report["corpus_stats"]["phoneme_duration"]["mean"] == pytest.approx(7.0)
        # All blanks are zero
        assert report["corpus_stats"]["blank_stats"]["all_zero_rate"] == pytest.approx(1.0)

    def test_missing_duration_reported(self, tmp_path):
        """Missing .npy files are counted in the report."""
        dur_dir = tmp_path / "durations"
        dur_dir.mkdir()
        fl_path = tmp_path / "fl.txt"
        # Filelist references a file, but no .npy exists
        fl_path.write_text("/data/wavs/jvs001/UTT0000.wav|0|こんにちは\n", encoding="utf-8")

        report = generate_report(dur_dir, str(fl_path))

        assert report["total_in_filelist"] == 1
        assert report["total_loaded"] == 0
        assert report["missing_files"] == 1

    def test_degenerate_detection(self, tmp_path):
        """Degenerate samples are correctly detected and listed."""
        fl_path, dur_dir = _create_test_data(
            tmp_path,
            [
                ("こんにちは", True),  # healthy
                ("ありがとう", False),  # degenerate (all dur=1)
            ],
        )
        report = generate_report(dur_dir, fl_path)

        assert report["corpus_stats"]["degenerate_count"] == 1
        assert report["corpus_stats"]["degenerate_rate"] == pytest.approx(0.5)
        assert len(report["degenerate_samples"]) == 1

    def test_phoneme_class_stats_present(self, tmp_path):
        """Phoneme class stats are included in the report."""
        fl_path, dur_dir = _create_test_data(
            tmp_path,
            [("こんにちは", True)],
        )
        report = generate_report(dur_dir, fl_path)

        pcs = report["phoneme_class_stats"]
        # Should have at least vowel and consonant classes
        assert len(pcs) > 0
        # Each class has the expected keys
        for cls_stats in pcs.values():
            assert "count" in cls_stats
            assert "mean" in cls_stats
            assert "median" in cls_stats

    def test_json_output_via_cli(self, tmp_path):
        """CLI writes valid JSON report."""
        fl_path, dur_dir = _create_test_data(
            tmp_path,
            [("こんにちは", True)],
        )
        report_path = tmp_path / "report.json"

        exit_code = main(
            [
                "--duration-dir",
                str(dur_dir),
                "--filelist",
                fl_path,
                "--output-report",
                str(report_path),
            ]
        )

        assert exit_code == 0
        assert report_path.exists()
        with open(report_path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["total_in_filelist"] == 1
        assert data["total_loaded"] == 1

    def test_cli_missing_dir_returns_error(self, tmp_path):
        """CLI returns non-zero when duration directory does not exist."""
        fl_path = tmp_path / "fl.txt"
        fl_path.write_text("/a/b.wav|0|test\n")

        exit_code = main(
            [
                "--duration-dir",
                str(tmp_path / "nonexistent"),
                "--filelist",
                str(fl_path),
            ]
        )
        assert exit_code == 1

    def test_length_mismatch_detected(self, tmp_path):
        """Duration array with wrong length is reported as a mismatch."""
        dur_dir = tmp_path / "durations"
        dur_dir.mkdir()
        fl_path = tmp_path / "fl.txt"

        text = "こんにちは"
        wav_path = "/data/wavs/jvs001/UTT0000.wav"
        fl_path.write_text(f"{wav_path}|0|{text}\n", encoding="utf-8")

        name = make_output_name(wav_path)
        # Create a duration array with wrong length (too short)
        wrong_dur = np.array([0, 5, 0], dtype=np.int64)
        np.save(str(dur_dir / f"{name}.npy"), wrong_dur)

        report = generate_report(dur_dir, str(fl_path))

        assert len(report["length_mismatches"]) == 1
        assert report["length_mismatches"][0]["name"] == name
