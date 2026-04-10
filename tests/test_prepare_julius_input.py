"""Tests for scripts/prepare_julius_input.py.

Covers:
  - Filelist parsing
  - Text-to-katakana conversion (pyopenjtalk)
  - Audio resampling to 16kHz
  - Output directory structure and filename format
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

# Add scripts/ to path so we can import the module under test
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from prepare_julius_input import (  # noqa: E402
    JULIUS_SAMPLE_RATE,
    make_output_name,
    parse_filelist,
    resample_to_16k,
    text_to_katakana,
)

# Check if pyopenjtalk is available
try:
    import pyopenjtalk  # noqa: F401

    has_pyopenjtalk = True
except ImportError:
    has_pyopenjtalk = False


# ---------------------------------------------------------------------------
# 1. Filelist parsing
# ---------------------------------------------------------------------------


class TestParseFilelist:
    """Test pipe-delimited filelist parsing."""

    def test_parse_filelist(self, tmp_path):
        """Standard 3-field lines should be parsed correctly."""
        filelist = tmp_path / "test.txt"
        filelist.write_text(
            "/data/wavs/jvs001/BASIC5000_0025.wav|0|こんにちは\n"
            "/data/wavs/jvs002/TRAVEL1000_0001.wav|1|今日はいい天気ですね。\n",
            encoding="utf-8",
        )
        entries = parse_filelist(str(filelist))
        assert len(entries) == 2
        assert entries[0] == (
            "/data/wavs/jvs001/BASIC5000_0025.wav",
            "0",
            "こんにちは",
        )
        assert entries[1] == (
            "/data/wavs/jvs002/TRAVEL1000_0001.wav",
            "1",
            "今日はいい天気ですね。",
        )

    def test_parse_filelist_skips_blank_lines(self, tmp_path):
        """Blank lines should be skipped."""
        filelist = tmp_path / "test.txt"
        filelist.write_text(
            "/data/wavs/jvs001/A.wav|0|テスト\n"
            "\n"
            "  \n"
            "/data/wavs/jvs001/B.wav|0|テスト2\n",
            encoding="utf-8",
        )
        entries = parse_filelist(str(filelist))
        assert len(entries) == 2

    def test_parse_filelist_skips_malformed_lines(self, tmp_path):
        """Lines with wrong number of fields should be skipped."""
        filelist = tmp_path / "test.txt"
        filelist.write_text(
            "/data/wavs/jvs001/A.wav|0|テスト\n"
            "bad_line_no_pipes\n"
            "/data/wavs/jvs001/B.wav|0|テスト2\n",
            encoding="utf-8",
        )
        entries = parse_filelist(str(filelist))
        assert len(entries) == 2


# ---------------------------------------------------------------------------
# 2. Text-to-katakana conversion
# ---------------------------------------------------------------------------


class TestTextToKatakana:
    """Test katakana conversion via pyopenjtalk."""

    @pytest.fixture(autouse=True)
    def _require_pyopenjtalk(self):
        if not has_pyopenjtalk:
            pytest.skip("pyopenjtalk not installed")

    def test_text_to_katakana_basic(self):
        """Basic Japanese text should produce katakana output."""
        result = text_to_katakana("こんにちは")
        assert isinstance(result, str)
        assert len(result) > 0
        # Should contain katakana characters
        assert any("\u30A0" <= ch <= "\u30FF" for ch in result)

    def test_text_to_katakana_kanji(self):
        """Kanji text should be converted to katakana."""
        result = text_to_katakana("東京")
        assert isinstance(result, str)
        assert len(result) > 0
        # No kanji should remain
        assert not any("\u4E00" <= ch <= "\u9FFF" for ch in result)

    def test_text_to_katakana_punctuation_removed(self):
        """Punctuation marks should be removed from the output."""
        result = text_to_katakana("歯医者に見ていただく必要がありますか。")
        assert "。" not in result
        assert "、" not in result
        assert "." not in result
        assert "," not in result
        # Should still have katakana content
        assert len(result) > 0

    def test_text_to_katakana_mixed_punctuation(self):
        """Various punctuation types should all be removed."""
        result = text_to_katakana("これは、テストです。")
        assert "、" not in result
        assert "。" not in result
        assert len(result) > 0

    def test_text_to_katakana_returns_string(self):
        """Return type should always be a string."""
        result = text_to_katakana("テスト")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# 3. Audio resampling
# ---------------------------------------------------------------------------


class TestResampleTo16k:
    """Test audio resampling to 16kHz."""

    def _make_wav(self, path, sr, duration_sec=1.0, channels=1):
        """Create a dummy WAV file with a sine wave."""
        n_samples = int(sr * duration_sec)
        t = np.linspace(0, duration_sec, n_samples, endpoint=False, dtype=np.float32)
        data = 0.5 * np.sin(2 * np.pi * 440.0 * t)
        if channels == 2:
            data = np.column_stack([data, data])
        sf.write(str(path), data, sr)
        return data

    def test_resample_to_16k_output_sample_rate(self, tmp_path):
        """Output WAV should have 16kHz sample rate."""
        src = tmp_path / "input.wav"
        dst = tmp_path / "output.wav"
        self._make_wav(src, sr=22050)

        resample_to_16k(str(src), str(dst))

        data, sr = sf.read(str(dst))
        assert sr == JULIUS_SAMPLE_RATE
        assert sr == 16000

    def test_resample_to_16k_mono(self, tmp_path):
        """Stereo input should be converted to mono."""
        src = tmp_path / "stereo.wav"
        dst = tmp_path / "mono.wav"
        self._make_wav(src, sr=22050, channels=2)

        resample_to_16k(str(src), str(dst))

        data, sr = sf.read(str(dst))
        assert sr == 16000
        # Output should be 1D (mono)
        assert data.ndim == 1

    def test_resample_preserves_duration(self, tmp_path):
        """Duration should be preserved within 10ms tolerance after resampling."""
        duration_sec = 1.5
        src = tmp_path / "input.wav"
        dst = tmp_path / "output.wav"
        self._make_wav(src, sr=22050, duration_sec=duration_sec)

        resample_to_16k(str(src), str(dst))

        data, sr = sf.read(str(dst))
        output_duration = len(data) / sr
        assert abs(output_duration - duration_sec) < 0.01  # 10ms tolerance

    def test_resample_already_16k(self, tmp_path):
        """If input is already 16kHz, output should still be valid 16kHz."""
        src = tmp_path / "already16k.wav"
        dst = tmp_path / "output.wav"
        self._make_wav(src, sr=16000)

        resample_to_16k(str(src), str(dst))

        data, sr = sf.read(str(dst))
        assert sr == 16000

    def test_resample_from_48k(self, tmp_path):
        """Resampling from 48kHz should work correctly."""
        src = tmp_path / "hires.wav"
        dst = tmp_path / "output.wav"
        duration_sec = 1.0
        self._make_wav(src, sr=48000, duration_sec=duration_sec)

        resample_to_16k(str(src), str(dst))

        data, sr = sf.read(str(dst))
        assert sr == 16000
        output_duration = len(data) / sr
        assert abs(output_duration - duration_sec) < 0.01


# ---------------------------------------------------------------------------
# 4. Output directory structure and filename format
# ---------------------------------------------------------------------------


class TestOutputStructure:
    """Test output directory structure and filename conventions."""

    def test_output_directory_structure(self, tmp_path):
        """The output should create wav/ and txt/ subdirectories."""
        wav_dir = tmp_path / "wav"
        txt_dir = tmp_path / "txt"
        wav_dir.mkdir()
        txt_dir.mkdir()

        assert wav_dir.is_dir()
        assert txt_dir.is_dir()

    def test_filename_format(self):
        """Filename should be {spk}_{utt_id} derived from the wav path."""
        wav_path = "/data/Matcha-TTS/data/jvs/wavs/jvs001/BASIC5000_0025.wav"
        name = make_output_name(wav_path)
        assert name == "jvs001_BASIC5000_0025"

    def test_filename_format_different_speaker(self):
        """Filename format should work for different speakers."""
        wav_path = "/data/Matcha-TTS/data/jvs/wavs/jvs099/TRAVEL1000_0100.wav"
        name = make_output_name(wav_path)
        assert name == "jvs099_TRAVEL1000_0100"

    def test_filename_format_voiceactress(self):
        """Filename format should work for VOICEACTRESS utterances."""
        wav_path = "/data/Matcha-TTS/data/jvs/wavs/jvs050/VOICEACTRESS100_042.wav"
        name = make_output_name(wav_path)
        assert name == "jvs050_VOICEACTRESS100_042"

    def test_end_to_end_file_creation(self, tmp_path):
        """End-to-end test: create input, process, verify output files."""
        if not has_pyopenjtalk:
            pytest.skip("pyopenjtalk not installed")

        # Create a dummy filelist
        wav_dir_src = tmp_path / "wavs" / "jvs001"
        wav_dir_src.mkdir(parents=True)
        wav_path = wav_dir_src / "TEST_0001.wav"
        sf.write(str(wav_path), np.zeros(22050, dtype="float32"), 22050)

        filelist = tmp_path / "filelist.txt"
        filelist.write_text(
            f"{wav_path}|0|こんにちは\n",
            encoding="utf-8",
        )

        # Create output directories
        out_dir = tmp_path / "output"
        wav_out = out_dir / "wav"
        txt_out = out_dir / "txt"
        wav_out.mkdir(parents=True)
        txt_out.mkdir(parents=True)

        # Process the single entry
        entries = parse_filelist(str(filelist))
        assert len(entries) == 1

        wav_p, spk_id, text = entries[0]
        name = make_output_name(wav_p)

        # Resample
        resample_to_16k(wav_p, str(wav_out / f"{name}.wav"))
        assert (wav_out / f"{name}.wav").exists()

        # Convert text
        kana = text_to_katakana(text)
        (txt_out / f"{name}.txt").write_text(kana, encoding="utf-8")
        assert (txt_out / f"{name}.txt").exists()

        # Verify name format
        assert name == "jvs001_TEST_0001"

        # Verify output audio
        data, sr = sf.read(str(wav_out / f"{name}.wav"))
        assert sr == 16000

        # Verify text content
        content = (txt_out / f"{name}.txt").read_text(encoding="utf-8")
        assert len(content) > 0
        assert "。" not in content
