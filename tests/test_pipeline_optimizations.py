"""Tests for preprocessing pipeline optimizations (Tier 1 + Tier 2 + Tier 3).

Covers:
  - T1-2: sf.info-based mel_frames calculation (formula + real data)
  - T1-3: Text cache (hiragana conversion, pickle round-trip, dedup)
  - T2-1: Dual resample in prepare_jvs (22kHz + 16kHz Julius output)
  - T2-2: precompute_with_alignment.py unified .pt generation
  - T3 (fast path): build_tasks / build_text_sequence_cache / _load_one /
    _mel_batch_gpu (cpu) / _finalize_one / run_fast_pipeline e2e + legacy parity
  - run_julius_alignment.py: check_prerequisites + run_segkit_batch layout regression
  - prepare_jvs.py: trim_silence edge cases
  - Syntax/import/CLI validation for pipeline scripts
"""

import ast
import inspect
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

# ---------------------------------------------------------------------------
# phonemizer mock (same pattern as test_text_ja.py)
# Must be installed before any matcha.text import occurs.
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

# Add scripts/ to path for imports
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# Constants
HOP_LENGTH = 256
SAMPLE_RATE = 22050


# ===========================================================================
# T1-2: sf.info mel_frames calculation
# ===========================================================================


class TestMelFramesFromAudioInfo:
    """Verify mel_frames can be computed from audio file metadata."""

    def test_formula_correctness_synthetic(self):
        """Basic formula: frames = audio_samples // hop_length on synthetic audio."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            duration_sec = 3
            total_samples = SAMPLE_RATE * duration_sec
            samples = np.random.randn(total_samples).astype(np.float32)
            sf.write(f.name, samples, SAMPLE_RATE)
            info = sf.info(f.name)
            expected = info.frames // HOP_LENGTH
            assert info.frames == total_samples
            assert expected == total_samples // HOP_LENGTH

    def test_formula_various_lengths(self):
        """Formula works for various audio lengths (1s, 5s, 10s)."""
        for duration_sec in [1, 5, 10]:
            total_samples = SAMPLE_RATE * duration_sec
            samples = np.random.randn(total_samples).astype(np.float32)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, samples, SAMPLE_RATE)
                info = sf.info(f.name)
                assert info.frames // HOP_LENGTH == total_samples // HOP_LENGTH

    def test_formula_non_aligned_length(self):
        """Formula works when sample count is not a multiple of hop_length."""
        # 22050 * 2.5 = 55125 samples, 55125 // 256 = 215 frames
        total_samples = int(SAMPLE_RATE * 2.5)
        samples = np.random.randn(total_samples).astype(np.float32)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, samples, SAMPLE_RATE)
            info = sf.info(f.name)
            assert info.frames == total_samples
            assert info.frames // HOP_LENGTH == total_samples // HOP_LENGTH

    @pytest.mark.slow
    def test_mel_frames_matches_pt_real_data(self):
        """mel_frames from sf.info should match mel shape in .pt files (real data)."""
        pt_dirs = [
            Path("/dev/shm/jvs_precomputed_aligned/train"),
            Path("/dev/shm/jvs_precomputed_aligned/val"),
            Path("/dev/shm/jvs_precomputed/train"),
            Path("/dev/shm/jvs_precomputed/val"),
        ]
        wavs_base = Path("/data/Matcha-TTS/data/jvs/wavs")

        checked = 0
        for pt_dir in pt_dirs:
            if not pt_dir.exists():
                continue
            pt_files = sorted(pt_dir.glob("*.pt"))[:5]  # Check first 5 per dir
            for pt_path in pt_files:
                # Parse name: e.g. jvs001_BASIC5000_0025.pt
                stem = pt_path.stem
                parts = stem.split("_", 1)
                spk_name = parts[0]
                utt_id = parts[1]
                wav_path = wavs_base / spk_name / f"{utt_id}.wav"
                if not wav_path.exists():
                    continue

                info = sf.info(str(wav_path))
                mel_from_formula = info.frames // HOP_LENGTH
                pt_data = torch.load(str(pt_path), weights_only=True)
                mel_from_pt = pt_data["mel"].shape[-1]
                assert mel_from_formula == mel_from_pt, f"{stem}: sf.info={mel_from_formula}, pt={mel_from_pt}"
                checked += 1

        if checked == 0:
            pytest.skip("No real data samples available on this machine")

    @pytest.mark.slow
    def test_mel_frames_matches_mel_spectrogram_computation(self):
        """sf.info // HOP matches actual mel_spectrogram output on synthetic audio."""
        from matcha.utils.audio import mel_spectrogram

        total_samples = SAMPLE_RATE * 3
        audio_np = np.random.randn(total_samples).astype(np.float32) * 0.1
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio_np, SAMPLE_RATE)
            info = sf.info(f.name)
            mel_from_formula = info.frames // HOP_LENGTH

        audio_tensor = torch.from_numpy(audio_np).unsqueeze(0)
        mel = mel_spectrogram(audio_tensor, 1024, 80, SAMPLE_RATE, HOP_LENGTH, 1024, 0.0, 8000, center=False).squeeze()
        mel_from_computation = mel.shape[-1]
        assert mel_from_formula == mel_from_computation


# ===========================================================================
# T1-3: Text cache
# ===========================================================================


class TestTextCache:
    """Test text pre-computation cache (hiragana conversion, pickle, dedup)."""

    @pytest.fixture(autouse=True)
    def _skip_without_pyopenjtalk(self):
        pytest.importorskip("pyopenjtalk")

    def test_hiragana_roundtrip(self):
        """Cached hiragana matches direct computation -- result is consistent."""
        from run_full_alignment_pipeline import text_to_hiragana

        texts = ["こんにちは世界", "音声合成のテスト", "今日はいい天気です"]
        for text in texts:
            result = text_to_hiragana(text)
            assert isinstance(result, str)
            assert len(result) > 0
            # Verify no katakana remains (U+30A1..U+30F6)
            for ch in result:
                assert not (0x30A1 <= ord(ch) <= 0x30F6), (
                    f"Katakana found in output: '{ch}' (U+{ord(ch):04X}) for input '{text}'"
                )

    def test_hiragana_deterministic(self):
        """Calling text_to_hiragana twice on the same input gives the same result."""
        from run_full_alignment_pipeline import text_to_hiragana

        text = "東京は日本の首都です"
        result1 = text_to_hiragana(text)
        result2 = text_to_hiragana(text)
        assert result1 == result2

    def test_katakana_to_hiragana_conversion(self):
        """katakana_to_hiragana correctly converts all katakana."""
        from run_full_alignment_pipeline import katakana_to_hiragana

        # Full katakana string
        assert katakana_to_hiragana("テスト") == "てすと"
        assert katakana_to_hiragana("コンニチワ") == "こんにちわ"
        # Mixed: only katakana is converted
        assert katakana_to_hiragana("ひらがなカタカナ") == "ひらがなかたかな"
        # ASCII is untouched
        assert katakana_to_hiragana("abc") == "abc"
        # Empty string
        assert katakana_to_hiragana("") == ""

    def test_cache_pickle_roundtrip(self):
        """Cache dict can be saved and loaded via pickle."""
        cache = {
            "テスト": {
                "hiragana": "てすと",
                "text_norm": [0, 1, 0, 2, 0],
                "cleaned_text": "t e s u t o",
            },
            "こんにちは": {
                "hiragana": "こんにちわ",
                "text_norm": [0, 3, 0, 4, 0],
                "cleaned_text": "k o N n i ch i w a",
            },
        }
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump(cache, f)
            f.flush()
            with open(f.name, "rb") as rf:
                loaded = pickle.load(rf)
        assert loaded == cache
        assert loaded["テスト"]["hiragana"] == "てすと"
        assert loaded["こんにちは"]["text_norm"] == [0, 3, 0, 4, 0]

    def test_unique_text_dedup(self):
        """Deduplication of filelist entries by text works correctly."""
        entries = [
            ("path/jvs001/UTT001.wav", "0", "同じテキスト"),
            ("path/jvs002/UTT001.wav", "1", "同じテキスト"),
            ("path/jvs003/UTT001.wav", "2", "異なるテキスト"),
            ("path/jvs004/UTT001.wav", "3", "第三のテキスト"),
            ("path/jvs005/UTT001.wav", "4", "異なるテキスト"),
        ]
        unique_texts = set(text for _, _, text in entries)
        assert len(unique_texts) == 3
        assert "同じテキスト" in unique_texts
        assert "異なるテキスト" in unique_texts
        assert "第三のテキスト" in unique_texts

    def test_text_to_hiragana_removes_punctuation(self):
        """Punctuation is stripped from the hiragana output."""
        from run_full_alignment_pipeline import text_to_hiragana

        # Input with punctuation
        result = text_to_hiragana("はい、そうです。")
        # Result should not contain punctuation characters
        punct_chars = set("。、！？!?,.-「」『』（）()【】[]{}・…―─")
        for ch in result:
            assert ch not in punct_chars, f"Punctuation '{ch}' found in output: '{result}'"


# ===========================================================================
# T2-1: Dual resample (prepare_jvs.py)
# ===========================================================================


class TestDualResample:
    """Test dual-rate resampling output (22kHz + 16kHz Julius)."""

    def test_dual_output_exists(self):
        """Both 22kHz and 16kHz files are produced by resample_audio."""
        from prepare_jvs import resample_audio

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create 24kHz input (typical JVS source rate)
            input_path = Path(tmpdir) / "input.wav"
            samples = np.random.randn(24000 * 2).astype(np.float32) * 0.1
            sf.write(str(input_path), samples, 24000)

            output_22k = Path(tmpdir) / "output_22k.wav"
            output_16k = Path(tmpdir) / "output_16k.wav"

            resample_audio(
                str(input_path),
                str(output_22k),
                orig_sr=24000,
                target_sr=22050,
                do_trim=False,
                julius_output_path=str(output_16k),
            )

            assert output_22k.exists(), "22kHz output not created"
            assert output_16k.exists(), "16kHz output not created"

    def test_sample_rates(self):
        """Output sample rates are 22050 and 16000 respectively."""
        from prepare_jvs import resample_audio

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.wav"
            samples = np.random.randn(24000 * 2).astype(np.float32) * 0.1
            sf.write(str(input_path), samples, 24000)

            output_22k = Path(tmpdir) / "output_22k.wav"
            output_16k = Path(tmpdir) / "output_16k.wav"

            resample_audio(
                str(input_path),
                str(output_22k),
                orig_sr=24000,
                target_sr=22050,
                do_trim=False,
                julius_output_path=str(output_16k),
            )

            info_22k = sf.info(str(output_22k))
            info_16k = sf.info(str(output_16k))
            assert info_22k.samplerate == 22050
            assert info_16k.samplerate == 16000

    def test_julius_output_pcm16(self):
        """16kHz output is 16-bit PCM (Julius requirement)."""
        from prepare_jvs import resample_audio

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.wav"
            samples = np.random.randn(24000 * 2).astype(np.float32) * 0.1
            sf.write(str(input_path), samples, 24000)

            output_22k = Path(tmpdir) / "output_22k.wav"
            output_16k = Path(tmpdir) / "output_16k.wav"

            resample_audio(
                str(input_path),
                str(output_22k),
                orig_sr=24000,
                target_sr=22050,
                do_trim=False,
                julius_output_path=str(output_16k),
            )

            info_16k = sf.info(str(output_16k))
            assert info_16k.subtype == "PCM_16", f"Expected PCM_16, got {info_16k.subtype}"

    def test_no_julius_output_when_none(self):
        """When julius_output_path is None, only 22kHz is produced."""
        from prepare_jvs import resample_audio

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.wav"
            samples = np.random.randn(24000 * 2).astype(np.float32) * 0.1
            sf.write(str(input_path), samples, 24000)

            output_22k = Path(tmpdir) / "output_22k.wav"
            output_16k = Path(tmpdir) / "output_16k.wav"

            resample_audio(
                str(input_path),
                str(output_22k),
                orig_sr=24000,
                target_sr=22050,
                do_trim=False,
                julius_output_path=None,
            )

            assert output_22k.exists()
            assert not output_16k.exists()

    def test_dual_output_with_trim(self):
        """Both outputs are produced even when do_trim=True."""
        from prepare_jvs import resample_audio

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create audio with silence padding
            input_path = Path(tmpdir) / "input.wav"
            silence = np.zeros(24000)  # 1s silence
            speech = np.random.randn(24000 * 2).astype(np.float32) * 0.3
            audio = np.concatenate([silence, speech, silence]).astype(np.float32)
            sf.write(str(input_path), audio, 24000)

            output_22k = Path(tmpdir) / "output_22k.wav"
            output_16k = Path(tmpdir) / "output_16k.wav"

            resample_audio(
                str(input_path),
                str(output_22k),
                orig_sr=24000,
                target_sr=22050,
                do_trim=True,
                julius_output_path=str(output_16k),
            )

            assert output_22k.exists()
            assert output_16k.exists()
            # Trimmed outputs should be shorter than untrimmed
            info_22k = sf.info(str(output_22k))
            # Original duration ~4s, trimmed should be shorter
            assert info_22k.frames < int(4 * 22050)

    def test_resample_worker_dual_output(self):
        """_resample_worker handles 6-tuple for dual output."""
        from prepare_jvs import _resample_worker

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.wav"
            samples = np.random.randn(24000).astype(np.float32) * 0.1
            sf.write(str(input_path), samples, 24000)

            output_22k = Path(tmpdir) / "output_22k.wav"
            output_16k = Path(tmpdir) / "output_16k.wav"

            # 6-tuple format: (src, dst, target_sr, do_trim, julius_wav, julius_sr)
            args_tuple = (
                str(input_path),
                str(output_22k),
                22050,
                False,
                str(output_16k),
                16000,
            )
            src_path, error = _resample_worker(args_tuple)
            assert error is None, f"Worker error: {error}"
            assert output_22k.exists()
            assert output_16k.exists()

    def test_resample_worker_4_tuple(self):
        """_resample_worker handles 4-tuple (no Julius output, backward compat)."""
        from prepare_jvs import _resample_worker

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.wav"
            samples = np.random.randn(24000).astype(np.float32) * 0.1
            sf.write(str(input_path), samples, 24000)

            output_22k = Path(tmpdir) / "output_22k.wav"

            # 4-tuple format: (src, dst, target_sr, do_trim)
            args_tuple = (str(input_path), str(output_22k), 22050, False)
            src_path, error = _resample_worker(args_tuple)
            assert error is None, f"Worker error: {error}"
            assert output_22k.exists()


# ===========================================================================
# T2-2: precompute_with_alignment.py unified .pt generation
# ===========================================================================


class TestPrecomputeWithAlignment:
    """Test unified .pt generation from precompute_with_alignment.py."""

    @pytest.fixture(autouse=True)
    def _skip_without_pyopenjtalk(self):
        pytest.importorskip("pyopenjtalk")

    @pytest.mark.slow
    def test_pt_contains_all_keys_real_data(self):
        """Unified .pt from real wav + real .lab contains all required keys."""
        from precompute_with_alignment import process_sample_with_alignment

        wav_path = Path("/data/Matcha-TTS/data/jvs/wavs/jvs001/BASIC5000_0025.wav")
        lab_path = Path("/data/Matcha-TTS/data/julius_work/wav/jvs001_BASIC5000_0025.lab")
        if not wav_path.exists() or not lab_path.exists():
            pytest.skip("Real wav/lab data not available")

        # Find text from filelist
        text = None
        for fl in ["train.txt", "val.txt"]:
            fl_path = Path("/data/Matcha-TTS/data/jvs") / fl
            if not fl_path.exists():
                continue
            with open(fl_path, encoding="utf-8") as f:
                for line in f:
                    if "jvs001/BASIC5000_0025" in line:
                        text = line.strip().split("|")[2]
                        break
            if text:
                break
        if not text:
            pytest.skip("Cannot find text for jvs001_BASIC5000_0025 in filelist")

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "jvs001_BASIC5000_0025.pt"
            result_path, skipped, msg = process_sample_with_alignment(
                str(wav_path),
                0,
                text,
                str(lab_path),
                str(out_path),
                mel_mean=-6.550095,
                mel_std=2.383771,
                align_mode="auto",
            )
            assert not skipped, f"Unexpected skip: {msg}"

            data = torch.load(str(out_path), weights_only=True)
            assert "mel" in data, "Missing 'mel' key"
            assert "text" in data, "Missing 'text' key"
            assert "spk" in data, "Missing 'spk' key"
            assert "cleaned_text" in data, "Missing 'cleaned_text' key"
            assert "durations" in data, "Missing 'durations' key"

            # Consistency checks
            assert data["mel"].dim() == 2 and data["mel"].shape[0] == 80
            assert len(data["text"]) == len(data["durations"])
            assert data["durations"].sum().item() == data["mel"].shape[-1]
            assert data["durations"].dtype == torch.int64

    def test_missing_lab_skips(self):
        """process_sample_with_alignment skips when .lab file is missing."""
        from precompute_with_alignment import process_sample_with_alignment

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            wav_dir = tmpdir / "wavs" / "jvs001"
            wav_dir.mkdir(parents=True)
            wav_path = wav_dir / "UTT001.wav"
            sf.write(
                str(wav_path),
                np.random.randn(22050).astype(np.float32) * 0.1,
                SAMPLE_RATE,
            )

            out_path = tmpdir / "output.pt"
            lab_path = tmpdir / "nonexistent.lab"
            _, skipped, msg = process_sample_with_alignment(
                str(wav_path),
                0,
                "テスト",
                str(lab_path),
                str(out_path),
                -6.55,
                2.38,
            )
            assert skipped is True
            assert "no .lab file" in msg

    def test_parse_filelist(self):
        """parse_filelist correctly parses pipe-delimited entries."""
        from precompute_with_alignment import parse_filelist

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("path/a.wav|0|テスト\n")
            f.write("path/b.wav|1|こんにちは\n")
            f.write("\n")  # empty line
            f.write("path/c.wav|2|音声合成\n")
            f.flush()
            entries = parse_filelist(f.name)

        assert len(entries) == 3
        assert entries[0] == ["path/a.wav", "0", "テスト"]
        assert entries[2] == ["path/c.wav", "2", "音声合成"]

    @pytest.mark.slow
    def test_real_aligned_pt_has_all_keys(self):
        """Real .pt files in jvs_precomputed_aligned contain all required keys."""
        pt_dir = Path("/dev/shm/jvs_precomputed_aligned/train")
        if not pt_dir.exists():
            pytest.skip("Aligned precomputed data not available")

        pt_files = sorted(pt_dir.glob("*.pt"))[:10]
        assert len(pt_files) > 0, "No .pt files found"

        for pt_path in pt_files:
            data = torch.load(str(pt_path), weights_only=True)
            assert "mel" in data, f"Missing 'mel' in {pt_path.name}"
            assert "text" in data, f"Missing 'text' in {pt_path.name}"
            assert "spk" in data, f"Missing 'spk' in {pt_path.name}"
            assert "cleaned_text" in data, f"Missing 'cleaned_text' in {pt_path.name}"
            assert "durations" in data, f"Missing 'durations' in {pt_path.name}"

            # Validate shapes are consistent
            text_len = data["text"].shape[0]
            dur_len = data["durations"].shape[0]
            assert text_len == dur_len, f"{pt_path.name}: text length {text_len} != duration length {dur_len}"
            # Duration sum should equal mel frames
            mel_frames = data["mel"].shape[-1]
            dur_sum = data["durations"].sum().item()
            assert dur_sum == mel_frames, f"{pt_path.name}: duration sum {dur_sum} != mel frames {mel_frames}"

    @pytest.mark.slow
    def test_aligned_and_nonaligned_mel_match(self):
        """Mel spectrograms should be identical between aligned and non-aligned .pt files."""
        aligned_dir = Path("/dev/shm/jvs_precomputed_aligned/train")
        nonaligned_dir = Path("/dev/shm/jvs_precomputed/train")
        if not aligned_dir.exists() or not nonaligned_dir.exists():
            pytest.skip("Precomputed data not available")

        aligned_files = sorted(aligned_dir.glob("*.pt"))[:5]
        checked = 0
        for aligned_pt in aligned_files:
            nonaligned_pt = nonaligned_dir / aligned_pt.name
            if not nonaligned_pt.exists():
                continue
            data_a = torch.load(str(aligned_pt), weights_only=True)
            data_na = torch.load(str(nonaligned_pt), weights_only=True)

            # Mel should be identical (same wav, same normalization)
            assert torch.allclose(data_a["mel"], data_na["mel"], atol=1e-6), f"{aligned_pt.name}: mel mismatch"
            # Text should be identical
            assert torch.equal(data_a["text"], data_na["text"]), f"{aligned_pt.name}: text mismatch"
            # Speaker ID should be identical
            assert data_a["spk"] == data_na["spk"], f"{aligned_pt.name}: spk mismatch"
            # Non-aligned should NOT have durations
            assert "durations" not in data_na, f"{nonaligned_pt.name}: unexpectedly has 'durations' key"
            # Aligned MUST have durations
            assert "durations" in data_a, f"{aligned_pt.name}: missing 'durations' key"
            checked += 1

        assert checked > 0, "No matching file pairs found"

    def test_compute_duration_from_lab_rejects_empty(self):
        """compute_duration_from_lab returns None for empty .lab files."""
        from precompute_with_alignment import compute_duration_from_lab

        with tempfile.NamedTemporaryFile(mode="w", suffix=".lab", delete=False, encoding="utf-8") as f:
            f.write("")
            f.flush()
            result, msg = compute_duration_from_lab(f.name, "テスト", 100)

        assert result is None
        assert "Empty" in msg


# ===========================================================================
# Syntax / Import / CLI validation
# ===========================================================================


class TestScriptSyntax:
    """All Python pipeline scripts must parse without syntax errors."""

    @pytest.mark.parametrize(
        "script",
        [
            "precompute_with_alignment.py",
            "prepare_jvs.py",
            "run_full_alignment_pipeline.py",
            "precompute_dataset.py",
        ],
    )
    def test_script_parses(self, script):
        path = SCRIPTS_DIR / script
        assert path.exists(), f"{script} not found at {path}"
        source = path.read_text(encoding="utf-8")
        ast.parse(source, filename=str(path))


class TestScriptImports:
    """Verify that cross-script imports resolve correctly."""

    def test_precompute_with_alignment_imports(self):
        """precompute_with_alignment.py imports from convert_julius_to_durations."""
        from convert_julius_to_durations import (
            align_julius_with_pyopenjtalk,
            build_duration_array_with_blanks,
            parse_lab_file,
            time_to_frames,
        )

        assert callable(align_julius_with_pyopenjtalk)
        assert callable(build_duration_array_with_blanks)
        assert callable(parse_lab_file)
        assert callable(time_to_frames)

    def test_matcha_text_imports(self):
        """Core matcha imports used by precompute_with_alignment.py."""
        from matcha.text import text_to_sequence
        from matcha.text.julius_to_pyopenjtalk import map_julius_sequence
        from matcha.utils.audio import mel_spectrogram
        from matcha.utils.model import normalize
        from matcha.utils.utils import intersperse

        assert callable(text_to_sequence)
        assert callable(map_julius_sequence)
        assert callable(mel_spectrogram)
        assert callable(normalize)
        assert callable(intersperse)


class TestCLIInterfaces:
    """Verify CLI flags for pipeline scripts."""

    def test_precompute_with_alignment_help(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / "precompute_with_alignment.py"), "--help"],
            capture_output=True,
            encoding="utf-8",
            timeout=30,
            check=False,
            # The help text contains non-ASCII chars; force UTF-8 on the child's
            # stdout (and decode it as UTF-8 here) so the test does not crash
            # with cp932 encode/decode errors on Windows.
            env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
        )
        assert result.returncode == 0
        assert "--filelist" in result.stdout
        assert "--lab-dir" in result.stdout
        assert "--output-dir" in result.stdout
        assert "--mel-mean" in result.stdout
        assert "--mel-std" in result.stdout
        assert "--align-mode" in result.stdout

    def test_prepare_jvs_has_julius_flag(self):
        """prepare_jvs.py has --julius-output-dir flag for dual resample."""
        result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / "prepare_jvs.py"), "--help"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0
        assert "--julius-output-dir" in result.stdout


class TestPrepareJvsBackwardCompat:
    """Verify prepare_jvs.py changes are backward compatible."""

    def test_resample_audio_default_args(self):
        """resample_audio() new params have defaults so old callers work."""
        from prepare_jvs import resample_audio

        sig = inspect.signature(resample_audio)
        assert sig.parameters["julius_output_path"].default is None
        assert sig.parameters["julius_sr"].default == 16000

    def test_resample_worker_handles_both_tuple_sizes(self):
        """_resample_worker() accepts both 4-tuple (old) and 6-tuple (new)."""
        from prepare_jvs import _resample_worker

        source = inspect.getsource(_resample_worker)
        assert "len(args_tuple) == 6" in source
        assert "julius_wav, julius_sr = None, None" in source


# ===========================================================================
# Helpers for fast-path (Tier 3) tests: synthetic wav / .lab / text-cache
# ===========================================================================

FRAME_SEC = HOP_LENGTH / SAMPLE_RATE

# pyopenjtalk tokens with no Julius segment (prosody-only, duration=0)
_PROSODY_ONLY = {"#", "[", "]", "?"}

# pyopenjtalk token -> raw Julius phoneme as it appears in a .lab file
_PYOPENJTALK_TO_JULIUS_RAW = {
    "^": "silB",
    "$": "silE",
    "_": "pau",
    "A": "a",
    "I": "i",
    "U": "u",
    "E": "e",
    "O": "o",
}

# Canned japanese_cleaners outputs so the fast path can be exercised without
# pyopenjtalk. Keys are sentinel texts (never real Japanese, so they cannot
# collide with real-cleaner results in the text_to_sequence LRU cache).
_FAKE_JA_LEXICON = {
    "FAKE_TEXT_ALPHA": "^ k o N n i ch i w a $",
    "FAKE_TEXT_BETA": "^ a r i g a t o _ o $",
    "FAKE_TEXT_GAMMA": "^ t e s U t o $",
}


def _install_fake_japanese_cleaner(monkeypatch):
    """Replace japanese_cleaners with a canned lookup (no pyopenjtalk required)."""
    from matcha.text import cleaners as cleaners_mod

    monkeypatch.setattr(cleaners_mod, "japanese_cleaners", lambda text: _FAKE_JA_LEXICON[text])


def _write_sine_wav(path, seconds, freq=440.0, amplitude=0.3, stereo=False):
    """Write a deterministic sine wav at 22050 Hz; return the sample count."""
    n = int(SAMPLE_RATE * seconds)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    audio = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    if stereo:
        audio = np.stack([audio, 0.5 * audio], axis=1)
    sf.write(str(path), audio, SAMPLE_RATE)
    return n


def _expected_mel_frames(n_samples):
    """center=False mel frame count: 1 + (L - hop) // hop."""
    return 1 + (n_samples - HOP_LENGTH) // HOP_LENGTH


def _julius_phonemes_for_cleaned(cleaned_text):
    """Raw Julius phoneme stream matching a pyopenjtalk cleaned string."""
    return [_PYOPENJTALK_TO_JULIUS_RAW.get(tok, tok) for tok in cleaned_text.split() if tok not in _PROSODY_ONLY]


def _write_lab_for_cleaned(lab_path, cleaned_text, total_frames):
    """Write a float-seconds .lab whose segments cover exactly total_frames."""
    phones = _julius_phonemes_for_cleaned(cleaned_text)
    base = total_frames // len(phones)
    assert base >= 1, "synthetic audio too short for this phoneme count"
    durations = [base] * len(phones)
    durations[-1] += total_frames - base * len(phones)
    lines = []
    cum = 0
    for ph, dur in zip(phones, durations):
        lines.append(f"{cum * FRAME_SEC:.7f} {(cum + dur) * FRAME_SEC:.7f} {ph}")
        cum += dur
    Path(lab_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_args(**overrides):
    """argparse.Namespace stand-in with the fast-path defaults run_fast_pipeline reads."""
    defaults = {
        "mel_mean": -6.550095,
        "mel_std": 2.383771,
        "align_mode": "auto",
        "io_workers": 2,
        "batch_size": 2,
    }
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


# ===========================================================================
# T3: build_tasks naming contract
# ===========================================================================


class TestBuildTasks:
    """build_tasks derives names/paths from the filelist entries."""

    def test_naming_contract(self, tmp_path):
        """name == '{wav parent dir}_{stem}' and lab/out paths follow it."""
        from precompute_with_alignment import build_tasks

        lab_dir = tmp_path / "labs"
        out_dir = tmp_path / "out"
        entries = [
            ["C:/corpus/wavs/jvs042/BASIC5000_0123.wav", "41", "text a"],
            ["rel/path/jvs007/VOICE_9.wav", "6", "text b"],
        ]
        tasks = build_tasks(entries, lab_dir, out_dir)

        assert [t.name for t in tasks] == ["jvs042_BASIC5000_0123", "jvs007_VOICE_9"]
        assert tasks[0].lab_path == str(lab_dir / "jvs042_BASIC5000_0123.lab")
        assert tasks[0].out_path == str(out_dir / "jvs042_BASIC5000_0123.pt")
        assert tasks[1].lab_path == str(lab_dir / "jvs007_VOICE_9.lab")
        assert tasks[1].out_path == str(out_dir / "jvs007_VOICE_9.pt")

    def test_speaker_and_text_passthrough(self, tmp_path):
        """Speaker ids are parsed to int; wav path and text are kept verbatim."""
        from precompute_with_alignment import build_tasks

        entries = [["data/wavs/jvs005/UTT001.wav", "4", "こんにちは"]]
        task = build_tasks(entries, tmp_path / "labs", tmp_path / "out")[0]

        assert task.spk == 4
        assert isinstance(task.spk, int)
        assert task.wav_path == "data/wavs/jvs005/UTT001.wav"
        assert task.text == "こんにちは"


# ===========================================================================
# T3-1: build_text_sequence_cache
# ===========================================================================


class TestBuildTextSequenceCache:
    """Shared text cache: dedup, sequential-path selection, value correctness."""

    def test_duplicate_texts_computed_once_order_preserved(self, monkeypatch):
        """Duplicated inputs are computed once, in first-seen order."""
        import precompute_with_alignment as pwa

        calls = []

        def fake_worker(text):
            calls.append(text)
            return text, [1, 2, 3], f"cleaned {text}"

        monkeypatch.setattr(pwa, "_text_cache_worker", fake_worker)
        cache = pwa.build_text_sequence_cache(["b", "a", "b", "c", "a"], num_workers=1)

        assert calls == ["b", "a", "c"]
        assert cache == {t: ([1, 2, 3], f"cleaned {t}") for t in ["b", "a", "c"]}

    def test_sequential_path_selection(self, monkeypatch):
        """<=100 unique texts (or num_workers<=1) run in-process, no ProcessPool.

        The fake worker is a local closure that cannot be pickled to a spawned
        worker process, so any recorded call proves the sequential path ran.
        """
        import precompute_with_alignment as pwa

        calls = []

        def fake_worker(text):
            calls.append(text)
            return text, [0], f"c {text}"

        monkeypatch.setattr(pwa, "_text_cache_worker", fake_worker)

        # Small unique set with many workers -> sequential path
        pwa.build_text_sequence_cache(["x", "y", "z"], num_workers=8)
        assert calls == ["x", "y", "z"]

        # >100 unique texts with num_workers=1 -> still sequential
        calls.clear()
        many = [f"t{i}" for i in range(150)]
        cache = pwa.build_text_sequence_cache(many, num_workers=1)
        assert calls == many
        assert len(cache) == 150

    def test_cache_values_match_direct_text_to_sequence(self):
        """Cache stores exactly the (seq, cleaned) pairs text_to_sequence returns."""
        pytest.importorskip("pyopenjtalk")
        from precompute_with_alignment import build_text_sequence_cache

        from matcha.text import text_to_sequence

        texts = ["こんにちは", "音声合成のテスト", "こんにちは"]
        cache = build_text_sequence_cache(texts, num_workers=1)

        assert set(cache) == {"こんにちは", "音声合成のテスト"}
        for text in set(texts):
            seq, cleaned = text_to_sequence(text, ["japanese_cleaners"], language="ja")
            assert cache[text] == (seq, cleaned)


# ===========================================================================
# T3-2: _load_one error / normalization paths
# ===========================================================================


class TestLoadOne:
    """_load_one: wav + .lab + text-cache lookup with per-sample error handling."""

    _LAB_BODY = "0.0000000 0.1000000 silB\n0.1000000 0.2000000 a\n0.2000000 0.3000000 silE\n"
    _CACHE = {"hello": ([1, 2, 3], "^ a $")}

    def _make_task(self, tmp_path, text="hello"):
        from precompute_with_alignment import build_tasks

        wav_dir = tmp_path / "wavs" / "jvs001"
        wav_dir.mkdir(parents=True, exist_ok=True)
        lab_dir = tmp_path / "labs"
        lab_dir.mkdir(exist_ok=True)
        out_dir = tmp_path / "out"
        out_dir.mkdir(exist_ok=True)
        wav_path = wav_dir / "UTT001.wav"
        return build_tasks([[str(wav_path), "0", text]], lab_dir, out_dir)[0]

    def test_missing_lab_is_error(self, tmp_path):
        from precompute_with_alignment import _load_one

        task = self._make_task(tmp_path)
        _write_sine_wav(task.wav_path, 0.5)
        sample = _load_one(task, self._CACHE)

        assert sample.error == "no .lab file"
        assert sample.audio is None

    def test_empty_lab_is_error(self, tmp_path):
        from precompute_with_alignment import _load_one

        task = self._make_task(tmp_path)
        _write_sine_wav(task.wav_path, 0.5)
        Path(task.lab_path).write_text("", encoding="utf-8")
        sample = _load_one(task, self._CACHE)

        assert sample.error is not None
        assert sample.error.startswith("Empty .lab file")

    def test_text_absent_from_cache_is_error(self, tmp_path):
        from precompute_with_alignment import _load_one

        task = self._make_task(tmp_path, text="not-cached")
        _write_sine_wav(task.wav_path, 0.5)
        Path(task.lab_path).write_text(self._LAB_BODY, encoding="utf-8")
        sample = _load_one(task, self._CACHE)

        assert sample.error == "text not in cache"

    def test_wrong_sample_rate_is_load_error(self, tmp_path):
        from precompute_with_alignment import _load_one

        task = self._make_task(tmp_path)
        n = 16000
        t = np.arange(n, dtype=np.float32) / 16000
        sf.write(str(task.wav_path), (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), 16000)
        Path(task.lab_path).write_text(self._LAB_BODY, encoding="utf-8")
        sample = _load_one(task, self._CACHE)

        assert sample.error is not None
        assert sample.error.startswith("load error")
        assert "Expected 22050 Hz, got 16000" in sample.error

    def test_stereo_wav_is_downmixed_to_mono_float32(self, tmp_path):
        from precompute_with_alignment import _load_one

        task = self._make_task(tmp_path)
        n = _write_sine_wav(task.wav_path, 0.5, stereo=True)
        Path(task.lab_path).write_text(self._LAB_BODY, encoding="utf-8")
        sample = _load_one(task, self._CACHE)

        assert sample.error is None
        assert sample.audio.ndim == 1
        assert sample.audio.dtype == np.float32
        assert len(sample.audio) == n

    def test_success_carries_cached_seq_and_lab_segments(self, tmp_path):
        from precompute_with_alignment import _load_one

        task = self._make_task(tmp_path)
        n = _write_sine_wav(task.wav_path, 0.5)
        Path(task.lab_path).write_text(self._LAB_BODY, encoding="utf-8")
        sample = _load_one(task, self._CACHE)

        assert sample.error is None
        assert sample.text_seq == [1, 2, 3]
        assert sample.cleaned_text == "^ a $"
        assert len(sample.lab_segments) == 3
        assert sample.lab_segments[1][2] == "a"
        assert len(sample.audio) == n


# ===========================================================================
# T3-3: _mel_batch_gpu on CPU device (slicing math / zero-pad-leak guard)
# ===========================================================================


class TestMelBatchCpuParity:
    """_mel_batch_gpu(device=cpu) must match _mel_single_cpu frame-for-frame."""

    MEL_MEAN = -6.550095
    MEL_STD = 2.383771

    def _sine(self, seconds, freq=440.0):
        n = int(SAMPLE_RATE * seconds)
        t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
        return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)

    def test_frame_counts_and_parity_with_single_path(self):
        """Each batched mel has 1 + (L-256)//256 frames and matches the single path."""
        from precompute_with_alignment import _mel_batch_gpu, _mel_single_cpu

        audios = [self._sine(0.5), self._sine(1.0, freq=523.25), self._sine(1.7, freq=349.23)]
        mels = _mel_batch_gpu(audios, torch.device("cpu"), self.MEL_MEAN, self.MEL_STD)

        assert len(mels) == 3
        for audio, mel in zip(audios, mels):
            expected_frames = 1 + (len(audio) - HOP_LENGTH) // HOP_LENGTH
            assert mel.shape == (80, expected_frames)
            single = _mel_single_cpu(audio, self.MEL_MEAN, self.MEL_STD)
            assert torch.allclose(mel, single, atol=1e-4)

    def test_shortest_sample_final_frames_no_zero_pad_leak(self):
        """The zero tail padding the shortest row must not leak into its last frames."""
        from precompute_with_alignment import _mel_batch_gpu, _mel_single_cpu

        audios = [self._sine(0.5), self._sine(1.7)]  # shortest first, heavy zero pad
        mels = _mel_batch_gpu(audios, torch.device("cpu"), self.MEL_MEAN, self.MEL_STD)
        single = _mel_single_cpu(audios[0], self.MEL_MEAN, self.MEL_STD)

        assert mels[0].shape == single.shape
        assert torch.allclose(mels[0][:, -3:], single[:, -3:], atol=1e-4)

    def test_batch_of_one_matches_single_path(self):
        from precompute_with_alignment import _mel_batch_gpu, _mel_single_cpu

        audio = self._sine(0.8, freq=392.0)
        (mel,) = _mel_batch_gpu([audio], torch.device("cpu"), self.MEL_MEAN, self.MEL_STD)
        single = _mel_single_cpu(audio, self.MEL_MEAN, self.MEL_STD)

        assert mel.shape == single.shape
        assert torch.allclose(mel, single, atol=1e-4)


# ===========================================================================
# T3-4: _finalize_one validation invariants
# ===========================================================================


class TestFinalizeOne:
    """_finalize_one: duration validation and .pt writing from cached inputs."""

    def _make_sample(self, tmp_path, *, cleaned_text, text_seq, julius, name="jvs001_X"):
        """Build a LoadedSample with .lab segments given as (phoneme, n_frames)."""
        from precompute_with_alignment import LoadedSample, Task

        segments = []
        cum = 0
        for ph, frames in julius:
            segments.append((cum * FRAME_SEC, (cum + frames) * FRAME_SEC, ph))
            cum += frames
        task = Task(
            wav_path="unused.wav",
            spk=7,
            text="raw text",
            lab_path="unused.lab",
            out_path=str(tmp_path / f"{name}.pt"),
            name=name,
        )
        return LoadedSample(
            task=task,
            audio=None,
            text_seq=text_seq,
            cleaned_text=cleaned_text,
            lab_segments=segments,
        )

    def test_success_saves_five_keys_and_uses_cached_seq_verbatim(self, tmp_path):
        """Saved text equals the cached seq (deliberately wrong ids): no re-phonemization."""
        from precompute_with_alignment import _finalize_one

        # [7, 9] is NOT what "a i" would phonemize to -- if _finalize_one re-ran
        # text_to_sequence the saved tensor would differ.
        sample = self._make_sample(tmp_path, cleaned_text="a i", text_seq=[7, 9], julius=[("a", 3), ("i", 4)])
        mel = torch.randn(80, 7)
        out_path, skipped, msg = _finalize_one(sample, mel, "auto")

        assert (skipped, msg) == (False, "ok")
        data = torch.load(out_path, weights_only=True)
        assert set(data.keys()) == {"mel", "text", "spk", "cleaned_text", "durations"}
        assert torch.equal(data["mel"], mel)
        assert data["spk"] == 7
        assert data["cleaned_text"] == "a i"
        assert data["durations"].dtype == torch.int64
        assert data["durations"].tolist() == [0, 3, 0, 4, 0]
        assert data["text"].dtype == torch.int32  # torch.IntTensor
        assert data["text"].tolist() == [0, 7, 0, 9, 0]

    def test_duration_text_length_mismatch_skips_without_pt(self, tmp_path):
        from precompute_with_alignment import _finalize_one

        # 3 cached ids vs 2 phonemes: interspersed text len 7 != duration len 5
        sample = self._make_sample(tmp_path, cleaned_text="a i", text_seq=[7, 9, 11], julius=[("a", 3), ("i", 4)])
        out_path, skipped, msg = _finalize_one(sample, torch.randn(80, 7), "auto")

        assert skipped is True
        assert "duration/text length mismatch" in msg
        assert not Path(out_path).exists()

    def test_duration_sum_mismatch_skips_without_pt(self, tmp_path):
        from precompute_with_alignment import _finalize_one

        # Julius says 20 frames but mel has only 5: the last-phoneme adjustment
        # clamps at 0 and cannot absorb the difference -> sum validation fails.
        sample = self._make_sample(tmp_path, cleaned_text="a i", text_seq=[7, 9], julius=[("a", 10), ("i", 10)])
        out_path, skipped, msg = _finalize_one(sample, torch.randn(80, 5), "auto")

        assert skipped is True
        assert "duration sum mismatch" in msg
        assert not Path(out_path).exists()


# ===========================================================================
# T3-5: run_fast_pipeline end-to-end on synthetic data
# ===========================================================================


class TestRunFastPipeline:
    """End-to-end fast pipeline on synthetic sine wavs + consistent .lab files."""

    def _build_corpus(self, tmp_path, texts, seconds):
        """Create wavs/labs/out dirs and tasks; return (tasks, {name: n_samples})."""
        from precompute_with_alignment import build_tasks

        lab_dir = tmp_path / "labs"
        out_dir = tmp_path / "out"
        lab_dir.mkdir()
        out_dir.mkdir()
        entries = []
        n_samples_by_name = {}
        for i, (text, dur) in enumerate(zip(texts, seconds)):
            wav_dir = tmp_path / "wavs" / f"jvs{i + 1:03d}"
            wav_dir.mkdir(parents=True)
            wav_path = wav_dir / f"UTT{i:03d}.wav"
            n = _write_sine_wav(wav_path, dur, freq=330.0 + 110.0 * i)
            entries.append([str(wav_path), str(i), text])
            n_samples_by_name[f"jvs{i + 1:03d}_UTT{i:03d}"] = n
        return build_tasks(entries, lab_dir, out_dir), n_samples_by_name

    def _assert_valid_pt(self, task, text_cache):
        from matcha.utils.utils import intersperse

        data = torch.load(task.out_path, weights_only=True)
        assert set(data.keys()) == {"mel", "text", "spk", "cleaned_text", "durations"}
        assert data["mel"].dim() == 2
        assert data["mel"].shape[0] == 80
        assert data["durations"].dtype == torch.int64
        assert int(data["durations"].sum()) == data["mel"].shape[-1]
        assert len(data["text"]) == len(data["durations"])
        assert data["spk"] == task.spk
        seq, cleaned = text_cache[task.text]
        assert data["cleaned_text"] == cleaned
        assert data["text"].tolist() == intersperse(seq, 0)

    def test_e2e_synthetic_fake_cleaner(self, tmp_path, monkeypatch):
        """Full run: text cache -> .lab -> run_fast_pipeline -> validated .pt files."""
        from precompute_with_alignment import build_text_sequence_cache, run_fast_pipeline

        _install_fake_japanese_cleaner(monkeypatch)
        texts = ["FAKE_TEXT_ALPHA", "FAKE_TEXT_BETA"]
        tasks, n_samples = self._build_corpus(tmp_path, texts, [0.7, 1.1])

        text_cache = build_text_sequence_cache(texts, num_workers=1)
        for task in tasks:
            _, cleaned = text_cache[task.text]
            _write_lab_for_cleaned(task.lab_path, cleaned, _expected_mel_frames(n_samples[task.name]))

        success, skip, errors = run_fast_pipeline(tasks, _make_args(), text_cache, torch.device("cpu"))

        assert (success, skip, errors) == (2, 0, [])
        for task in tasks:
            self._assert_valid_pt(task, text_cache)

    def test_missing_lab_counts_as_skip_and_writes_no_pt(self, tmp_path, monkeypatch):
        from precompute_with_alignment import build_text_sequence_cache, run_fast_pipeline

        _install_fake_japanese_cleaner(monkeypatch)
        texts = ["FAKE_TEXT_ALPHA", "FAKE_TEXT_BETA"]
        tasks, n_samples = self._build_corpus(tmp_path, texts, [0.7, 0.9])

        text_cache = build_text_sequence_cache(texts, num_workers=1)
        # Only the first task gets a .lab file; the second must be skipped.
        _, cleaned = text_cache[tasks[0].text]
        _write_lab_for_cleaned(tasks[0].lab_path, cleaned, _expected_mel_frames(n_samples[tasks[0].name]))

        success, skip, errors = run_fast_pipeline(tasks, _make_args(), text_cache, torch.device("cpu"))

        assert (success, skip, errors) == (1, 1, [])
        assert Path(tasks[0].out_path).exists()
        assert not Path(tasks[1].out_path).exists()

    def test_e2e_real_pyopenjtalk(self, tmp_path):
        """Same e2e with the real japanese_cleaners (skipped without pyopenjtalk)."""
        pytest.importorskip("pyopenjtalk")
        from precompute_with_alignment import build_text_sequence_cache, run_fast_pipeline

        texts = ["こんにちは", "音声合成のテスト", "今日はいい天気です"]
        tasks, n_samples = self._build_corpus(tmp_path, texts, [1.0, 1.2, 1.4])

        text_cache = build_text_sequence_cache(texts, num_workers=1)
        for task in tasks:
            _, cleaned = text_cache[task.text]
            _write_lab_for_cleaned(task.lab_path, cleaned, _expected_mel_frames(n_samples[task.name]))

        success, skip, errors = run_fast_pipeline(tasks, _make_args(), text_cache, torch.device("cpu"))

        assert (success, skip, errors) == (3, 0, [])
        for task in tasks:
            self._assert_valid_pt(task, text_cache)


# ===========================================================================
# T3-6: legacy (process_sample_with_alignment) vs fast path parity
# ===========================================================================


class TestLegacyFastParity:
    """The fast path claims binary compatibility with the legacy per-sample path."""

    def test_fast_pt_matches_legacy_single_call(self, tmp_path, monkeypatch):
        from precompute_with_alignment import build_tasks, process_sample_with_alignment, run_fast_pipeline

        from matcha.text import text_to_sequence

        _install_fake_japanese_cleaner(monkeypatch)
        text = "FAKE_TEXT_GAMMA"

        wav_dir = tmp_path / "wavs" / "jvs009"
        wav_dir.mkdir(parents=True)
        wav_path = wav_dir / "PARITY_1.wav"
        n = _write_sine_wav(wav_path, 0.9, freq=294.0)

        lab_dir = tmp_path / "labs"
        lab_dir.mkdir()
        fast_dir = tmp_path / "fast"
        fast_dir.mkdir()
        legacy_dir = tmp_path / "legacy"
        legacy_dir.mkdir()

        seq, cleaned = text_to_sequence(text, ["japanese_cleaners"], language="ja")
        lab_path = lab_dir / "jvs009_PARITY_1.lab"
        _write_lab_for_cleaned(lab_path, cleaned, _expected_mel_frames(n))

        # Legacy path: direct call to the exact function run_legacy_pipeline
        # submits to its ProcessPool (avoids spawn overhead on Windows).
        legacy_out = legacy_dir / "jvs009_PARITY_1.pt"
        _, skipped, msg = process_sample_with_alignment(
            str(wav_path),
            8,
            text,
            str(lab_path),
            str(legacy_out),
            mel_mean=-6.550095,
            mel_std=2.383771,
            align_mode="auto",
        )
        assert not skipped, f"legacy path skipped: {msg}"

        # Fast path on identical inputs
        tasks = build_tasks([[str(wav_path), "8", text]], lab_dir, fast_dir)
        success, skip, errors = run_fast_pipeline(tasks, _make_args(), {text: (seq, cleaned)}, torch.device("cpu"))
        assert (success, skip, errors) == (1, 0, [])

        legacy = torch.load(str(legacy_out), weights_only=True)
        fast = torch.load(tasks[0].out_path, weights_only=True)

        assert set(legacy.keys()) == set(fast.keys())
        assert torch.equal(legacy["mel"], fast["mel"])  # bit-exact
        assert torch.equal(legacy["text"], fast["text"])
        assert torch.equal(legacy["durations"], fast["durations"])
        assert legacy["spk"] == fast["spk"]
        assert legacy["cleaned_text"] == fast["cleaned_text"]


# ===========================================================================
# run_julius_alignment.py: run_segkit_batch (2026-04-15 layout regression)
# ===========================================================================


class TestRunSegkitBatch:
    """run_segkit_batch must lay out wav+txt side by side with bin/ and models/."""

    @pytest.fixture()
    def no_symlink(self, monkeypatch):
        """Emulate Path.symlink_to by copying (no symlink privilege on Windows)."""

        def _copy_instead(self, target, target_is_directory=False):
            target = Path(target)
            if target.is_dir():
                shutil.copytree(target, self)
            else:
                shutil.copy2(target, self)

        monkeypatch.setattr(Path, "symlink_to", _copy_instead)

    def _make_segkit(self, tmp_path):
        segkit = tmp_path / "segkit"
        (segkit / "bin").mkdir(parents=True)
        (segkit / "models").mkdir()
        (segkit / "segment_julius.pl").write_text("# perl stub\n", encoding="utf-8")
        (segkit / "bin" / "julius.bin").write_text("stub", encoding="utf-8")
        (segkit / "models" / "hmmdefs").write_text("stub", encoding="utf-8")
        return segkit

    def _make_inputs(self, tmp_path, names):
        src = tmp_path / "inputs"
        src.mkdir()
        wav_files, txt_files = [], []
        for name in names:
            wav = src / f"{name}.wav"
            wav.write_bytes(b"RIFF0000WAVE")
            txt = src / f"{name}.txt"
            txt.write_text("こんにちは", encoding="utf-8")
            wav_files.append((name, str(wav)))
            txt_files.append((name, str(txt)))
        return wav_files, txt_files

    def test_working_layout_and_lab_collection(self, tmp_path, no_symlink, monkeypatch):
        """wav/{name}.wav AND wav/{name}.txt in the SAME dir; bin/ + models/ at root."""
        import run_julius_alignment as rja

        names = ["jvs001_U1", "jvs002_U2"]
        segkit = self._make_segkit(tmp_path)
        wav_files, txt_files = self._make_inputs(tmp_path, names)
        out_dir = tmp_path / "out"

        layout = {}

        def fake_run(cmd, **kwargs):
            work = Path(kwargs["cwd"])
            wav_dir = work / "wav"
            layout["perl_invocation"] = cmd[0] == "perl" and cmd[1].endswith("segment_julius.pl")
            layout["bin_is_dir"] = (work / "bin").is_dir()
            layout["models_is_dir"] = (work / "models").is_dir()
            # The 2026-04-15 regression: .txt used to be placed in tmp/txt/,
            # but segment_julius.pl reads it from the same dir as the .wav.
            layout["wav_txt_pairs"] = all(
                (wav_dir / f"{n}.wav").exists() and (wav_dir / f"{n}.txt").exists() for n in names
            )
            for n in names:
                (wav_dir / f"{n}.lab").write_text("0.0000000 0.5000000 silB\n", encoding="utf-8")
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(rja.subprocess, "run", fake_run)
        successes, errors = rja.run_segkit_batch(wav_files, txt_files, str(segkit), str(out_dir), timeout=5)

        assert errors == []
        assert sorted(n for n, _ in successes) == sorted(names)
        assert layout == {
            "perl_invocation": True,
            "bin_is_dir": True,
            "models_is_dir": True,
            "wav_txt_pairs": True,
        }
        for n in names:
            assert (out_dir / f"{n}.lab").read_text(encoding="utf-8").startswith("0.0000000")

    def test_no_lab_produced_reports_stderr_snippet(self, tmp_path, no_symlink, monkeypatch):
        import run_julius_alignment as rja

        names = ["jvs001_U1", "jvs002_U2"]
        segkit = self._make_segkit(tmp_path)
        wav_files, txt_files = self._make_inputs(tmp_path, names)

        def fake_run(cmd, **kwargs):
            return types.SimpleNamespace(returncode=1, stdout="", stderr="JULIUS_FAILED_MARKER: model not found")

        monkeypatch.setattr(rja.subprocess, "run", fake_run)
        successes, errors = rja.run_segkit_batch(wav_files, txt_files, str(segkit), str(tmp_path / "out"), timeout=5)

        assert successes == []
        assert sorted(n for n, _ in errors) == sorted(names)
        for _, msg in errors:
            assert "No .lab file produced" in msg
            assert "JULIUS_FAILED_MARKER" in msg

    def test_timeout_reports_all_names(self, tmp_path, no_symlink, monkeypatch):
        import run_julius_alignment as rja

        names = ["jvs001_U1", "jvs002_U2", "jvs003_U3"]
        segkit = self._make_segkit(tmp_path)
        wav_files, txt_files = self._make_inputs(tmp_path, names)

        def fake_run(cmd, **kwargs):
            raise subprocess.TimeoutExpired(cmd="perl", timeout=5)

        monkeypatch.setattr(rja.subprocess, "run", fake_run)
        successes, errors = rja.run_segkit_batch(wav_files, txt_files, str(segkit), str(tmp_path / "out"), timeout=5)

        assert successes == []
        assert {n for n, _ in errors} == set(names)
        assert all(msg == "Timeout expired" for _, msg in errors)

    @pytest.mark.parametrize("lab_location", ["lab", "."])
    def test_lab_discovered_in_alternate_locations(self, tmp_path, no_symlink, monkeypatch, lab_location):
        """.lab files written to tmp/lab/ or the tmp root are still collected."""
        import run_julius_alignment as rja

        names = ["jvs001_U1"]
        segkit = self._make_segkit(tmp_path)
        wav_files, txt_files = self._make_inputs(tmp_path, names)
        out_dir = tmp_path / "out"

        def fake_run(cmd, **kwargs):
            dest = Path(kwargs["cwd"]) / lab_location
            dest.mkdir(exist_ok=True)
            for n in names:
                (dest / f"{n}.lab").write_text("0.0000000 0.5000000 silB\n", encoding="utf-8")
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(rja.subprocess, "run", fake_run)
        successes, errors = rja.run_segkit_batch(wav_files, txt_files, str(segkit), str(out_dir), timeout=5)

        assert errors == []
        assert [n for n, _ in successes] == names
        assert (out_dir / "jvs001_U1.lab").exists()


# ===========================================================================
# run_julius_alignment.py: check_prerequisites
# ===========================================================================


class TestCheckPrerequisites:
    """check_prerequisites raises actionable FileNotFoundError messages."""

    def _make_valid_segkit(self, tmp_path):
        segkit = tmp_path / "segkit"
        segkit.mkdir()
        (segkit / "segment_julius.pl").write_text("# stub\n", encoding="utf-8")
        return segkit

    def test_missing_segkit_dir(self, tmp_path):
        import run_julius_alignment as rja

        with pytest.raises(FileNotFoundError, match="segmentation-kit not found"):
            rja.check_prerequisites(str(tmp_path / "missing_segkit"))

    def test_missing_segment_script(self, tmp_path):
        import run_julius_alignment as rja

        segkit = tmp_path / "segkit"
        segkit.mkdir()
        with pytest.raises(FileNotFoundError, match="segment_julius.pl not found"):
            rja.check_prerequisites(str(segkit))

    def test_julius_not_on_path(self, tmp_path, monkeypatch):
        import run_julius_alignment as rja

        segkit = self._make_valid_segkit(tmp_path)
        monkeypatch.setattr(shutil, "which", lambda cmd: None)
        with pytest.raises(FileNotFoundError, match="julius command not found"):
            rja.check_prerequisites(str(segkit))

    def test_perl_not_on_path(self, tmp_path, monkeypatch):
        import run_julius_alignment as rja

        segkit = self._make_valid_segkit(tmp_path)
        monkeypatch.setattr(shutil, "which", lambda cmd: "C:/fake/julius.exe" if cmd == "julius" else None)
        with pytest.raises(FileNotFoundError, match="perl command not found"):
            rja.check_prerequisites(str(segkit))


# ===========================================================================
# prepare_jvs.py: trim_silence edge cases
# ===========================================================================


class TestTrimSilence:
    """Energy-based silence trimming: boundaries, margins, degenerate inputs."""

    SR = 22050
    FRAME = int(SR * 0.01)  # 220 samples per 10ms energy frame
    MARGIN = int(SR * 50 / 1000)  # 1102 samples for the default margin_ms=50

    def _burst_waveform(self):
        """0.4s silence + 0.5s 440Hz sine burst + 0.4s silence."""
        silence = np.zeros(int(self.SR * 0.4), dtype=np.float32)
        t = np.arange(int(self.SR * 0.5), dtype=np.float32) / self.SR
        burst = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
        sig = np.concatenate([silence, burst, silence])
        return torch.from_numpy(sig).unsqueeze(0), len(burst)

    def test_burst_boundaries_within_margin_plus_one_frame(self):
        from prepare_jvs import trim_silence

        waveform, burst_len = self._burst_waveform()
        trimmed = trim_silence(waveform, self.SR)

        slack = self.MARGIN + self.FRAME
        assert burst_len <= trimmed.shape[-1] <= burst_len + 2 * slack

    def test_burst_onset_not_clipped(self):
        """Trimming only removes silence, so the signal energy is fully preserved."""
        from prepare_jvs import trim_silence

        waveform, _ = self._burst_waveform()
        trimmed = trim_silence(waveform, self.SR)

        assert torch.allclose(trimmed.pow(2).sum(), waveform.pow(2).sum())

    def test_all_zero_input_unchanged(self):
        from prepare_jvs import trim_silence

        waveform = torch.zeros(1, self.SR)
        out = trim_silence(waveform, self.SR)

        assert torch.equal(out, waveform)

    def test_shorter_than_one_frame_unchanged(self):
        from prepare_jvs import trim_silence

        waveform = torch.linspace(-0.3, 0.3, self.FRAME - 1).unsqueeze(0)
        out = trim_silence(waveform, self.SR)

        assert torch.equal(out, waveform)

    def test_zero_length_input_unchanged(self):
        from prepare_jvs import trim_silence

        waveform = torch.zeros(1, 0)
        out = trim_silence(waveform, self.SR)

        assert out.shape == (1, 0)

    def test_fully_voiced_returns_almost_full_length(self):
        from prepare_jvs import trim_silence

        t = torch.arange(self.SR, dtype=torch.float32) / self.SR
        waveform = (0.5 * torch.sin(2 * torch.pi * 440.0 * t)).unsqueeze(0)
        out = trim_silence(waveform, self.SR)

        assert out.shape[-1] >= self.SR - self.FRAME
