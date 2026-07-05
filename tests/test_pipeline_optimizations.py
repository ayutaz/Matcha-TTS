"""Tests for preprocessing pipeline optimizations (Tier 1 + Tier 2).

Covers:
  - T1-2: sf.info-based mel_frames calculation (formula + real data)
  - T1-3: Text cache (hiragana conversion, pickle round-trip, dedup)
  - T2-1: Dual resample in prepare_jvs (22kHz + 16kHz Julius output)
  - T2-2: precompute_with_alignment.py unified .pt generation
  - Syntax/import/CLI validation for pipeline scripts
"""

import ast
import inspect
import pickle
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

        assert checked > 0, "No real data samples found to verify"

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
            text=True,
            timeout=30,
            check=False,
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
