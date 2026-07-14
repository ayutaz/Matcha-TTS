"""Tests for ONNX export/infer Japanese language support."""

import sys
import types

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Mock phonemizer before any matcha import, because matcha/text/cleaners.py
# executes `phonemizer.backend.EspeakBackend(...)` at module level and would
# fail without espeak-ng installed.
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

from matcha.onnx.export import get_inputs  # noqa: E402


class TestGetInputsVocab:
    """Test get_inputs with different vocabulary sizes."""

    def test_default_vocab_size(self):
        """Default vocab size should be 178 (English)."""
        inputs, names = get_inputs(False)
        x = inputs[0]
        assert x.dtype == torch.long
        assert x.shape == (1, 50)
        assert x.min() >= 0
        assert x.max() < 178

    def test_japanese_vocab_size(self):
        """Japanese vocab (n_vocab=55) should constrain dummy input range."""
        inputs, names = get_inputs(False, n_vocab=55)
        x = inputs[0]
        assert x.max() < 55

    def test_small_vocab_size(self):
        """Very small vocab should still work."""
        inputs, names = get_inputs(False, n_vocab=5)
        x = inputs[0]
        assert x.max() < 5

    def test_input_names_single_speaker(self):
        """Single speaker should have 3 inputs."""
        inputs, names = get_inputs(False, n_vocab=55)
        assert len(inputs) == 3
        assert names == ["x", "x_lengths", "scales"]

    def test_input_names_multi_speaker(self):
        """Multi speaker should have 4 inputs with spks."""
        inputs, names = get_inputs(True, n_vocab=55)
        assert len(inputs) == 4
        assert names == ["x", "x_lengths", "scales", "spks"]

    def test_scales_values(self):
        """Scales should contain temperature and length_scale."""
        inputs, _ = get_inputs(False, n_vocab=55)
        scales = inputs[2]
        assert scales.shape == (2,)
        assert scales[0].item() == pytest.approx(0.667)
        assert scales[1].item() == pytest.approx(1.0)

    def test_x_lengths_matches_x(self):
        """x_lengths should match x sequence length."""
        inputs, _ = get_inputs(False, n_vocab=55)
        x, x_lengths = inputs[0], inputs[1]
        assert x_lengths.item() == x.shape[1]


# ---------------------------------------------------------------------------
# matcha.onnx.infer imports onnxruntime at module level, but the helpers under
# test (resolve_model_path, write_mels) never touch the ort object. Stub the
# module when the 'onnx' extra is not installed (it is absent locally and in
# CI, which installs dependency groups only) so the regression tests always
# execute instead of being skipped.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def onnx_infer():
    try:
        import onnxruntime  # noqa: F401

        stubbed = False
    except ImportError:
        sys.modules["onnxruntime"] = types.ModuleType("onnxruntime")
        stubbed = True
    try:
        from matcha.onnx import infer

        yield infer
    finally:
        if stubbed:
            del sys.modules["onnxruntime"]


class TestResolveModelPath:
    """Test INT8 quantized model path resolution."""

    def test_unquantized_path_returned_verbatim(self, onnx_infer, tmp_path):
        model = tmp_path / "model.onnx"
        assert onnx_infer.resolve_model_path(str(model), quantized=False) == str(model)

    def test_quantized_derives_int8_sibling(self, onnx_infer, tmp_path):
        model = tmp_path / "model.onnx"
        int8 = tmp_path / "model_int8.onnx"
        int8.write_bytes(b"\x00")
        assert onnx_infer.resolve_model_path(str(model), quantized=True) == str(int8)

    def test_quantized_accepts_already_int8_path(self, onnx_infer, tmp_path):
        """Regression: model_int8.onnx used to derive model_int8_int8.onnx
        and raise FileNotFoundError even though the given file exists."""
        int8 = tmp_path / "model_int8.onnx"
        int8.write_bytes(b"\x00")
        assert onnx_infer.resolve_model_path(str(int8), quantized=True) == str(int8)

    def test_quantized_falls_back_to_existing_given_path(self, onnx_infer, tmp_path):
        """No _int8 sibling, but the given model exists: use it with a warning."""
        model = tmp_path / "model.onnx"
        model.write_bytes(b"\x00")
        with pytest.warns(UserWarning, match="falling back"):
            assert onnx_infer.resolve_model_path(str(model), quantized=True) == str(model)

    def test_quantized_missing_everything_raises(self, onnx_infer, tmp_path):
        model = tmp_path / "model.onnx"
        with pytest.raises(FileNotFoundError, match="_int8"):
            onnx_infer.resolve_model_path(str(model), quantized=True)


class _FakeMelModel:
    """Stand-in for an ort.InferenceSession that returns fixed mels."""

    def __init__(self, mels, mel_lengths):
        self._outputs = (mels, mel_lengths)

    def run(self, output_names, inputs):
        return self._outputs


class TestWriteMels:
    """Test mel .npy output file naming."""

    def test_saves_plain_npy_files(self, onnx_infer, tmp_path, monkeypatch):
        """Regression: with_suffix('.numpy') made np.save append '.npy',
        producing output_N.numpy.npy instead of the promised output_N.npy."""
        monkeypatch.setattr(onnx_infer, "plot_spectrogram_to_numpy", lambda *args, **kwargs: None)
        mels = np.random.default_rng(0).standard_normal((2, 80, 12)).astype(np.float32)
        mel_lengths = np.array([12, 10], dtype=np.int64)
        model = _FakeMelModel(mels, mel_lengths)

        onnx_infer.write_mels(model, {"x": None}, tmp_path)

        for i in range(2):
            assert (tmp_path / f"output_{i + 1}.npy").exists()
            assert not (tmp_path / f"output_{i + 1}.numpy.npy").exists()
            np.testing.assert_array_equal(np.load(tmp_path / f"output_{i + 1}.npy"), mels[i])

    def test_original_indices_reorder_output_names(self, onnx_infer, tmp_path, monkeypatch):
        monkeypatch.setattr(onnx_infer, "plot_spectrogram_to_numpy", lambda *args, **kwargs: None)
        mels = np.stack([np.full((80, 8), 1.0, dtype=np.float32), np.full((80, 8), 2.0, dtype=np.float32)])
        mel_lengths = np.array([8, 8], dtype=np.int64)
        model = _FakeMelModel(mels, mel_lengths)

        # Sorted position 0 came from original index 1 and vice versa
        onnx_infer.write_mels(model, {}, tmp_path, original_indices=[1, 0])

        np.testing.assert_array_equal(np.load(tmp_path / "output_2.npy"), mels[0])
        np.testing.assert_array_equal(np.load(tmp_path / "output_1.npy"), mels[1])
