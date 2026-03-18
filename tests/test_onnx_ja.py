"""Tests for ONNX export/infer Japanese language support."""

import sys
import types

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
        """Japanese vocab (n_vocab=52) should constrain dummy input range."""
        inputs, names = get_inputs(False, n_vocab=52)
        x = inputs[0]
        assert x.max() < 52

    def test_small_vocab_size(self):
        """Very small vocab should still work."""
        inputs, names = get_inputs(False, n_vocab=5)
        x = inputs[0]
        assert x.max() < 5

    def test_input_names_single_speaker(self):
        """Single speaker should have 3 inputs."""
        inputs, names = get_inputs(False, n_vocab=52)
        assert len(inputs) == 3
        assert names == ["x", "x_lengths", "scales"]

    def test_input_names_multi_speaker(self):
        """Multi speaker should have 4 inputs with spks."""
        inputs, names = get_inputs(True, n_vocab=52)
        assert len(inputs) == 4
        assert names == ["x", "x_lengths", "scales", "spks"]

    def test_scales_values(self):
        """Scales should contain temperature and length_scale."""
        inputs, _ = get_inputs(False, n_vocab=52)
        scales = inputs[2]
        assert scales.shape == (2,)
        assert scales[0].item() == pytest.approx(0.667)
        assert scales[1].item() == pytest.approx(1.0)

    def test_x_lengths_matches_x(self):
        """x_lengths should match x sequence length."""
        inputs, _ = get_inputs(False, n_vocab=52)
        x, x_lengths = inputs[0], inputs[1]
        assert x_lengths.item() == x.shape[1]
