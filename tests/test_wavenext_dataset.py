"""TDD tests for wavenext_train.dataset (waveform dataset, soundfile load, peak-norm)."""

import inspect

import numpy as np
import soundfile as sf
import torch

import wavenext_train.dataset as dataset_mod
from wavenext_train.dataset import DataConfig, VocosDataModule, VocosDataset

SR = 22050
NUM = 16384


def _write_wav(path, samples, sr=SR):
    sf.write(str(path), np.asarray(samples, dtype=np.float32), sr)
    return str(path)


def _cfg(filelist, batch_size=1, num_workers=0):
    return DataConfig(filelist_path=str(filelist), sampling_rate=SR, num_samples=NUM,
                      batch_size=batch_size, num_workers=num_workers)


def _filelist(tmp_path, wavs):
    fl = tmp_path / "fl.txt"
    fl.write_text("\n".join(wavs) + "\n", encoding="utf-8")
    return fl


def test_getitem_returns_num_samples_1d_waveform(tmp_path):
    w = _write_wav(tmp_path / "a.wav", np.random.uniform(-1, 1, 30000))
    ds = VocosDataset(_cfg(_filelist(tmp_path, [w])), train=True)
    out = ds[0]
    assert out.shape == (NUM,) and out.ndim == 1 and out.dtype == torch.float32
    assert torch.isfinite(out).all()


def test_short_audio_is_repeat_padded_to_num_samples(tmp_path):
    w = _write_wav(tmp_path / "s.wav", np.random.uniform(-1, 1, 5000))
    ds = VocosDataset(_cfg(_filelist(tmp_path, [w])), train=False)  # fixed gain -> deterministic
    out = ds[0]
    assert out.shape == (NUM,)
    assert torch.allclose(out[5000:10000], out[0:5000])
    assert torch.allclose(out[10000:15000], out[0:5000])


def test_stereo_is_downmixed_to_mono(tmp_path):
    stereo = np.random.uniform(-1, 1, (30000, 2))
    w = _write_wav(tmp_path / "st.wav", stereo)
    ds = VocosDataset(_cfg(_filelist(tmp_path, [w])), train=False)
    out = ds[0]
    assert out.shape == (NUM,) and out.ndim == 1 and torch.isfinite(out).all()


def test_output_value_range_within_unit_interval(tmp_path):
    peaky = np.random.uniform(-1, 1, 30000).astype(np.float32)
    peaky[100] = 1.0
    w = _write_wav(tmp_path / "p.wav", peaky)
    for train in (True, False):
        out = VocosDataset(_cfg(_filelist(tmp_path, [w])), train=train)[0]
        assert out.abs().max() <= 1.0


def test_peak_normalization_matches_sox_norm_semantics(tmp_path):
    y = np.random.uniform(-0.5, 0.5, NUM).astype(np.float32)
    y[0] = 1.0  # global peak within the first NUM samples
    w = _write_wav(tmp_path / "n.wav", y)
    out = VocosDataset(_cfg(_filelist(tmp_path, [w])), train=False)[0]  # gain=-3 fixed
    assert abs(out.abs().max().item() - 10 ** (-3 / 20)) < 1e-3
    assert "sox_effects" not in inspect.getsource(dataset_mod)  # regression: no sox backend


def test_dataloader_produces_batched_waveforms(tmp_path):
    wavs = [_write_wav(tmp_path / f"d{i}.wav", np.random.uniform(-1, 1, 30000)) for i in range(5)]
    dm = VocosDataModule(_cfg(_filelist(tmp_path, wavs), batch_size=3), _cfg(_filelist(tmp_path, wavs), batch_size=3))
    batch = next(iter(dm.train_dataloader()))
    assert batch.shape == (3, NUM) and batch.dtype == torch.float32
