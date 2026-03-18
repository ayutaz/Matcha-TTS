"""Tests for matcha.data.text_mel_datamodule (collation, instantiation, utilities)."""

import pytest
import torch

from matcha.data.text_mel_datamodule import (
    TextMelBatchCollate,
    TextMelDataModule,
    parse_filelist,
)
from matcha.utils.audio import mel_spectrogram
from matcha.utils.model import fix_len_compatibility

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_hparams():
    """Return a minimal set of kwargs accepted by TextMelDataModule.__init__."""
    return dict(
        name="test",
        train_filelist_path="train.txt",
        valid_filelist_path="val.txt",
        batch_size=4,
        num_workers=0,
        pin_memory=False,
        cleaners=["english_cleaners2"],
        add_blank=True,
        n_spks=1,
        n_fft=1024,
        n_feats=80,
        sample_rate=22050,
        hop_length=256,
        win_length=1024,
        f_min=0,
        f_max=8000,
        data_statistics={"mel_mean": 0, "mel_std": 1},
        seed=42,
        load_durations=False,
    )


def _make_batch_item(x_len, y_len, n_feats=80, spk=None, durations=None):
    """Synthesise one element as would be returned by TextMelDataset.__getitem__."""
    return {
        "x": torch.randint(1, 178, (x_len,)),
        "y": torch.randn(n_feats, y_len),
        "spk": spk,
        "filepath": f"/fake/audio_{x_len}.wav",
        "x_text": "hello world",
        "durations": durations,
    }


# ---------------------------------------------------------------------------
# parse_filelist
# ---------------------------------------------------------------------------


class TestParseFilelist:
    def test_basic_parsing(self, tmp_path):
        flist = tmp_path / "filelist.txt"
        flist.write_text("audio1.wav|Hello world\naudio2.wav|Goodbye\n")
        result = parse_filelist(str(flist))
        assert len(result) == 2
        assert result[0] == ["audio1.wav", "Hello world"]
        assert result[1] == ["audio2.wav", "Goodbye"]

    def test_custom_delimiter(self, tmp_path):
        flist = tmp_path / "filelist.txt"
        flist.write_text("audio1.wav\tHello world\naudio2.wav\tGoodbye\n")
        result = parse_filelist(str(flist), split_char="\t")
        assert len(result) == 2
        assert result[0] == ["audio1.wav", "Hello world"]

    def test_multispeaker_format(self, tmp_path):
        flist = tmp_path / "filelist.txt"
        flist.write_text("audio1.wav|0|Hello world\naudio2.wav|1|Goodbye\n")
        result = parse_filelist(str(flist))
        assert len(result) == 2
        assert result[0] == ["audio1.wav", "0", "Hello world"]

    def test_empty_file(self, tmp_path):
        flist = tmp_path / "filelist.txt"
        flist.write_text("")
        result = parse_filelist(str(flist))
        # An empty file produces one entry with a single empty string
        # because the final newline is missing; verify it does not crash.
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# TextMelDataModule instantiation
# ---------------------------------------------------------------------------


class TestTextMelDataModuleInstantiation:
    def test_instantiation_stores_hparams(self):
        dm = TextMelDataModule(**_default_hparams())
        assert dm.hparams.name == "test"
        assert dm.hparams.batch_size == 4
        assert dm.hparams.n_feats == 80
        assert dm.hparams.n_spks == 1
        assert dm.hparams.sample_rate == 22050

    def test_instantiation_with_multispeaker(self):
        hp = _default_hparams()
        hp["n_spks"] = 10
        dm = TextMelDataModule(**hp)
        assert dm.hparams.n_spks == 10

    def test_instantiation_custom_batch_size(self):
        hp = _default_hparams()
        hp["batch_size"] = 16
        dm = TextMelDataModule(**hp)
        assert dm.hparams.batch_size == 16

    def test_state_dict_empty(self):
        dm = TextMelDataModule(**_default_hparams())
        assert dm.state_dict() == {}

    def test_load_state_dict_noop(self):
        dm = TextMelDataModule(**_default_hparams())
        # Should not raise
        dm.load_state_dict({"key": "value"})

    def test_teardown_noop(self):
        dm = TextMelDataModule(**_default_hparams())
        # Should not raise for any stage
        dm.teardown(stage="fit")
        dm.teardown(stage="test")
        dm.teardown(stage=None)


# ---------------------------------------------------------------------------
# TextMelBatchCollate — single-speaker
# ---------------------------------------------------------------------------


class TestTextMelBatchCollateSingleSpeaker:
    """Collation tests with n_spks=1 (no speaker id tensor)."""

    def _collate(self, batch):
        return TextMelBatchCollate(n_spks=1)(batch)

    def test_output_keys(self):
        batch = [_make_batch_item(10, 50)]
        out = self._collate(batch)
        expected_keys = {"x", "x_lengths", "y", "y_lengths", "spks", "filepaths", "x_texts", "durations"}
        assert set(out.keys()) == expected_keys

    def test_single_item_shapes(self):
        n_feats = 80
        x_len, y_len = 12, 60
        batch = [_make_batch_item(x_len, y_len, n_feats=n_feats)]
        out = self._collate(batch)

        y_compat = fix_len_compatibility(y_len)

        assert out["x"].shape == (1, x_len)
        assert out["x_lengths"].shape == (1,)
        assert out["x_lengths"].item() == x_len
        assert out["y"].shape == (1, n_feats, y_compat)
        assert out["y_lengths"].shape == (1,)
        assert out["y_lengths"].item() == y_len

    def test_spks_none_for_single_speaker(self):
        batch = [_make_batch_item(10, 50)]
        out = self._collate(batch)
        assert out["spks"] is None

    def test_durations_none_when_absent(self):
        batch = [_make_batch_item(10, 50)]
        out = self._collate(batch)
        assert out["durations"] is None

    def test_multiple_items_padded_correctly(self):
        n_feats = 80
        items = [
            _make_batch_item(8, 40, n_feats=n_feats),
            _make_batch_item(15, 70, n_feats=n_feats),
            _make_batch_item(5, 30, n_feats=n_feats),
        ]
        out = self._collate(items)
        B = 3
        x_max = 15
        y_max = fix_len_compatibility(70)

        assert out["x"].shape == (B, x_max)
        assert out["y"].shape == (B, n_feats, y_max)
        assert out["x_lengths"].tolist() == [8, 15, 5]
        assert out["y_lengths"].tolist() == [40, 70, 30]

    def test_padding_is_zero(self):
        """Padding regions in x and y must be zero."""
        n_feats = 80
        items = [
            _make_batch_item(5, 30, n_feats=n_feats),
            _make_batch_item(10, 50, n_feats=n_feats),
        ]
        out = self._collate(items)

        # For the shorter x (index 0), positions 5.. should be 0
        assert (out["x"][0, 5:] == 0).all()

        # For the shorter y (index 0), time positions 30.. should be 0
        assert (out["y"][0, :, 30:] == 0).all()

    def test_filepaths_and_texts_preserved(self):
        items = [
            _make_batch_item(5, 30),
            _make_batch_item(8, 40),
        ]
        out = self._collate(items)
        assert len(out["filepaths"]) == 2
        assert len(out["x_texts"]) == 2
        assert all(isinstance(fp, str) for fp in out["filepaths"])

    def test_dtypes(self):
        batch = [_make_batch_item(10, 50)]
        out = self._collate(batch)
        assert out["x"].dtype == torch.long
        assert out["y"].dtype == torch.float32
        assert out["x_lengths"].dtype == torch.long
        assert out["y_lengths"].dtype == torch.long


# ---------------------------------------------------------------------------
# TextMelBatchCollate — multi-speaker
# ---------------------------------------------------------------------------


class TestTextMelBatchCollateMultiSpeaker:
    """Collation tests with n_spks > 1 (speaker ids present)."""

    def _collate(self, batch):
        return TextMelBatchCollate(n_spks=2)(batch)

    def test_spks_tensor_present(self):
        items = [
            _make_batch_item(10, 50, spk=0),
            _make_batch_item(8, 40, spk=1),
        ]
        out = self._collate(items)
        assert out["spks"] is not None
        assert out["spks"].dtype == torch.long
        assert out["spks"].tolist() == [0, 1]

    def test_spks_shape(self):
        items = [
            _make_batch_item(10, 50, spk=0),
            _make_batch_item(8, 40, spk=1),
            _make_batch_item(12, 60, spk=0),
        ]
        out = self._collate(items)
        assert out["spks"].shape == (3,)


# ---------------------------------------------------------------------------
# TextMelBatchCollate — with durations
# ---------------------------------------------------------------------------


class TestTextMelBatchCollateWithDurations:
    """Collation tests when duration tensors are provided."""

    def _collate(self, batch):
        return TextMelBatchCollate(n_spks=1)(batch)

    def test_durations_present(self):
        x_len = 10
        durs = torch.ones(x_len, dtype=torch.long) * 5
        items = [_make_batch_item(x_len, 50, durations=durs)]
        out = self._collate(items)
        assert out["durations"] is not None
        assert out["durations"].shape[0] == 1
        assert out["durations"].shape[1] == x_len

    def test_durations_padded_to_max_x_len(self):
        durs_short = torch.ones(5, dtype=torch.long) * 3
        durs_long = torch.ones(12, dtype=torch.long) * 2
        items = [
            _make_batch_item(5, 40, durations=durs_short),
            _make_batch_item(12, 60, durations=durs_long),
        ]
        out = self._collate(items)
        assert out["durations"].shape == (2, 12)
        # Check that short durations are padded with zeros
        assert (out["durations"][0, 5:] == 0).all()
        # Check that actual values are preserved
        assert (out["durations"][0, :5] == 3).all()
        assert (out["durations"][1, :12] == 2).all()

    def test_all_none_durations_returns_none(self):
        items = [
            _make_batch_item(5, 40, durations=None),
            _make_batch_item(8, 50, durations=None),
        ]
        out = self._collate(items)
        assert out["durations"] is None


# ---------------------------------------------------------------------------
# fix_len_compatibility
# ---------------------------------------------------------------------------


class TestFixLenCompatibility:
    """Verify the U-Net length rounding used during collation."""

    def test_already_compatible(self):
        assert fix_len_compatibility(4) == 4
        assert fix_len_compatibility(8) == 8

    def test_rounds_up(self):
        assert fix_len_compatibility(5) == 8
        assert fix_len_compatibility(3) == 4

    def test_custom_downsampling(self):
        # With 3 downsamplings, the factor is 2^3 = 8
        assert fix_len_compatibility(9, num_downsamplings_in_unet=3) == 16
        assert fix_len_compatibility(8, num_downsamplings_in_unet=3) == 8


# ---------------------------------------------------------------------------
# mel_spectrogram utility
# ---------------------------------------------------------------------------


class TestMelSpectrogram:
    """Basic smoke test that mel_spectrogram runs on CPU with a synthetic signal."""

    def test_output_shape(self):
        sr = 22050
        n_fft = 1024
        hop_length = 256
        n_mels = 80
        duration_sec = 0.5
        n_samples = int(sr * duration_sec)

        waveform = torch.randn(1, n_samples)
        mel = mel_spectrogram(
            waveform,
            n_fft=n_fft,
            num_mels=n_mels,
            sampling_rate=sr,
            hop_size=hop_length,
            win_size=n_fft,
            fmin=0,
            fmax=8000,
            center=False,
        )
        assert mel.ndim == 3
        assert mel.shape[0] == 1
        assert mel.shape[1] == n_mels
        # Time frames: roughly n_samples / hop_length
        expected_frames = n_samples // hop_length
        assert abs(mel.shape[2] - expected_frames) <= 2

    def test_deterministic(self):
        sr = 22050
        waveform = torch.randn(1, sr // 2)
        kwargs = dict(
            n_fft=1024,
            num_mels=80,
            sampling_rate=sr,
            hop_size=256,
            win_size=1024,
            fmin=0,
            fmax=8000,
            center=False,
        )
        mel1 = mel_spectrogram(waveform, **kwargs)
        mel2 = mel_spectrogram(waveform, **kwargs)
        assert torch.allclose(mel1, mel2)
