"""Tests for matcha.data.text_mel_datamodule (collation, instantiation, utilities)."""

import numpy as np
import pytest
import torch

from matcha.data.text_mel_datamodule import (
    TextMelBatchCollate,
    TextMelDataModule,
    TextMelDataset,
    parse_filelist,
)
from matcha.text import cleaned_text_to_sequence
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

    @pytest.mark.parametrize("load_durations", [True, False])
    def test_dataloaders_pass_load_durations_to_collate(self, tmp_path, load_durations):
        """load_durations must be wired through to the dataloader collate_fn so
        that duration-return semantics stay consistent with the dataset."""
        flist = tmp_path / "filelist.txt"
        flist.write_text("dummy.wav|hello\n", encoding="utf-8")
        hp = _default_hparams()
        hp["train_filelist_path"] = str(flist)
        hp["valid_filelist_path"] = str(flist)
        hp["load_durations"] = load_durations
        dm = TextMelDataModule(**hp)
        dm.setup()
        assert dm.train_dataloader().collate_fn.load_durations is load_durations
        assert dm.val_dataloader().collate_fn.load_durations is load_durations


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


# ---------------------------------------------------------------------------
# Helpers for filelist-backed datamodule / dataset construction
# ---------------------------------------------------------------------------


def _write_filelist(tmp_path, line="dummy.wav|a i u", name="filelist.txt"):
    """Write a one-line filelist and return its path as a string."""
    flist = tmp_path / name
    flist.write_text(line + "\n", encoding="utf-8")
    return str(flist)


def _make_dataset(tmp_path, language="ja", load_durations=False, name="filelist.txt"):
    """Build a TextMelDataset over a tiny filelist (basic_cleaners, no pyopenjtalk needed)."""
    return TextMelDataset(
        filelist_path=_write_filelist(tmp_path, name=name),
        n_spks=1,
        cleaners=["basic_cleaners"],
        add_blank=True,
        seed=42,
        load_durations=load_durations,
        language=language,
    )


# ---------------------------------------------------------------------------
# Language plumbing (TextMelDataModule.setup)
# ---------------------------------------------------------------------------


class TestLanguagePlumbing:
    """setup() must forward the `language` hparam to both train and valid datasets."""

    def _datamodule(self, tmp_path, **overrides):
        flist = _write_filelist(tmp_path)
        hp = _default_hparams()
        hp["train_filelist_path"] = flist
        hp["valid_filelist_path"] = flist
        hp.update(overrides)
        dm = TextMelDataModule(**hp)
        dm.setup()
        return dm

    def test_language_ja_reaches_both_datasets(self, tmp_path):
        dm = self._datamodule(tmp_path, language="ja")
        assert dm.trainset.language == "ja"
        assert dm.validset.language == "ja"

    def test_language_defaults_to_en_when_omitted(self, tmp_path):
        """Omitting `language` falls back to 'en' (getattr default in setup)."""
        dm = self._datamodule(tmp_path)
        assert dm.trainset.language == "en"
        assert dm.validset.language == "en"


# ---------------------------------------------------------------------------
# TextMelDataset.get_text — Japanese language dispatch
# ---------------------------------------------------------------------------


class TestGetTextJapanese:
    """get_text must dispatch to the Japanese symbol table when language='ja'.

    Uses basic_cleaners over space-separated phonemes so no pyopenjtalk is
    required (same trick as tests/test_text_ja.py).
    """

    TEXT = "k o N n i ch i w a"

    def test_blank_interspersed_ja_ids(self, tmp_path):
        ds = _make_dataset(tmp_path, language="ja")
        x, cleaned = ds.get_text(self.TEXT, add_blank=True)
        expected_ids = cleaned_text_to_sequence(cleaned, language="ja")
        # Odd positions carry the phoneme ids, even positions are blanks (0)
        assert x[1::2].tolist() == expected_ids
        assert (x[0::2] == 0).all()
        assert len(x) == 2 * len(expected_ids) + 1

    def test_ja_one_id_per_token(self, tmp_path):
        ds = _make_dataset(tmp_path, language="ja")
        x, _ = ds.get_text(self.TEXT, add_blank=True)
        n_tokens = len(self.TEXT.split())
        assert len(x[1::2]) == n_tokens

    def test_en_and_ja_encodings_differ(self, tmp_path):
        ds_ja = _make_dataset(tmp_path, language="ja", name="filelist_ja.txt")
        ds_en = _make_dataset(tmp_path, language="en", name="filelist_en.txt")
        x_ja, _ = ds_ja.get_text(self.TEXT, add_blank=True)
        x_en, _ = ds_en.get_text(self.TEXT, add_blank=True)
        # ja encodes one id per space-separated token, en one id per character
        assert x_ja.tolist() != x_en.tolist()


# ---------------------------------------------------------------------------
# TextMelDataset.get_durations
# ---------------------------------------------------------------------------


class TestGetDurations:
    """get_durations resolves <data_dir>/durations/<name>.npy relative to the wav path."""

    def _layout(self, tmp_path):
        """Create <tmp_path>/data_dir/{wavs,durations} and return the two paths.

        The wav file itself is never opened by get_durations — only its path matters.
        """
        wav_dir = tmp_path / "data_dir" / "wavs"
        dur_dir = tmp_path / "data_dir" / "durations"
        wav_dir.mkdir(parents=True)
        dur_dir.mkdir(parents=True)
        return wav_dir / "utt1.wav", dur_dir / "utt1.npy"

    def _text_tensor(self, tmp_path):
        ds = _make_dataset(tmp_path, language="ja", load_durations=True)
        text, _ = ds.get_text("a i u", add_blank=True)  # interspersed: 2*3 + 1 = 7
        return ds, text

    def test_matching_length_returns_saved_values(self, tmp_path):
        wav_path, dur_path = self._layout(tmp_path)
        ds, text = self._text_tensor(tmp_path)
        saved = np.array([1, 2, 3, 4, 5, 6, 7])
        np.save(dur_path, saved)

        durs = ds.get_durations(str(wav_path), text)

        assert isinstance(durs, torch.Tensor)
        assert not durs.dtype.is_floating_point
        assert durs.tolist() == saved.tolist()
        assert len(durs) == len(text)

    def test_length_mismatch_raises_assertion(self, tmp_path):
        wav_path, dur_path = self._layout(tmp_path)
        ds, text = self._text_tensor(tmp_path)
        np.save(dur_path, np.array([1, 2, 3]))  # 3 != 7

        with pytest.raises(AssertionError, match="do not match"):
            ds.get_durations(str(wav_path), text)

    def test_missing_npy_raises_with_guidance(self, tmp_path):
        wav_path, _ = self._layout(tmp_path)  # .npy intentionally not written
        ds, text = self._text_tensor(tmp_path)

        with pytest.raises(FileNotFoundError, match="make sure you've generate the durations"):
            ds.get_durations(str(wav_path), text)


# ---------------------------------------------------------------------------
# DataLoader kwargs guard (persistent_workers / prefetch_factor)
# ---------------------------------------------------------------------------


class TestDataLoaderKwargs:
    """num_workers=0 must not pass persistent_workers/prefetch_factor to DataLoader.

    Constructing the DataLoaders is side-effect free (no workers spawned, no
    audio opened), so these tests never iterate them.
    """

    def _datamodule(self, tmp_path, num_workers=0):
        flist = _write_filelist(tmp_path)
        hp = _default_hparams()
        hp["train_filelist_path"] = flist
        hp["valid_filelist_path"] = flist
        hp["num_workers"] = num_workers
        dm = TextMelDataModule(**hp)
        dm.setup()
        return dm

    def test_zero_workers_constructs_without_persistence(self, tmp_path):
        dm = self._datamodule(tmp_path, num_workers=0)
        train_dl = dm.train_dataloader()  # would raise if persistent_workers were passed
        val_dl = dm.val_dataloader()
        assert train_dl.num_workers == 0
        assert train_dl.persistent_workers is False
        assert val_dl.persistent_workers is False

    def test_train_drops_last_val_keeps_last(self, tmp_path):
        dm = self._datamodule(tmp_path, num_workers=0)
        assert dm.train_dataloader().drop_last is True
        assert dm.val_dataloader().drop_last is False

    def test_positive_workers_enable_persistence(self, tmp_path):
        dm = self._datamodule(tmp_path, num_workers=2)
        train_dl = dm.train_dataloader()
        val_dl = dm.val_dataloader()
        assert train_dl.persistent_workers is True
        assert train_dl.prefetch_factor == 4
        assert val_dl.persistent_workers is True
        assert val_dl.prefetch_factor == 4


# ---------------------------------------------------------------------------
# mel_spectrogram — exact frame count with center=False
# ---------------------------------------------------------------------------


class TestMelSpectrogramExactFrames:
    """With center=False and the (n_fft - hop)/2 reflect pad, the frame count
    must be exactly 1 + (L - hop) // hop == L // hop for any sample length L."""

    @pytest.mark.parametrize("length", [4 * 256, 4 * 256 + 1, 4 * 256 + 255, 7 * 256 + 128])
    def test_exact_frame_count(self, length):
        torch.manual_seed(0)
        waveform = torch.rand(1, length) * 1.8 - 0.9  # in [-0.9, 0.9]
        mel = mel_spectrogram(
            waveform,
            n_fft=1024,
            num_mels=80,
            sampling_rate=22050,
            hop_size=256,
            win_size=1024,
            fmin=0,
            fmax=8000,
            center=False,
        )
        expected_frames = 1 + (length - 256) // 256
        assert expected_frames == length // 256
        assert mel.shape == (1, 80, expected_frames)
        assert torch.isfinite(mel).all()
        assert mel.dtype == torch.float32
