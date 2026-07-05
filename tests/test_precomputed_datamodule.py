"""Tests for PrecomputedDataModule duration loading."""

from pathlib import Path

import pytest
import torch

from matcha.data.precomputed_datamodule import PrecomputedTextMelDataset
from matcha.data.text_mel_datamodule import TextMelBatchCollate


def _make_pt_file(path: Path, text_len: int, mel_len: int, spk: int, include_durations: bool = False):
    """Helper: create a dummy .pt file."""
    text = torch.randint(0, 55, (text_len,), dtype=torch.int32)
    mel = torch.randn(80, mel_len)
    save_dict = {
        "mel": mel,
        "text": text,
        "spk": spk,
        "cleaned_text": "dummy text",
    }
    if include_durations:
        dur = torch.zeros(text_len, dtype=torch.int32)
        dur[1::2] = 5  # 5 frames at phoneme positions
        save_dict["durations"] = dur
    torch.save(save_dict, path)


class TestLoadFromDiskWithDurations:
    def test_load_with_durations(self, tmp_path):
        """load_durations=True loads duration array from .pt file."""
        _make_pt_file(tmp_path / "sample.pt", 11, 100, 0, include_durations=True)
        ds = PrecomputedTextMelDataset(tmp_path, n_spks=100, load_durations=True)
        item = ds[0]
        assert item["durations"] is not None
        assert len(item["durations"]) == 11

    def test_load_without_durations(self, tmp_path):
        """load_durations=False returns durations=None."""
        _make_pt_file(tmp_path / "sample.pt", 11, 100, 0, include_durations=False)
        ds = PrecomputedTextMelDataset(tmp_path, n_spks=100, load_durations=False)
        item = ds[0]
        assert item["durations"] is None

    def test_missing_durations_key_raises(self, tmp_path):
        """load_durations=True with missing key raises KeyError."""
        _make_pt_file(tmp_path / "sample.pt", 11, 100, 0, include_durations=False)
        ds = PrecomputedTextMelDataset(tmp_path, n_spks=100, load_durations=True)
        with pytest.raises(KeyError, match="load_durations=True"):
            ds[0]

    def test_duration_length_mismatch_raises(self, tmp_path):
        """Mismatched duration and text lengths raise ValueError."""
        # text=11, duration=7 (mismatch)
        text = torch.randint(0, 55, (11,), dtype=torch.int32)
        dur = torch.zeros(7, dtype=torch.int32)
        torch.save(
            {"mel": torch.randn(80, 100), "text": text, "spk": 0, "cleaned_text": "dummy", "durations": dur},
            tmp_path / "bad.pt",
        )
        ds = PrecomputedTextMelDataset(tmp_path, n_spks=100, load_durations=True)
        with pytest.raises(ValueError, match="Duration length"):
            ds[0]

    def test_preload_to_memory_with_durations(self, tmp_path):
        """preload_to_memory=True caches durations in memory."""
        _make_pt_file(tmp_path / "s1.pt", 11, 100, 0, include_durations=True)
        _make_pt_file(tmp_path / "s2.pt", 9, 80, 1, include_durations=True)
        ds = PrecomputedTextMelDataset(tmp_path, n_spks=100, load_durations=True, preload_to_memory=True)
        assert 0 in ds._cache
        assert ds._cache[0]["durations"] is not None


class TestCollateWithDurations:
    def _make_batch(self, include_durations=False, text_lens=(11, 9)):
        """Generate a dummy batch."""
        batch = []
        for i, tl in enumerate(text_lens):
            item = {
                "x": torch.randint(0, 55, (tl,)),
                "y": torch.randn(80, 50 + i * 10),
                "spk": i,
                "filepath": f"/path/sample_{i}.pt",
                "x_text": f"text_{i}",
                "durations": None,
            }
            if include_durations:
                dur = torch.zeros(tl, dtype=torch.long)
                dur[1::2] = 5
                item["durations"] = dur
            batch.append(item)
        return batch

    def test_collate_with_durations_shape(self):
        """Batch with durations returns (B, max_x_len) shaped tensor."""
        batch = self._make_batch(include_durations=True)
        collate = TextMelBatchCollate(n_spks=100, load_durations=True)
        result = collate(batch)
        assert result["durations"] is not None
        assert result["durations"].shape[0] == 2  # batch size
        assert result["durations"].shape[1] >= 11  # max text length
        assert result["durations"].dtype == torch.long

    def test_collate_without_durations_returns_none(self):
        """Batch with durations=None and load_durations=False returns None."""
        batch = self._make_batch(include_durations=False)
        collate = TextMelBatchCollate(n_spks=100, load_durations=False)
        result = collate(batch)
        assert result["durations"] is None

    def test_collate_load_durations_true_never_returns_none(self):
        """load_durations=True always returns a tensor even if all zeros."""
        batch = self._make_batch(include_durations=False)
        # durations are None, but collate initializes an all-zero tensor
        collate = TextMelBatchCollate(n_spks=100, load_durations=True)
        result = collate(batch)
        # load_durations=True: always return the tensor
        assert result["durations"] is not None
        assert isinstance(result["durations"], torch.Tensor)

    def test_collate_padding_correctness(self):
        """Padding region of durations must be zero."""
        batch = self._make_batch(include_durations=True, text_lens=(11, 5))
        collate = TextMelBatchCollate(n_spks=100, load_durations=True)
        result = collate(batch)
        # Second sample (text_len=5) should have zero padding beyond position 5
        assert result["durations"][1, 5:].sum() == 0

    def test_backward_compat_old_collate(self):
        """Existing code without load_durations kwarg still works."""
        batch = self._make_batch(include_durations=False)
        collate = TextMelBatchCollate(n_spks=100)  # no load_durations arg
        result = collate(batch)
        assert result["durations"] is None

    def test_single_speaker_mode(self):
        """n_spks=1 returns spks=None (backward compat) with durations present."""
        batch = self._make_batch(include_durations=True)
        for item in batch:
            item["spk"] = None
        collate = TextMelBatchCollate(n_spks=1, load_durations=True)
        result = collate(batch)
        assert result["spks"] is None
        assert result["durations"] is not None
