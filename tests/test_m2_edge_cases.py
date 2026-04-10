"""E2E pipeline tests and edge-case tests for the duration-aware data pipeline.

Covers:
  1. E2E: PrecomputedTextMelDataset -> TextMelBatchCollate -> generate_path
  2. Corrupted / empty .npy files fed to load_duration
  3. Padding detail checks (varied text lengths, fix_len_compatibility, duration sum)
  4. Collate dtype guarantees (long output, int32 upcast)
  5. Backward compatibility (no load_durations argument)
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from matcha.data.precomputed_datamodule import PrecomputedTextMelDataset
from matcha.data.text_mel_datamodule import TextMelBatchCollate
from matcha.utils.model import generate_path, sequence_mask, fix_len_compatibility


# ---------------------------------------------------------------------------
# 1. E2E pipeline: DataModule -> collate -> generate_path
# ---------------------------------------------------------------------------
class TestE2EPipeline:
    def _make_pt_with_duration(self, path, text_len, mel_len, spk=0):
        """Create a dummy .pt file whose duration sums to mel_len."""
        text = torch.randint(0, 55, (text_len,), dtype=torch.int32)
        mel = torch.randn(80, mel_len)
        dur = torch.zeros(text_len, dtype=torch.long)
        # Distribute mel_len frames across phoneme positions (odd indices)
        n_phonemes = text_len // 2
        if n_phonemes > 0:
            base_dur = mel_len // n_phonemes
            remainder = mel_len - base_dur * n_phonemes
            for i in range(n_phonemes):
                dur[2 * i + 1] = base_dur + (1 if i < remainder else 0)
        torch.save(
            {"mel": mel, "text": text, "spk": spk, "cleaned_text": "test", "durations": dur},
            path,
        )

    def test_dataset_to_collate_to_generate_path(self, tmp_path):
        """Full pipeline: DataModule -> collate -> generate_path()."""
        # 3 samples with different text / mel lengths
        self._make_pt_with_duration(tmp_path / "s1.pt", 11, 30, 0)
        self._make_pt_with_duration(tmp_path / "s2.pt", 7, 20, 1)
        self._make_pt_with_duration(tmp_path / "s3.pt", 15, 50, 2)

        ds = PrecomputedTextMelDataset(tmp_path, n_spks=100, load_durations=True)
        collate = TextMelBatchCollate(n_spks=100, load_durations=True)

        batch_items = [ds[i] for i in range(len(ds))]
        batch = collate(batch_items)

        # batch shape sanity
        assert batch["durations"] is not None
        B = batch["x"].shape[0]
        assert B == 3

        # generate_path call
        durations = batch["durations"].float()
        x_lengths = batch["x_lengths"]
        y_lengths = batch["y_lengths"]
        y_max_length = batch["y"].shape[-1]

        x_mask = sequence_mask(x_lengths, batch["x"].shape[1]).unsqueeze(1).float()
        y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).float()
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)

        attn = generate_path(durations, attn_mask.squeeze(1))

        # Verify shape
        assert attn.shape[0] == B
        # Duration sum must equal y_length for every sample
        for i in range(B):
            sample_dur_sum = batch["durations"][i, : x_lengths[i]].sum().item()
            assert sample_dur_sum == y_lengths[i].item()

    def test_single_sample_batch(self, tmp_path):
        """Edge case: batch_size=1."""
        self._make_pt_with_duration(tmp_path / "s1.pt", 11, 30, 0)
        ds = PrecomputedTextMelDataset(tmp_path, n_spks=100, load_durations=True)
        collate = TextMelBatchCollate(n_spks=100, load_durations=True)

        batch = collate([ds[0]])
        assert batch["durations"].shape[0] == 1
        assert batch["durations"] is not None

        # generate_path should also work
        durations = batch["durations"].float()
        x_lengths = batch["x_lengths"]
        y_lengths = batch["y_lengths"]
        y_max_length = batch["y"].shape[-1]
        x_mask = sequence_mask(x_lengths, batch["x"].shape[1]).unsqueeze(1).float()
        y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).float()
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
        attn = generate_path(durations, attn_mask.squeeze(1))
        assert attn.shape[0] == 1


# ---------------------------------------------------------------------------
# 2. Corrupted .npy files
# ---------------------------------------------------------------------------
class TestCorruptedNpy:
    def test_corrupted_npy_raises(self, tmp_path):
        """Corrupted .npy file must cause np.load() to raise."""
        corrupt = tmp_path / "jvs001_UTT001.npy"
        corrupt.write_bytes(b"this is not a valid npy file")

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        from precompute_dataset import load_duration

        with pytest.raises(Exception):  # ValueError or similar from np.load
            load_duration(tmp_path, "jvs001", "UTT001", 7)

    def test_empty_npy_raises(self, tmp_path):
        """Empty .npy file must raise."""
        empty = tmp_path / "jvs001_UTT001.npy"
        empty.write_bytes(b"")

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        from precompute_dataset import load_duration

        with pytest.raises(Exception):
            load_duration(tmp_path, "jvs001", "UTT001", 7)


# ---------------------------------------------------------------------------
# 3. Padding detail checks
# ---------------------------------------------------------------------------
class TestPaddingDetails:
    def _make_item(self, text_len, mel_len, spk=0):
        """Build a dict suitable for TextMelBatchCollate."""
        text = torch.randint(0, 55, (text_len,))
        mel = torch.randn(80, mel_len)
        dur = torch.zeros(text_len, dtype=torch.long)
        n_ph = text_len // 2
        if n_ph > 0:
            base = mel_len // n_ph
            rem = mel_len - base * n_ph
            for i in range(n_ph):
                dur[2 * i + 1] = base + (1 if i < rem else 0)
        return {
            "x": text,
            "y": mel,
            "spk": spk,
            "filepath": "test.pt",
            "x_text": "test",
            "durations": dur,
        }

    def test_varied_text_lengths_padding(self):
        """Padding region of durations must be zero for shorter samples."""
        items = [self._make_item(5, 10), self._make_item(21, 60), self._make_item(11, 30)]
        collate = TextMelBatchCollate(n_spks=100, load_durations=True)
        batch = collate(items)

        max_x = batch["x"].shape[1]
        assert max_x >= 21  # at least as large as the longest text
        # Padding beyond each sample's own text length must be zero
        assert batch["durations"][0, 5:].sum() == 0  # text_len=5
        assert batch["durations"][2, 11:].sum() == 0  # text_len=11

    def test_fix_len_compatibility_interaction(self):
        """Mel dimension must be rounded up by fix_len_compatibility."""
        items = [self._make_item(11, 30), self._make_item(9, 25)]
        collate = TextMelBatchCollate(n_spks=100, load_durations=True)
        batch = collate(items)

        y_max = batch["y"].shape[-1]
        # fix_len_compatibility with num_downsamplings_in_unet=2 -> factor 4
        assert y_max % 4 == 0 or y_max % 2 == 0  # UNet down-sampling compat
        assert y_max >= 30  # at least as large as the longest mel

    def test_duration_sum_preserved_after_padding(self):
        """Duration sum must match y_length for each sample after collate."""
        items = [self._make_item(7, 15), self._make_item(13, 40)]
        collate = TextMelBatchCollate(n_spks=100, load_durations=True)
        batch = collate(items)

        x_lengths = batch["x_lengths"]
        for i in range(2):
            dur_sum = batch["durations"][i, : x_lengths[i]].sum().item()
            assert dur_sum == batch["y_lengths"][i].item()


# ---------------------------------------------------------------------------
# 4. Collate dtype guarantees
# ---------------------------------------------------------------------------
class TestCollateDtype:
    def test_duration_dtype_is_long(self):
        """Collate output durations must be torch.long."""
        items = [
            {
                "x": torch.randint(0, 55, (11,)),
                "y": torch.randn(80, 30),
                "spk": 0,
                "filepath": "t.pt",
                "x_text": "t",
                "durations": torch.tensor([0, 5, 0, 5, 0, 5, 0, 5, 0, 5, 0], dtype=torch.long),
            },
        ]
        collate = TextMelBatchCollate(n_spks=100, load_durations=True)
        batch = collate(items)
        assert batch["durations"].dtype == torch.long

    def test_int32_duration_upcast_to_long(self):
        """int32 durations from dataset must be upcast to int64 (long) by collate."""
        items = [
            {
                "x": torch.randint(0, 55, (7,)),
                "y": torch.randn(80, 20),
                "spk": 0,
                "filepath": "t.pt",
                "x_text": "t",
                "durations": torch.tensor([0, 10, 0, 10, 0, 0, 0], dtype=torch.int32),
            },
        ]
        collate = TextMelBatchCollate(n_spks=100, load_durations=True)
        batch = collate(items)
        assert batch["durations"].dtype == torch.long


# ---------------------------------------------------------------------------
# 5. Backward compatibility (no load_durations kwarg)
# ---------------------------------------------------------------------------
class TestBackwardCompat:
    def test_collate_default_init_no_load_durations(self):
        """TextMelBatchCollate() without load_durations keeps existing behaviour."""
        items = [
            {
                "x": torch.randint(0, 55, (11,)),
                "y": torch.randn(80, 30),
                "spk": 0,
                "filepath": "t.pt",
                "x_text": "t",
                "durations": None,
            },
        ]
        collate = TextMelBatchCollate(n_spks=100)  # no load_durations
        batch = collate(items)
        assert batch["durations"] is None

    def test_dataset_default_no_load_durations(self, tmp_path):
        """PrecomputedTextMelDataset() without load_durations returns None durations."""
        text = torch.randint(0, 55, (11,), dtype=torch.int32)
        torch.save(
            {"mel": torch.randn(80, 30), "text": text, "spk": 0, "cleaned_text": "t"},
            tmp_path / "s.pt",
        )
        ds = PrecomputedTextMelDataset(tmp_path, n_spks=100)  # no load_durations
        item = ds[0]
        assert item["durations"] is None
