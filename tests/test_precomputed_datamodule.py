"""Tests for PrecomputedDataModule duration loading."""

import os
import random
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import SequentialSampler

from matcha.data.precomputed_datamodule import (
    BucketBatchSampler,
    DistributedBucketBatchSampler,
    PrecomputedTextMelDataModule,
    PrecomputedTextMelDataset,
)
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


def _make_pt_dir(dir_path: Path, n_files: int, include_durations: bool = False, base_mel_len: int = 20):
    """Helper: create a directory of dummy .pt files with distinct sizes (varying mel_len)."""
    dir_path.mkdir(parents=True, exist_ok=True)
    for i in range(n_files):
        _make_pt_file(
            dir_path / f"sample_{i:03d}.pt", 11, base_mel_len + i * 10, i, include_durations=include_durations
        )


def _make_datamodule(tmp_path: Path, load_durations: bool = False, batch_size: int = 2, seed: int = 42):
    """Helper: build a PrecomputedTextMelDataModule over synthetic train/val dirs and run setup()."""
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    _make_pt_dir(train_dir, 8, include_durations=load_durations)
    _make_pt_dir(val_dir, 8, include_durations=load_durations)
    dm = PrecomputedTextMelDataModule(
        name="test_precomputed",
        train_pt_dir=str(train_dir),
        val_pt_dir=str(val_dir),
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False,
        n_spks=100,
        n_feats=80,
        seed=seed,
        load_durations=load_durations,
        preload_to_memory=False,
        num_buckets=2,
    )
    dm.setup()
    return dm


class TestDataModuleSamplerSelection:
    """train_dataloader/val_dataloader sampler selection based on trainer device count."""

    def test_no_trainer_uses_bucket_batch_sampler(self, tmp_path):
        """Without an attached trainer, train_dataloader uses BucketBatchSampler with drop_last=True."""
        dm = _make_datamodule(tmp_path)
        dm.trainer = None
        loader = dm.train_dataloader()
        assert isinstance(loader.batch_sampler, BucketBatchSampler)
        assert not isinstance(loader.batch_sampler, DistributedBucketBatchSampler)
        assert loader.batch_sampler.drop_last is True
        assert loader.batch_sampler.batch_size == 2
        # batch_sampler mode: DataLoader's own batch_size is unset
        assert loader.batch_size is None

    def test_single_device_trainer_uses_bucket_batch_sampler(self, tmp_path):
        """A trainer with num_devices=1 selects the non-distributed BucketBatchSampler."""
        dm = _make_datamodule(tmp_path)
        dm.trainer = SimpleNamespace(num_devices=1, num_nodes=1, global_rank=0)
        loader = dm.train_dataloader()
        assert isinstance(loader.batch_sampler, BucketBatchSampler)
        assert not isinstance(loader.batch_sampler, DistributedBucketBatchSampler)

    def test_multi_device_trainer_uses_distributed_sampler(self, tmp_path):
        """A trainer with num_devices=2 selects DistributedBucketBatchSampler with matching replicas/rank."""
        dm = _make_datamodule(tmp_path)
        dm.trainer = SimpleNamespace(num_devices=2, num_nodes=1, global_rank=1)
        loader = dm.train_dataloader()
        sampler = loader.batch_sampler
        assert isinstance(sampler, DistributedBucketBatchSampler)
        assert sampler.num_replicas == 2
        assert sampler.rank == 1
        assert sampler.drop_last is True

    def test_multi_node_replica_count(self, tmp_path):
        """num_replicas is num_devices * num_nodes."""
        dm = _make_datamodule(tmp_path)
        dm.trainer = SimpleNamespace(num_devices=2, num_nodes=2, global_rank=3)
        loader = dm.train_dataloader()
        sampler = loader.batch_sampler
        assert isinstance(sampler, DistributedBucketBatchSampler)
        assert sampler.num_replicas == 4
        assert sampler.rank == 3

    @pytest.mark.parametrize("load_durations", [False, True])
    def test_collate_fn_mirrors_load_durations(self, tmp_path, load_durations):
        """Both dataloaders' collate_fn.load_durations mirrors the hparams value."""
        dm = _make_datamodule(tmp_path, load_durations=load_durations)
        dm.trainer = None
        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader()
        assert isinstance(train_loader.collate_fn, TextMelBatchCollate)
        assert isinstance(val_loader.collate_fn, TextMelBatchCollate)
        assert train_loader.collate_fn.load_durations is load_durations
        assert val_loader.collate_fn.load_durations is load_durations

    def test_val_dataloader_plain_batching(self, tmp_path):
        """val_dataloader uses plain sequential batching: no bucket sampler, no shuffle, no drop_last."""
        dm = _make_datamodule(tmp_path, batch_size=2)
        loader = dm.val_dataloader()
        assert not isinstance(loader.batch_sampler, (BucketBatchSampler, DistributedBucketBatchSampler))
        assert isinstance(loader.sampler, SequentialSampler)  # shuffle=False
        assert loader.batch_size == 2
        assert loader.drop_last is False


class TestScanAndSeededShuffle:
    """PrecomputedTextMelDataset.__init__ directory scan and seeded shuffle."""

    def test_scan_picks_only_top_level_pt_files(self, tmp_path):
        """Only regular top-level .pt files are collected; other entries are ignored."""
        for i in range(10):
            _make_pt_file(tmp_path / f"s{i:02d}.pt", 11, 20 + i * 5, 0)
        (tmp_path / "stats.json").write_text("{}", encoding="utf-8")
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        _make_pt_file(subdir / "nested.pt", 11, 30, 0)  # nested .pt must not be picked up
        (tmp_path / "fake.pt").mkdir()  # directory with .pt suffix must be skipped

        ds = PrecomputedTextMelDataset(tmp_path, n_spks=100, seed=42)
        names = sorted(os.path.basename(p) for p in ds.pt_paths)
        assert names == [f"s{i:02d}.pt" for i in range(10)]

    def test_same_seed_gives_identical_order(self, tmp_path):
        """Two datasets over the same dir with the same seed have identical pt_paths order."""
        for i in range(10):
            _make_pt_file(tmp_path / f"s{i:02d}.pt", 11, 20 + i * 5, 0)
        ds1 = PrecomputedTextMelDataset(tmp_path, n_spks=100, seed=42)
        ds2 = PrecomputedTextMelDataset(tmp_path, n_spks=100, seed=42)
        assert ds1.pt_paths == ds2.pt_paths

    def test_different_seed_gives_different_order(self, tmp_path):
        """Different seeds shuffle pt_paths into different orders (same set of files)."""
        for i in range(10):
            _make_pt_file(tmp_path / f"s{i:02d}.pt", 11, 20 + i * 5, 0)
        ds1 = PrecomputedTextMelDataset(tmp_path, n_spks=100, seed=42)
        ds2 = PrecomputedTextMelDataset(tmp_path, n_spks=100, seed=43)
        assert sorted(ds1.pt_paths) == sorted(ds2.pt_paths)
        assert ds1.pt_paths != ds2.pt_paths

    def test_global_random_state_untouched(self, tmp_path):
        """The seeded shuffle must use a local RNG, not reseed the global random module."""
        for i in range(6):
            _make_pt_file(tmp_path / f"s{i:02d}.pt", 11, 20 + i * 5, 0)
        random.seed(999)
        state_before = random.getstate()
        PrecomputedTextMelDataset(tmp_path, n_spks=100, seed=42)
        assert random.getstate() == state_before


class TestGetFileSizes:
    """get_file_sizes index alignment (post-shuffle) and caching."""

    def test_sizes_align_with_shuffled_pt_paths(self, tmp_path):
        """get_file_sizes()[i] corresponds to pt_paths[i] after the seeded shuffle."""
        _make_pt_dir(tmp_path, 8)  # distinct mel_len => distinct file sizes
        ds = PrecomputedTextMelDataset(tmp_path, n_spks=100, seed=7)
        sizes = ds.get_file_sizes()
        assert len(sizes) == len(ds.pt_paths) == 8
        # Sizes must be distinct so the alignment check below is meaningful
        assert len(set(sizes)) == len(sizes)
        for i, path in enumerate(ds.pt_paths):
            assert sizes[i] == os.path.getsize(path)

    def test_second_call_returns_cached_list(self, tmp_path, monkeypatch):
        """The second call returns the cached list without re-stat-ing files."""
        _make_pt_dir(tmp_path, 4)
        ds = PrecomputedTextMelDataset(tmp_path, n_spks=100, seed=7)
        first = ds.get_file_sizes()

        def _fail(_path):
            raise AssertionError("os.path.getsize must not be called on a cached get_file_sizes()")

        monkeypatch.setattr(os.path, "getsize", _fail)
        second = ds.get_file_sizes()
        assert second is first


class TestPreloadServing:
    """preload_to_memory=True serves items from the in-memory cache."""

    @pytest.mark.parametrize("load_durations", [False, True])
    def test_items_served_after_files_deleted(self, tmp_path, load_durations):
        """With preload, every item is still served after all .pt files are removed from disk."""
        _make_pt_dir(tmp_path, 4, include_durations=load_durations)
        ds = PrecomputedTextMelDataset(
            tmp_path, n_spks=100, seed=1, preload_to_memory=True, load_durations=load_durations
        )
        for path in ds.pt_paths:
            os.remove(path)

        assert len(ds) == 4
        for i in range(len(ds)):
            item = ds[i]
            assert item["y"].shape[0] == 80
            assert item["x"].shape[0] == 11
            assert item["x_text"] == "dummy text"
            if load_durations:
                assert item["durations"] is not None
                assert len(item["durations"]) == 11
            else:
                assert item["durations"] is None

    def test_without_preload_deleted_files_raise(self, tmp_path):
        """Contrast: without preload, __getitem__ reads from disk and fails once files are gone."""
        _make_pt_dir(tmp_path, 2)
        ds = PrecomputedTextMelDataset(tmp_path, n_spks=100, seed=1, preload_to_memory=False)
        for path in ds.pt_paths:
            os.remove(path)
        with pytest.raises((FileNotFoundError, OSError, RuntimeError)):
            ds[0]

    def test_mutating_returned_item_does_not_corrupt_cache(self, tmp_path):
        """__getitem__ must return a copy of the cached dict: in-place mutation of a
        returned item must not affect the next access to the same index."""
        _make_pt_dir(tmp_path, 2)
        ds = PrecomputedTextMelDataset(tmp_path, n_spks=100, seed=1, preload_to_memory=True)

        item = ds[0]
        original_x = item["x"]
        item["x"] = None
        item.pop("y")

        fresh = ds[0]
        assert fresh is not item
        assert fresh["x"] is not None
        assert torch.equal(fresh["x"], original_x)
        assert "y" in fresh


class TestSetupIdempotent:
    """setup() must not rebuild the datasets when called for a second stage
    (Lightning calls it for fit AND validate/test), otherwise preload_to_memory
    re-loads everything into RAM twice."""

    def _datamodule(self, tmp_path):
        train_dir = tmp_path / "train"
        val_dir = tmp_path / "val"
        _make_pt_dir(train_dir, 4)
        _make_pt_dir(val_dir, 4)
        return PrecomputedTextMelDataModule(
            name="test_precomputed",
            train_pt_dir=str(train_dir),
            val_pt_dir=str(val_dir),
            batch_size=2,
            num_workers=0,
            pin_memory=False,
            n_spks=100,
            n_feats=80,
            seed=42,
            load_durations=False,
            preload_to_memory=True,
            num_buckets=2,
        )

    def test_second_setup_does_not_reload(self, tmp_path, monkeypatch):
        dm = self._datamodule(tmp_path)

        calls = {"n": 0}
        real_load = torch.load

        def counting_load(*args, **kwargs):
            calls["n"] += 1
            return real_load(*args, **kwargs)

        monkeypatch.setattr(torch, "load", counting_load)

        dm.setup("fit")
        first_pass_loads = calls["n"]
        assert first_pass_loads == 8  # 4 train + 4 val preloaded once
        trainset, validset = dm.trainset, dm.validset

        dm.setup("validate")
        assert calls["n"] == first_pass_loads  # no re-load
        assert dm.trainset is trainset  # same dataset objects kept
        assert dm.validset is validset


class TestDistributedSamplerEqualBatchCounts:
    """NCCL-deadlock guard: every rank must yield exactly the same number of batches."""

    @pytest.mark.parametrize(
        "dataset_size,num_replicas",
        [(10, 2), (11, 2), (10, 3), (11, 3)],
    )
    def test_equal_batch_counts_across_ranks_and_epochs(self, dataset_size, num_replicas):
        """With drop_last=True, all ranks yield identical batch counts at epochs 0, 1 and 2."""
        rng = random.Random(0)
        file_sizes = [rng.randint(5000, 50000) for _ in range(dataset_size)]
        samplers = [
            DistributedBucketBatchSampler(
                file_sizes,
                batch_size=2,
                num_replicas=num_replicas,
                rank=rank,
                num_buckets=2,
                drop_last=True,
                seed=0,
            )
            for rank in range(num_replicas)
        ]
        for epoch in (0, 1, 2):
            counts = []
            for sampler in samplers:
                sampler.set_epoch(epoch)
                batches = list(sampler)
                assert all(len(b) == 2 for b in batches)
                counts.append(len(batches))
            assert len(set(counts)) == 1, f"Unequal batch counts across ranks at epoch {epoch}: {counts}"
            # Sanity: at least one full batch per rank so the check is non-trivial
            assert counts[0] >= 1


class TestBucketSamplerEpochAdvance:
    """BucketBatchSampler auto-advances its epoch as a fallback when nothing calls
    set_epoch between iterations.  ``epoch`` reflects the epoch used by the most
    recent iteration (the advance is applied lazily at the start of the next one)."""

    @staticmethod
    def _file_sizes(n=64, seed=42):
        rng = random.Random(seed)
        return [rng.randint(5000, 50000) for _ in range(n)]

    def test_second_iteration_differs_from_first(self):
        """Iterating the same sampler twice yields different batch orders (epoch auto-advances)."""
        file_sizes = self._file_sizes()
        sampler = BucketBatchSampler(file_sizes, batch_size=4, num_buckets=4, drop_last=False, seed=123)
        assert sampler.epoch == 0
        first = [tuple(b) for b in sampler]
        assert sampler.epoch == 0  # epoch used by the iteration just taken
        second = [tuple(b) for b in sampler]
        assert sampler.epoch == 1  # fallback advance applied on the second iteration
        assert first != second
        # Both epochs cover the same sample set
        assert sorted(i for b in first for i in b) == sorted(i for b in second for i in b)

    def test_fresh_sampler_reproduces_first_epoch(self):
        """A fresh sampler with the same seed reproduces the first iteration exactly (resume determinism)."""
        file_sizes = self._file_sizes()
        sampler_a = BucketBatchSampler(file_sizes, batch_size=4, num_buckets=4, drop_last=False, seed=123)
        first_a = [tuple(b) for b in sampler_a]

        sampler_b = BucketBatchSampler(file_sizes, batch_size=4, num_buckets=4, drop_last=False, seed=123)
        first_b = [tuple(b) for b in sampler_b]
        assert first_a == first_b
