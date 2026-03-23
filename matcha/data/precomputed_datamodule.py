import logging
import math
import os
import random
from pathlib import Path
from typing import Any

import torch
from lightning import LightningDataModule
from torch.utils.data import Dataset, Sampler
from torch.utils.data.dataloader import DataLoader

from matcha.data.text_mel_datamodule import TextMelBatchCollate

log = logging.getLogger(__name__)


class BucketBatchSampler(Sampler):
    """Batch sampler that groups samples by mel length (approximated via file size)
    to minimize padding waste within each batch.

    Samples are sorted by file size into buckets, then batches are drawn from
    within each bucket.  Bucket order is shuffled each epoch so that training
    remains stochastic while individual batches contain similarly-sized items.
    """

    def __init__(
        self, file_sizes: list[int], batch_size: int, num_buckets: int = 10, drop_last: bool = False, seed: int = 0
    ):
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

        # Sort indices by file size (proxy for mel length)
        sorted_indices = sorted(range(len(file_sizes)), key=lambda i: file_sizes[i])

        # Split sorted indices into roughly equal-sized buckets
        bucket_size = max(1, len(sorted_indices) // num_buckets)
        self.buckets: list[list[int]] = []
        for start in range(0, len(sorted_indices), bucket_size):
            bucket = sorted_indices[start : start + bucket_size]
            if bucket:
                self.buckets.append(bucket)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1

        # Build batches within each bucket, then shuffle bucket order
        all_batches = []
        for bucket in self.buckets:
            indices = list(bucket)
            rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                all_batches.append(batch)

        rng.shuffle(all_batches)
        yield from all_batches

    def __len__(self):
        total = sum(len(b) for b in self.buckets)
        if self.drop_last:
            return total // self.batch_size
        return (total + self.batch_size - 1) // self.batch_size


class DistributedBucketBatchSampler(Sampler):
    """Distributed batch sampler with bucket-based length grouping.

    Combines DistributedSampler's index partitioning with BucketBatchSampler's
    length-aware batching.  Each rank receives a disjoint subset of dataset
    indices (exactly as ``DistributedSampler`` would assign), then those
    rank-local indices are sorted by file size and grouped into buckets so that
    batches contain similarly-sized items, reducing padding waste.

    The deterministic shuffling logic mirrors ``torch.utils.data.DistributedSampler``
    so that all ranks agree on the global permutation before splitting.
    """

    def __init__(
        self,
        file_sizes: list[int],
        batch_size: int,
        num_replicas: int,
        rank: int,
        num_buckets: int = 10,
        drop_last: bool = False,
        seed: int = 0,
    ):
        self.file_sizes = file_sizes
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_buckets = num_buckets
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

        self.total_size = len(file_sizes)
        # Pad total to be evenly divisible by num_replicas (same as DistributedSampler)
        self.num_samples = math.ceil(self.total_size / self.num_replicas)
        self.padded_total = self.num_samples * self.num_replicas

    def __iter__(self):
        # --- 1. Deterministic global shuffle (identical across all ranks) ---
        rng = random.Random(self.seed + self.epoch)
        indices = list(range(self.total_size))
        rng.shuffle(indices)

        # Pad to make evenly divisible (wrap-around, same as DistributedSampler)
        padding_size = self.padded_total - len(indices)
        if padding_size > 0:
            indices += indices[:padding_size]

        # --- 2. Subsample for this rank ---
        rank_indices = indices[self.rank : self.padded_total : self.num_replicas]
        assert len(rank_indices) == self.num_samples

        # --- 3. Bucket sort rank-local indices by file size ---
        rank_indices_sorted = sorted(rank_indices, key=lambda i: self.file_sizes[i % self.total_size])

        bucket_size = max(1, len(rank_indices_sorted) // self.num_buckets)
        buckets: list[list[int]] = []
        for start in range(0, len(rank_indices_sorted), bucket_size):
            bucket = rank_indices_sorted[start : start + bucket_size]
            if bucket:
                buckets.append(bucket)

        # --- 4. Build batches within buckets, then shuffle batch order ---
        # Use rank-specific RNG for intra-bucket shuffle so each rank sees
        # a different batch ordering while keeping the global split deterministic.
        epoch_rng = random.Random(self.seed + self.epoch + self.rank)
        self.epoch += 1

        all_batches = []
        for bucket in buckets:
            items = list(bucket)
            epoch_rng.shuffle(items)
            for bstart in range(0, len(items), self.batch_size):
                batch = items[bstart : bstart + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                all_batches.append(batch)

        epoch_rng.shuffle(all_batches)
        yield from all_batches

    def __len__(self):
        if self.drop_last:
            return self.num_samples // self.batch_size
        return (self.num_samples + self.batch_size - 1) // self.batch_size

    def set_epoch(self, epoch: int):
        """Set the epoch for deterministic shuffling (called by Lightning)."""
        self.epoch = epoch


class PrecomputedTextMelDataset(Dataset):
    """Dataset for pre-computed .pt files containing mel spectrograms and text sequences.

    Each .pt file is expected to contain a dict with keys:
        - "mel": Tensor of shape (n_feats, mel_length)
        - "text": IntTensor of phoneme indices
        - "spk": int speaker id
        - "cleaned_text": str
    """

    def __init__(self, pt_dir, n_spks, seed=None, preload_to_memory=False):
        self.pt_dir = Path(pt_dir)
        self.n_spks = n_spks
        self.pt_paths = sorted(
            os.path.join(str(self.pt_dir), entry.name)
            for entry in os.scandir(str(self.pt_dir))
            if entry.name.endswith(".pt") and entry.is_file()
        )

        random.seed(seed)
        random.shuffle(self.pt_paths)

        self._cache: dict[int, dict] = {}
        if preload_to_memory:
            self._preload()

    def _preload(self):
        """Load all .pt files into memory to eliminate NFS I/O during training."""
        log.info("Preloading %d samples to memory...", len(self))
        for i in range(len(self)):
            self._cache[i] = self._load_from_disk(i)
        log.info("Preload complete.")

    def _load_from_disk(self, index):
        """Load a single sample from disk and return as a dict."""
        pt_path = self.pt_paths[index]
        data = torch.load(pt_path, weights_only=True)

        mel = data["mel"]
        text = data["text"]
        spk = data["spk"] if self.n_spks > 1 else None
        cleaned_text = data["cleaned_text"]

        return {
            "x": text,
            "y": mel,
            "spk": spk,
            "filepath": str(pt_path),
            "x_text": cleaned_text,
            "durations": None,
        }

    def get_file_sizes(self) -> list[int]:
        """Return file sizes for all .pt files (proxy for mel length)."""
        if not hasattr(self, "_file_sizes_cache") or self._file_sizes_cache is None:
            self._file_sizes_cache = [os.path.getsize(p) for p in self.pt_paths]
        return self._file_sizes_cache

    def __len__(self):
        return len(self.pt_paths)

    def __getitem__(self, index):
        if index in self._cache:
            return self._cache[index]
        return self._load_from_disk(index)


class PrecomputedTextMelDataModule(LightningDataModule):
    def __init__(  # pylint: disable=unused-argument
        self,
        name,
        train_pt_dir,
        val_pt_dir,
        batch_size,
        num_workers,
        pin_memory,
        n_spks,
        n_feats,
        seed,
        data_statistics=None,
        load_durations=False,
        preload_to_memory=False,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

    def setup(self, stage: str | None = None):  # pylint: disable=unused-argument
        self.trainset = PrecomputedTextMelDataset(  # pylint: disable=attribute-defined-outside-init
            self.hparams.train_pt_dir,
            self.hparams.n_spks,
            self.hparams.seed,
            preload_to_memory=self.hparams.preload_to_memory,
        )
        self.validset = PrecomputedTextMelDataset(  # pylint: disable=attribute-defined-outside-init
            self.hparams.val_pt_dir,
            self.hparams.n_spks,
            self.hparams.seed,
            preload_to_memory=self.hparams.preload_to_memory,
        )

    def train_dataloader(self):
        trainer = self.trainer
        is_distributed = trainer is not None and getattr(trainer, "num_devices", 1) > 1

        file_sizes = self.trainset.get_file_sizes()
        seed = self.hparams.seed or 0

        if is_distributed:
            # Use DistributedBucketBatchSampler to combine DDP index
            # partitioning with length-aware batching, avoiding the
            # 20-40% padding waste of plain DistributedSampler + random batches.
            num_replicas = trainer.num_devices * getattr(trainer, "num_nodes", 1)
            rank = trainer.global_rank
            bucket_sampler = DistributedBucketBatchSampler(
                file_sizes=file_sizes,
                batch_size=self.hparams.batch_size,
                num_replicas=num_replicas,
                rank=rank,
                drop_last=True,
                seed=seed,
            )
        else:
            bucket_sampler = BucketBatchSampler(
                file_sizes=file_sizes,
                batch_size=self.hparams.batch_size,
                drop_last=True,
                seed=seed,
            )

        nw = self.hparams.num_workers
        return DataLoader(
            dataset=self.trainset,
            batch_sampler=bucket_sampler,
            num_workers=nw,
            pin_memory=self.hparams.pin_memory,
            collate_fn=TextMelBatchCollate(self.hparams.n_spks),
            persistent_workers=nw > 0,
            prefetch_factor=8 if nw > 0 else None,
        )

    def val_dataloader(self):
        nw = self.hparams.num_workers
        return DataLoader(
            dataset=self.validset,
            batch_size=self.hparams.batch_size,
            num_workers=nw,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=TextMelBatchCollate(self.hparams.n_spks),
            persistent_workers=nw > 0,
            prefetch_factor=8 if nw > 0 else None,
        )

    def teardown(self, stage: str | None = None):
        """Clean up after fit or test."""
        pass  # pylint: disable=unnecessary-pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass  # pylint: disable=unnecessary-pass
