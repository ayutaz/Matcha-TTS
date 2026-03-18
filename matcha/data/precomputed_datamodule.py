import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from lightning import LightningDataModule
from torch.utils.data import Dataset, Sampler
from torch.utils.data.dataloader import DataLoader

from matcha.data.text_mel_datamodule import TextMelBatchCollate


class BucketBatchSampler(Sampler):
    """Batch sampler that groups samples by mel length (approximated via file size)
    to minimize padding waste within each batch.

    Samples are sorted by file size into buckets, then batches are drawn from
    within each bucket.  Bucket order is shuffled each epoch so that training
    remains stochastic while individual batches contain similarly-sized items.
    """

    def __init__(self, file_sizes: List[int], batch_size: int, num_buckets: int = 10, drop_last: bool = False, seed: int = 0):
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

        # Sort indices by file size (proxy for mel length)
        sorted_indices = sorted(range(len(file_sizes)), key=lambda i: file_sizes[i])

        # Split sorted indices into roughly equal-sized buckets
        bucket_size = max(1, len(sorted_indices) // num_buckets)
        self.buckets: List[List[int]] = []
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


class PrecomputedTextMelDataset(Dataset):
    """Dataset for pre-computed .pt files containing mel spectrograms and text sequences.

    Each .pt file is expected to contain a dict with keys:
        - "mel": Tensor of shape (n_feats, mel_length)
        - "text": IntTensor of phoneme indices
        - "spk": int speaker id
        - "cleaned_text": str
    """

    def __init__(self, pt_dir, n_spks, seed=None):
        self.pt_dir = Path(pt_dir)
        self.n_spks = n_spks
        self.pt_paths = sorted(
            os.path.join(str(self.pt_dir), entry.name)
            for entry in os.scandir(str(self.pt_dir))
            if entry.name.endswith(".pt") and entry.is_file()
        )

        random.seed(seed)
        random.shuffle(self.pt_paths)

    def get_file_sizes(self) -> List[int]:
        """Return file sizes for all .pt files (proxy for mel length)."""
        if not hasattr(self, '_file_sizes_cache') or self._file_sizes_cache is None:
            self._file_sizes_cache = [os.path.getsize(p) for p in self.pt_paths]
        return self._file_sizes_cache

    def __len__(self):
        return len(self.pt_paths)

    def __getitem__(self, index):
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
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

    def setup(self, stage: Optional[str] = None):  # pylint: disable=unused-argument
        self.trainset = PrecomputedTextMelDataset(  # pylint: disable=attribute-defined-outside-init
            self.hparams.train_pt_dir,
            self.hparams.n_spks,
            self.hparams.seed,
        )
        self.validset = PrecomputedTextMelDataset(  # pylint: disable=attribute-defined-outside-init
            self.hparams.val_pt_dir,
            self.hparams.n_spks,
            self.hparams.seed,
        )

    def train_dataloader(self):
        bucket_sampler = BucketBatchSampler(
            file_sizes=self.trainset.get_file_sizes(),
            batch_size=self.hparams.batch_size,
            drop_last=True,
            seed=self.hparams.seed or 0,
        )

        return DataLoader(
            dataset=self.trainset,
            batch_sampler=bucket_sampler,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=TextMelBatchCollate(self.hparams.n_spks),
            persistent_workers=True,
            prefetch_factor=8,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.validset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=TextMelBatchCollate(self.hparams.n_spks),
            persistent_workers=True,
            prefetch_factor=8,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass  # pylint: disable=unnecessary-pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass  # pylint: disable=unnecessary-pass
