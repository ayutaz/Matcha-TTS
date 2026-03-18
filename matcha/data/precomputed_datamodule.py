import random
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from lightning import LightningDataModule
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from matcha.data.text_mel_datamodule import TextMelBatchCollate


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
        self.pt_paths = sorted(self.pt_dir.glob("*.pt"))

        random.seed(seed)
        random.shuffle(self.pt_paths)

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
        return DataLoader(
            dataset=self.trainset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=TextMelBatchCollate(self.hparams.n_spks),
            persistent_workers=True,
            prefetch_factor=4,
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
            prefetch_factor=4,
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
