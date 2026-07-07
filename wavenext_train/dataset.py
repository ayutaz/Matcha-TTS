"""WaveNeXt GAN training waveform dataset.

Ported from wetdog/wavenext_pytorch vocos/dataset.py. MIT License, Copyright (c) 2023
Charactr Inc. Intentional divergences from upstream (NOT transcription drift):
  - The torchaudio backend (which applied gain via the removed sox path) is gone in
    torchaudio>=2.9, so volume is set with a pure-torch peak normalize.
  - torchaudio.load crashes on torch>=2.10 (torchcodec dependency), so audio is read
    with soundfile (repo convention).
  - pytorch_lightning -> lightning (first-party dependency).
"""

from dataclasses import dataclass

import numpy as np
import soundfile as sf
import torch
import torchaudio
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

torch.set_num_threads(1)


@dataclass
class DataConfig:
    filelist_path: str
    sampling_rate: int
    num_samples: int
    batch_size: int
    num_workers: int


def load_audio(path):
    data, sr = sf.read(path, dtype="float32", always_2d=True)  # (T, C)
    return torch.from_numpy(data.T).contiguous(), int(sr)  # (C, T)


def peak_normalize(y, gain_db):
    target = 10.0 ** (gain_db / 20.0)
    peak = y.abs().max()
    return y * (target / peak) if peak > 0 else y


class VocosDataset(Dataset):
    def __init__(self, cfg, train):
        with open(cfg.filelist_path) as f:
            self.filelist = f.read().splitlines()
        self.sampling_rate = cfg.sampling_rate
        self.num_samples = cfg.num_samples
        self.train = train

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        y, sr = load_audio(self.filelist[index])
        if y.size(0) > 1:
            y = y.mean(dim=0, keepdim=True)
        gain = np.random.uniform(-1, -6) if self.train else -3
        y = peak_normalize(y, float(gain))
        if sr != self.sampling_rate:
            y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=self.sampling_rate)
        if y.size(-1) < self.num_samples:
            pad_length = self.num_samples - y.size(-1)
            padding_tensor = y.repeat(1, 1 + pad_length // y.size(-1))
            y = torch.cat((y, padding_tensor[:, :pad_length]), dim=1)
        elif self.train:
            start = np.random.randint(low=0, high=y.size(-1) - self.num_samples + 1)
            y = y[:, start : start + self.num_samples]
        else:
            y = y[:, : self.num_samples]
        return y[0]


class VocosDataModule(LightningDataModule):
    def __init__(self, train_params, val_params):
        super().__init__()
        self.train_config = train_params
        self.val_config = val_params

    def _get_dataloder(self, cfg, train):  # spelling kept verbatim from upstream
        return DataLoader(
            VocosDataset(cfg, train=train),
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            shuffle=train,
            pin_memory=True,
        )

    def train_dataloader(self):
        return self._get_dataloder(self.train_config, train=True)

    def val_dataloader(self):
        return self._get_dataloder(self.val_config, train=False)
