"""WaveNeXt fmax=11025 GAN vocoder training entry point.

Two-stage recipe (see docs/wavenext-training-tdd-plan.md):
  stage 1  base on MoeSpeech    (BSC generator init + scratch discriminators, pretrain_mel_steps=N)
  stage 2  fine-tune on Tsukuyomi (pretrain_mel_steps=0)
After training, extract the deployable generator with scripts/extract_wavenext_generator.py.

Usage:
    uv run python wavenext_train/train.py --config configs_wavenext/wavenext_11025.yaml \
        --train-filelist data/moespeech/wavenext_train.txt --val-filelist data/moespeech/wavenext_val.txt \
        [--init-state init.bin] [--ckpt-path last.ckpt]
"""

import argparse

import lightning as L
import torch
from omegaconf import OmegaConf

from matcha.wavenext.models import VocosBackbone, WaveNextHead
from wavenext_train.dataset import DataConfig, VocosDataModule
from wavenext_train.experiment import WaveNeXtExp
from wavenext_train.features import MatchaMelFeatures
from wavenext_train.loss import MelSpecReconstructionLoss


def load_config(path):
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def build_model(model_cfg):
    feature_extractor = MatchaMelFeatures(**model_cfg["feature_extractor"])
    backbone = VocosBackbone(**model_cfg["backbone"])
    head = WaveNextHead(**model_cfg["head"])
    melspec_loss = MelSpecReconstructionLoss(fmax=model_cfg["melspec_loss"]["fmax"])
    return WaveNeXtExp(feature_extractor, backbone, head, melspec_loss, **model_cfg.get("exp", {}))


def build_datamodule(data_cfg, train_filelist, val_filelist):
    def _cfg(filelist, num_samples):
        return DataConfig(
            filelist_path=filelist,
            sampling_rate=data_cfg["sampling_rate"],
            num_samples=num_samples,
            batch_size=data_cfg["batch_size"],
            num_workers=data_cfg["num_workers"],
        )

    return VocosDataModule(
        _cfg(train_filelist, data_cfg["num_samples"]),
        _cfg(val_filelist, data_cfg.get("val_num_samples", data_cfg["num_samples"])),
    )


def main(argv=None):
    p = argparse.ArgumentParser(description="WaveNeXt fmax=11025 GAN training")
    p.add_argument("--config", required=True)
    p.add_argument("--train-filelist", required=True)
    p.add_argument("--val-filelist", required=True)
    p.add_argument("--init-state", default=None, help="backbone./head. init (e.g. from BSC weights)")
    p.add_argument("--ckpt-path", default=None, help="resume from a training .ckpt")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    model = build_model(cfg["model"])
    if args.init_state:
        sd = torch.load(args.init_state, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        gen_missing = [k for k in missing if k.startswith(("backbone.", "head."))]
        assert not gen_missing, f"init-state missing generator keys: {gen_missing}"
        print(f"[+] init generator from {args.init_state} (unexpected ignored: {len(unexpected)})")
    dm = build_datamodule(cfg["data"], args.train_filelist, args.val_filelist)

    tr = dict(cfg.get("trainer", {}))
    # manual optimization clips inside the module; never pass gradient_clip_val to the Trainer.
    tr.pop("gradient_clip_val", None)
    trainer = L.Trainer(**tr)
    trainer.fit(model, dm, ckpt_path=args.ckpt_path)


if __name__ == "__main__":
    main()
