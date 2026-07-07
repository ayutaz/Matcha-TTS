"""TDD tests for wavenext_train config/scripts/E2E: init, extract (round-trips load_wavenext),
filelist, config->model build, a 2-batch manual-opt fit, and non-destructiveness of the
existing matcha/wavenext inference package.
"""

import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from extract_wavenext_generator import extract  # noqa: E402
from init_wavenext_from_bsc import build_init_state_dict  # noqa: E402
from make_wavenext_filelist import make_filelist  # noqa: E402

from matcha.cli import load_wavenext  # noqa: E402
from matcha.wavenext import WaveNeXtVocoder  # noqa: E402


def test_init_from_bsc_keeps_only_backbone_head():
    bsc = {"backbone.embed.weight": torch.randn(4), "head.linear_1.weight": torch.randn(4),
           "feature_extractor.mel_spec.weight": torch.randn(4)}
    out = build_init_state_dict(bsc)
    assert all(k.startswith(("backbone.", "head.")) for k in out)
    assert not any(k.startswith("feature_extractor.") for k in out)
    assert len(out) == 2


def test_extract_generator_roundtrips_load_wavenext(tmp_path):
    gen = WaveNeXtVocoder().state_dict()
    mixed = OrderedDict(gen)
    mixed["multiperioddisc.discriminators.0.convs.0.weight_v"] = torch.randn(4)
    mixed["multiresddisc.discriminators.0.conv_post.weight_v"] = torch.randn(4)
    mixed["n_batches"] = torch.zeros(1, dtype=torch.long)
    ckpt_path = tmp_path / "train.ckpt"
    torch.save({"state_dict": mixed}, ckpt_path)
    out_sd = extract(torch.load(ckpt_path, weights_only=False))
    assert set(out_sd.keys()) == set(gen.keys())
    out_bin = tmp_path / "gen.bin"
    torch.save(out_sd, out_bin)
    voc = load_wavenext(str(out_bin), "cpu")  # must not raise (missing/unexpected asserts inside)
    assert voc is not None


def test_extract_prefers_ema_weights():
    gen = WaveNeXtVocoder().state_dict()
    ema = OrderedDict((k, torch.zeros_like(v)) for k, v in gen.items())  # distinct values
    ckpt = {"state_dict": OrderedDict(gen), "ema_state_dict": ema}
    out = extract(ckpt, prefer_ema=True)
    assert all(torch.equal(out[k], ema[k]) for k in out)
    out2 = extract({"state_dict": OrderedDict(gen)}, prefer_ema=True)  # EMA absent -> fallback
    assert all(torch.equal(out2[k], gen[k]) for k in out2)


def test_make_filelist_enumerates_all_wavs(tmp_path):
    (tmp_path / "sub").mkdir()
    wavs = []
    for p in ["a.wav", "sub/b.wav", "sub/c.wav"]:
        fp = tmp_path / p
        sf.write(str(fp), np.zeros(1000, dtype=np.float32), 22050)
        wavs.append(fp)
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    fl = make_filelist(str(tmp_path))
    assert len(fl) == 3
    assert fl == sorted(fl)
    assert all(f.endswith(".wav") and Path(f).exists() for f in fl)


def test_config_loads_and_builds_model_cpu():
    from wavenext_train.experiment import WaveNeXtExp
    from wavenext_train.features import MatchaMelFeatures
    from wavenext_train.train import build_model, load_config

    cfg = load_config("configs_wavenext/wavenext_11025.yaml")
    model = build_model(cfg["model"])
    assert isinstance(model, WaveNeXtExp)
    assert isinstance(model.feature_extractor, MatchaMelFeatures) and model.feature_extractor.fmax == 11025
    assert model.automatic_optimization is False
    with torch.no_grad():
        out = model.generator_forward(torch.zeros(1, 16384))
    assert out.shape == (1, 16384)
    assert torch.isfinite(model.melspec_loss(out, torch.zeros(1, 16384))).all()


def test_fit_two_batches_manual_opt_cpu(tmp_path):
    import lightning as L

    from matcha.wavenext.models import VocosBackbone, WaveNextHead
    from wavenext_train.dataset import DataConfig, VocosDataModule
    from wavenext_train.experiment import WaveNeXtExp
    from wavenext_train.features import MatchaMelFeatures
    from wavenext_train.loss import MelSpecReconstructionLoss

    wavs = [str(tmp_path / f"w{i}.wav") for i in range(4)]
    for w in wavs:
        sf.write(w, np.random.uniform(-1, 1, 8000).astype(np.float32), 22050)
    fl = tmp_path / "fl.txt"
    fl.write_text("\n".join(wavs) + "\n", encoding="utf-8")
    cfg = DataConfig(filelist_path=str(fl), sampling_rate=22050, num_samples=4096, batch_size=2, num_workers=0)
    dm = VocosDataModule(cfg, cfg)
    model = WaveNeXtExp(
        MatchaMelFeatures(fmax=11025),
        VocosBackbone(input_channels=80, dim=32, intermediate_dim=64, num_layers=1),
        WaveNextHead(dim=32, n_fft=1024, hop_length=256),
        MelSpecReconstructionLoss(fmax=11025),
        pretrain_mel_steps=0, max_steps=2, num_warmup_steps=1,
    )
    L.Trainer(accelerator="cpu", precision="32-true", max_epochs=1, limit_val_batches=0,
              logger=False, enable_checkpointing=False, enable_progress_bar=False,
              enable_model_summary=False).fit(model, dm)
    assert int(model.n_batches.item()) >= 1


def test_build_logger_returns_requested_type(tmp_path):
    from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

    from wavenext_train.train import build_logger

    assert isinstance(build_logger(None, str(tmp_path)), TensorBoardLogger)
    assert isinstance(build_logger({"type": "tensorboard"}, str(tmp_path)), TensorBoardLogger)
    wl = build_logger({"type": "wandb", "project": "matcha-tts-ja", "name": "t", "offline": True}, str(tmp_path))
    assert isinstance(wl, WandbLogger)


def test_existing_wavenext_suite_untouched():
    voc = WaveNeXtVocoder()
    n = sum(p.numel() for p in voc.parameters())
    assert 13_600_000 <= n <= 13_800_000
    keys = voc.state_dict().keys()
    assert all(k.startswith(("backbone.", "head.")) for k in keys)
    assert "head.linear_2.bias" not in keys
