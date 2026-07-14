"""TDD tests for wavenext_train.experiment.WaveNeXtExp (Lightning 2.x manual optimization).

Pins the D11 corrections: n_batches buffer accounting (not global_step), y_hat.detach in the
D-step, self-implemented cosine+warmup LambdaLR (no transformers), pretrain gate, and that a
full manual-opt GAN training_step runs on CPU. Tiny model, all CPU.
"""

import inspect

import lightning as L
import torch

from matcha.wavenext.models import VocosBackbone, WaveNextHead
from wavenext_train.experiment import WaveNeXtExp, get_cosine_schedule_with_warmup
from wavenext_train.features import MatchaMelFeatures
from wavenext_train.loss import MelSpecReconstructionLoss

NUM = 4096  # 16 frames * 256; >= MRD fft 2048


def _components():
    return (
        MatchaMelFeatures(fmax=11025),
        VocosBackbone(input_channels=80, dim=16, intermediate_dim=32, num_layers=2),
        WaveNextHead(dim=16, n_fft=1024, hop_length=256),
        MelSpecReconstructionLoss(fmax=11025),
    )


def _exp(cls=WaveNeXtExp, **kw):
    feat, bb, head, ml = _components()
    kw.setdefault("num_warmup_steps", 50)
    kw.setdefault("max_steps", 1000)
    return cls(feat, bb, head, ml, **kw)


class _WavDS(torch.utils.data.Dataset):
    def __init__(self, n, length=NUM):
        self.data = torch.randn(n, length)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def _loader(n=6, bs=2):
    return torch.utils.data.DataLoader(_WavDS(n), batch_size=bs)


def _trainer(**kw):
    return L.Trainer(accelerator="cpu", precision="32-true", logger=False,
                     enable_checkpointing=False, enable_progress_bar=False,
                     enable_model_summary=False, limit_val_batches=0, **kw)


def test_configure_optimizers_returns_two_opts_two_scheds():
    exp = _exp(initial_learning_rate=1e-4)
    opts, scheds = exp.configure_optimizers()
    assert len(opts) == 2 and len(scheds) == 2
    assert all(isinstance(o, torch.optim.AdamW) for o in opts)
    assert opts[0].defaults["lr"] == 1e-4  # param_groups lr is 0.0 during warmup
    assert opts[0].param_groups[0]["betas"] == (0.8, 0.9)
    disc_ids = {id(p) for p in exp.multiperioddisc.parameters()} | {id(p) for p in exp.multiresddisc.parameters()}
    gen_ids = {id(p) for p in exp.backbone.parameters()} | {id(p) for p in exp.head.parameters()}
    assert {id(p) for g in opts[0].param_groups for p in g["params"]} == disc_ids
    assert {id(p) for g in opts[1].param_groups for p in g["params"]} == gen_ids
    assert all(isinstance(s, torch.optim.lr_scheduler.LambdaLR) for s in scheds)


def test_cosine_warmup_lambda_matches_formula():
    opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1e-4)
    sch = get_cosine_schedule_with_warmup(opt, num_warmup_steps=50, num_training_steps=500)
    fn = sch.lr_lambdas[0]
    assert abs(fn(0) - 0.0) < 1e-6
    assert abs(fn(25) - 0.5) < 1e-6
    assert abs(fn(50) - 1.0) < 1e-6
    assert abs(fn(275) - 0.5) < 1e-6
    assert abs(fn(500) - 0.0) < 1e-6
    assert "transformers" not in inspect.getsource(get_cosine_schedule_with_warmup)


def test_n_batches_buffer_increments_one_per_batch():
    exp = _exp(pretrain_mel_steps=0)
    assert exp.n_batches.dtype == torch.long and int(exp.n_batches.item()) == 0
    _trainer(max_epochs=1, limit_train_batches=3).fit(exp, _loader())
    assert int(exp.n_batches.item()) == 3
    assert exp.global_step != int(exp.n_batches.item())  # global_step double-counts (2/batch)


def test_pretrain_gate_toggles_train_discriminator():
    exp = _exp(pretrain_mel_steps=2)
    exp.n_batches.fill_(1)
    exp.on_train_batch_start()
    assert exp.train_discriminator is False
    exp.n_batches.fill_(2)
    exp.on_train_batch_start()
    assert exp.train_discriminator is True  # >= inclusive


def test_detach_blocks_generator_gradient_in_disc_step():
    exp = _exp(pretrain_mel_steps=0)
    audio = torch.randn(2, NUM)
    audio_hat = exp.generator_forward(audio)
    d_in = audio_hat.detach()
    assert d_in.requires_grad is False
    real_mp, gen_mp, _, _ = exp.multiperioddisc(y=audio, y_hat=d_in)
    loss_d, _, _ = exp.disc_loss(real_mp, gen_mp)
    loss_d.backward()
    assert all(p.grad is None for p in exp.backbone.parameters())
    assert all(p.grad is None for p in exp.head.parameters())
    assert any(p.grad is not None for p in exp.multiperioddisc.parameters())


def test_training_step_runs_full_gan_and_steps_both_optimizers():
    records = []

    class _Spy(WaveNeXtExp):
        def training_step(self, batch, batch_idx):
            out = super().training_step(batch, batch_idx)
            records.append(out)
            return out

    exp = _exp(cls=_Spy, pretrain_mel_steps=0)
    _trainer(max_epochs=1, limit_train_batches=2).fit(exp, _loader())
    assert records and all(torch.isfinite(r["loss_g"]).all() for r in records)
    assert all(r["loss_d"] is not None and torch.isfinite(r["loss_d"]).all() for r in records)


def test_pretrain_phase_skips_disc_and_adversarial_terms():
    records = []

    class _Spy(WaveNeXtExp):
        def training_step(self, batch, batch_idx):
            out = super().training_step(batch, batch_idx)
            records.append(out)
            return out

    exp = _exp(cls=_Spy, pretrain_mel_steps=100)  # never reached
    mpd_before = [p.detach().clone() for p in exp.multiperioddisc.parameters()]
    bb_before = [p.detach().clone() for p in exp.backbone.parameters()]
    _trainer(max_epochs=1, limit_train_batches=2).fit(exp, _loader())
    assert all(r["loss_d"] is None for r in records)
    # loss_g == mel_loss_coeff * mel_loss during pretrain
    assert all(torch.allclose(r["loss_g"], exp.mel_loss_coeff * r["mel_loss"], atol=1e-5) for r in records)
    assert all(torch.equal(a, b) for a, b in zip(exp.multiperioddisc.parameters(), mpd_before))
    assert any(not torch.equal(a, b) for a, b in zip(exp.backbone.parameters(), bb_before))
