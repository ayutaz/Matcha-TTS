"""WaveNeXt GAN training loop (Lightning 2.x manual optimization).

Structure ported from wetdog/wavenext_pytorch vocos/experiment.py (MIT License, (c) 2023
Charactr Inc.), adapted to Lightning 2.x manual optimization with the D11 corrections:
  - automatic_optimization=False; manual_backward + manual opt.step/zero_grad + sched.step.
  - a ``n_batches`` buffer (NOT global_step, which double-counts at 2 opt.step/batch) drives
    the pretrain gate, scheduler stepping and the stop condition.
  - the D-step uses ``audio_hat.detach()`` so the generator is not updated by the disc loss
    (generator forward runs once per batch).
  - a self-implemented cosine+warmup LambdaLR (no ``transformers`` dependency).
The generator (VocosBackbone + WaveNextHead) is reused from matcha/wavenext/models.py.
"""

import math

import lightning as L
import torch

from matcha.wavenext.models import VocosBackbone, WaveNextHead  # noqa: F401 (re-exported for callers)
from wavenext_train.discriminators import MultiPeriodDiscriminator, MultiResolutionDiscriminator
from wavenext_train.loss import DiscriminatorLoss, FeatureMatchingLoss, GeneratorLoss, MelSpecReconstructionLoss  # noqa: F401


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    """Cosine schedule with linear warmup as a plain LambdaLR — no external scheduler dep (D11)."""

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


class WaveNeXtExp(L.LightningModule):
    def __init__(
        self,
        feature_extractor,
        backbone,
        head,
        melspec_loss,
        sample_rate=22050,
        initial_learning_rate=1e-4,
        num_warmup_steps=500,
        max_steps=1_000_000,
        mel_loss_coeff=45,
        mrd_loss_coeff=0.1,
        pretrain_mel_steps=0,
        decay_mel_coeff=False,
        gradient_clip_val=None,
        mpd_periods=(2, 3, 5, 7, 11),
        mrd_fft_sizes=(2048, 1024, 512),
        use_ema=False,
        ema_decay=0.9995,
    ):
        super().__init__()
        self.automatic_optimization = False  # D11-1
        self.save_hyperparameters(ignore=["feature_extractor", "backbone", "head", "melspec_loss"])
        self.feature_extractor = feature_extractor
        self.backbone = backbone
        self.head = head
        self.melspec_loss = melspec_loss
        self.multiperioddisc = MultiPeriodDiscriminator(periods=tuple(mpd_periods))
        self.multiresddisc = MultiResolutionDiscriminator(fft_sizes=tuple(mrd_fft_sizes))
        self.disc_loss = DiscriminatorLoss()
        self.gen_loss = GeneratorLoss()
        self.feat_matching_loss = FeatureMatchingLoss()
        self.base_mel_coeff = self.mel_loss_coeff = mel_loss_coeff
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.train_discriminator = False
        # D11-2: batch counter (global_step counts 2/batch under manual opt). persistent -> resume-safe.
        self.register_buffer("n_batches", torch.zeros(1, dtype=torch.long))
        if use_ema:
            self._ema_shadow = {n: p.detach().clone() for n, p in self._gen_named_params()}

    def _gen_named_params(self):
        for n, p in self.backbone.named_parameters():
            yield f"backbone.{n}", p
        for n, p in self.head.named_parameters():
            yield f"head.{n}", p

    def generator_forward(self, audio):
        features = self.feature_extractor(audio)  # (B, 80, T) — FP32 mel is enforced inside features
        return self.head(self.backbone(features))  # (B, T*hop), 2D

    def _should_train_disc(self):
        return int(self.n_batches.item()) >= self.hparams.pretrain_mel_steps

    def on_train_batch_start(self, *args):
        self.train_discriminator = self._should_train_disc()

    def configure_optimizers(self):
        # ORDER FIXED: index 0 = discriminators, index 1 = generator.
        opt_d = torch.optim.AdamW(
            [{"params": self.multiperioddisc.parameters()}, {"params": self.multiresddisc.parameters()}],
            lr=self.hparams.initial_learning_rate,
            betas=(0.8, 0.9),
        )
        opt_g = torch.optim.AdamW(
            [{"params": self.backbone.parameters()}, {"params": self.head.parameters()}],  # feature_extractor is param-free
            lr=self.hparams.initial_learning_rate,
            betas=(0.8, 0.9),
        )
        n = self.hparams.max_steps  # batch-unit; scheduler stepped once per batch (no //2)
        sch_d = get_cosine_schedule_with_warmup(opt_d, self.hparams.num_warmup_steps, n)
        sch_g = get_cosine_schedule_with_warmup(opt_g, self.hparams.num_warmup_steps, n)
        return [opt_d, opt_g], [sch_d, sch_g]

    def _clip(self, opt):
        if self.hparams.gradient_clip_val:
            self.clip_gradients(opt, gradient_clip_val=self.hparams.gradient_clip_val, gradient_clip_algorithm="norm")

    def training_step(self, batch, batch_idx):
        audio = batch
        opt_d, opt_g = self.optimizers()
        sch_d, sch_g = self.lr_schedulers()
        train_disc = self._should_train_disc()
        audio_hat = self.generator_forward(audio)  # generator forward once (D11-3)
        out = {"loss_d": None, "loss_g": None, "mel_loss": None}

        if train_disc:
            opt_d.zero_grad(set_to_none=True)
            y_hat_d = audio_hat.detach()  # D11-3: no gradient to the generator
            real_mp, gen_mp, _, _ = self.multiperioddisc(y=audio, y_hat=y_hat_d)
            real_mrd, gen_mrd, _, _ = self.multiresddisc(y=audio, y_hat=y_hat_d)
            loss_mp, loss_mp_real, _ = self.disc_loss(real_mp, gen_mp)
            loss_mrd, loss_mrd_real, _ = self.disc_loss(real_mrd, gen_mrd)
            loss_mp = loss_mp / len(loss_mp_real)
            loss_mrd = loss_mrd / len(loss_mrd_real)
            loss_d = loss_mp + self.hparams.mrd_loss_coeff * loss_mrd
            self.manual_backward(loss_d)
            self._clip(opt_d)
            opt_d.step()
            out["loss_d"] = loss_d.detach()

        opt_g.zero_grad(set_to_none=True)
        if train_disc:
            _, gen_mp, fmap_rs_mp, fmap_gs_mp = self.multiperioddisc(y=audio, y_hat=audio_hat)
            _, gen_mrd, fmap_rs_mrd, fmap_gs_mrd = self.multiresddisc(y=audio, y_hat=audio_hat)
            loss_gen_mp, list_mp = self.gen_loss(gen_mp)
            loss_gen_mrd, list_mrd = self.gen_loss(gen_mrd)
            loss_gen_mp = loss_gen_mp / len(list_mp)
            loss_gen_mrd = loss_gen_mrd / len(list_mrd)
            loss_fm_mp = self.feat_matching_loss(fmap_rs_mp, fmap_gs_mp) / len(fmap_rs_mp)
            loss_fm_mrd = self.feat_matching_loss(fmap_rs_mrd, fmap_gs_mrd) / len(fmap_rs_mrd)
        else:
            loss_gen_mp = loss_gen_mrd = loss_fm_mp = loss_fm_mrd = 0.0
        mel_loss = self.melspec_loss(audio_hat, audio)
        loss_g = (
            loss_gen_mp
            + self.hparams.mrd_loss_coeff * loss_gen_mrd
            + loss_fm_mp
            + self.hparams.mrd_loss_coeff * loss_fm_mrd
            + self.mel_loss_coeff * mel_loss
        )
        self.manual_backward(loss_g)
        self._clip(opt_g)
        opt_g.step()
        out["loss_g"] = loss_g.detach()
        out["mel_loss"] = mel_loss.detach()

        metrics = {"train/loss_g": out["loss_g"], "train/mel_loss": out["mel_loss"]}
        if out["loss_d"] is not None:
            metrics["train/loss_d"] = out["loss_d"]
        self.log_dict(metrics, on_step=True, on_epoch=False, prog_bar=True, batch_size=audio.shape[0])

        sch_d.step()
        sch_g.step()  # D11-2: one scheduler step per batch
        if self.use_ema:
            self._update_ema()
        self.n_batches += 1
        if int(self.n_batches.item()) >= self.hparams.max_steps:
            self.trainer.should_stop = True
        return out

    def on_train_batch_end(self, *args):
        if self.hparams.decay_mel_coeff:
            self.mel_loss_coeff = self.base_mel_coeff * self._mel_coeff_decay(int(self.n_batches.item()))

    def _mel_coeff_decay(self, step, num_cycles=0.5):
        if step < self.hparams.num_warmup_steps:
            return 1.0
        progress = float(step - self.hparams.num_warmup_steps) / float(
            max(1, self.hparams.max_steps - self.hparams.num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    @torch.no_grad()
    def _update_ema(self):
        cur = dict(self._gen_named_params())
        for k, shadow in self._ema_shadow.items():
            shadow.mul_(self.ema_decay).add_(cur[k].detach(), alpha=1.0 - self.ema_decay)

    def validation_step(self, batch, batch_idx):
        mel_loss = self.melspec_loss(self.generator_forward(batch), batch)
        self.log("val/mel_loss", mel_loss, prog_bar=True, sync_dist=True)
        return mel_loss
