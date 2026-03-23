"""
This is a base lightning module that can be used to train a model.
The benefit of this abstraction is that all the logic outside of model definition can be reused for different models.
"""

from abc import ABC
from typing import Any, Dict

import hydra
import torch
from lightning import LightningModule
from lightning.pytorch.utilities import grad_norm

from matcha import utils
from matcha.utils.utils import plot_tensor

log = utils.get_pylogger(__name__)


class BaseLightningClass(LightningModule, ABC):
    def update_data_statistics(self, data_statistics):
        if data_statistics is None:
            data_statistics = {
                "mel_mean": 0.0,
                "mel_std": 1.0,
            }

        self.register_buffer("mel_mean", torch.tensor(data_statistics["mel_mean"]))
        self.register_buffer("mel_std", torch.tensor(data_statistics["mel_std"]))

    def configure_optimizers(self) -> Any:
        optimizer = self.hparams.optimizer(params=self.parameters())
        if getattr(self.hparams, "scheduler", None) is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                },
            }

        return {"optimizer": optimizer}

    def get_losses(self, batch):
        x, x_lengths = batch["x"], batch["x_lengths"]
        y, y_lengths = batch["y"], batch["y_lengths"]
        spks = batch["spks"]

        dur_loss, prior_loss, diff_loss, *_ = self(
            x=x,
            x_lengths=x_lengths,
            y=y,
            y_lengths=y_lengths,
            spks=spks,
            out_size=self.out_size,
            durations=batch["durations"],
        )
        return {
            "dur_loss": dur_loss,
            "prior_loss": prior_loss,
            "diff_loss": diff_loss,
        }

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        self.ckpt_loaded_epoch = checkpoint["epoch"]  # pylint: disable=attribute-defined-outside-init

    def training_step(self, batch: Any, batch_idx: int):
        loss_dict = self.get_losses(batch)
        total_loss = loss_dict["dur_loss"] + 0.5 * loss_dict["prior_loss"] + loss_dict["diff_loss"]

        self.log_dict(
            {
                "sub_loss/train_dur_loss": loss_dict["dur_loss"],
                "sub_loss/train_prior_loss": loss_dict["prior_loss"],
                "sub_loss/train_diff_loss": loss_dict["diff_loss"],
                "loss/train": total_loss,
            },
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            sync_dist=False,
        )

        return {"loss": total_loss, "log": loss_dict}

    def validation_step(self, batch: Any, batch_idx: int):
        loss_dict = self.get_losses(batch)
        total_loss = loss_dict["dur_loss"] + 0.5 * loss_dict["prior_loss"] + loss_dict["diff_loss"]

        self.log_dict(
            {
                "sub_loss/val_dur_loss": loss_dict["dur_loss"],
                "sub_loss/val_prior_loss": loss_dict["prior_loss"],
                "sub_loss/val_diff_loss": loss_dict["diff_loss"],
                "loss/val": total_loss,
            },
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            sync_dist=False,
        )

        return total_loss

    def on_validation_end(self) -> None:
        if self.trainer.is_global_zero:
            if self.current_epoch % 10 != 0:
                return
            one_batch = next(iter(self.trainer.val_dataloaders))
            if self.current_epoch == 0:
                log.debug("Plotting original samples")
                for i in range(2):
                    y = one_batch["y"][i].unsqueeze(0).to(self.device)
                    self.logger.experiment.add_image(
                        f"original/{i}",
                        plot_tensor(y.squeeze().cpu()),
                        self.current_epoch,
                        dataformats="HWC",
                    )

            log.debug("Synthesising...")
            for i in range(2):
                x = one_batch["x"][i].unsqueeze(0).to(self.device)
                x_lengths = one_batch["x_lengths"][i].unsqueeze(0).to(self.device)
                spks = one_batch["spks"][i].unsqueeze(0).to(self.device) if one_batch["spks"] is not None else None
                output = self.synthesise(x[:, :x_lengths], x_lengths, n_timesteps=10, spks=spks)
                y_enc, y_dec = output["encoder_outputs"], output["decoder_outputs"]
                attn = output["attn"]
                self.logger.experiment.add_image(
                    f"generated_enc/{i}",
                    plot_tensor(y_enc.squeeze().cpu()),
                    self.current_epoch,
                    dataformats="HWC",
                )
                self.logger.experiment.add_image(
                    f"generated_dec/{i}",
                    plot_tensor(y_dec.squeeze().cpu()),
                    self.current_epoch,
                    dataformats="HWC",
                )
                self.logger.experiment.add_image(
                    f"alignment/{i}",
                    plot_tensor(attn.squeeze().cpu()),
                    self.current_epoch,
                    dataformats="HWC",
                )

    def on_before_optimizer_step(self, optimizer):
        grad_norm_interval = getattr(self, "grad_norm_log_interval", 500)
        if grad_norm_interval > 0 and self.global_step % grad_norm_interval == 0:
            self.log_dict({f"grad_norm/{k}": v for k, v in grad_norm(self, norm_type=2).items()})
