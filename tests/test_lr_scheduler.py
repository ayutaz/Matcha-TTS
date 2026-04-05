"""Tests for warmup + cosine decay LR scheduler.

Validates that the SequentialLR built by build_warmup_cosine_scheduler behaves
correctly: linear warmup from start_factor*lr to lr, then cosine decay down to
eta_min, never exceeding peak lr and never dropping below eta_min.
"""

from types import SimpleNamespace

import pytest
import torch

from matcha.models.baselightningmodule import build_warmup_cosine_scheduler

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_optimizer(lr: float = 1e-4):
    """Create a minimal AdamW optimizer with a single dummy parameter."""
    param = torch.nn.Parameter(torch.zeros(1))
    return torch.optim.AdamW([param], lr=lr, weight_decay=0.0)


def _get_lr(optimizer):
    """Return the current learning rate of the first param group."""
    return optimizer.param_groups[0]["lr"]


def _make_default_cfg(**overrides):
    """Build a SimpleNamespace scheduler config with defaults."""
    cfg = {
        "type": "warmup_cosine_safe",
        "warmup_steps": 500,
        "start_factor": 0.1,
        "T_max": 20000,
        "eta_min": 5e-5,
    }
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestWarmupCosineScheduler:
    """Core behaviour of the warmup + cosine decay scheduler."""

    def test_initial_lr_is_start_factor_times_base(self):
        """At step 0, lr should equal base_lr * start_factor."""
        optimizer = _make_optimizer(lr=1e-4)
        cfg = _make_default_cfg()
        scheduler = build_warmup_cosine_scheduler(optimizer, cfg)

        lr = _get_lr(optimizer)
        assert lr == pytest.approx(1e-4 * 0.1, rel=1e-6), (
            f"Initial lr should be 1e-5, got {lr}"
        )

    def test_lr_increases_during_warmup(self):
        """LR should monotonically increase throughout the warmup phase."""
        optimizer = _make_optimizer(lr=1e-4)
        cfg = _make_default_cfg()
        scheduler = build_warmup_cosine_scheduler(optimizer, cfg)

        prev_lr = _get_lr(optimizer)
        for step in range(1, 500):
            scheduler.step()
            cur_lr = _get_lr(optimizer)
            assert cur_lr >= prev_lr, (
                f"LR decreased at warmup step {step}: {prev_lr} -> {cur_lr}"
            )
            prev_lr = cur_lr

    def test_peak_lr_at_warmup_end(self):
        """After the full warmup phase, lr should reach the base lr (1e-4)."""
        optimizer = _make_optimizer(lr=1e-4)
        cfg = _make_default_cfg()
        scheduler = build_warmup_cosine_scheduler(optimizer, cfg)

        for _ in range(500):
            scheduler.step()

        lr = _get_lr(optimizer)
        assert lr == pytest.approx(1e-4, rel=1e-5), (
            f"LR at warmup end should be 1e-4, got {lr}"
        )

    def test_cosine_decay_after_warmup(self):
        """After warmup, lr should decrease (cosine decay)."""
        optimizer = _make_optimizer(lr=1e-4)
        cfg = _make_default_cfg()
        scheduler = build_warmup_cosine_scheduler(optimizer, cfg)

        # Complete warmup
        for _ in range(500):
            scheduler.step()

        peak_lr = _get_lr(optimizer)

        # Step a few hundred more into cosine phase
        for _ in range(500):
            scheduler.step()

        lr_after_decay = _get_lr(optimizer)
        assert lr_after_decay < peak_lr, (
            f"LR should decrease after warmup: peak={peak_lr}, current={lr_after_decay}"
        )

    def test_lr_never_exceeds_peak(self):
        """LR should never exceed the base lr (1e-4) at any point."""
        optimizer = _make_optimizer(lr=1e-4)
        cfg = _make_default_cfg()
        scheduler = build_warmup_cosine_scheduler(optimizer, cfg)

        peak = 1e-4
        for step in range(20500):
            lr = _get_lr(optimizer)
            assert lr <= peak + 1e-9, (
                f"LR exceeded peak at step {step}: {lr} > {peak}"
            )
            scheduler.step()

    def test_lr_never_below_eta_min(self):
        """LR should never drop below eta_min (5e-5)."""
        optimizer = _make_optimizer(lr=1e-4)
        cfg = _make_default_cfg()
        scheduler = build_warmup_cosine_scheduler(optimizer, cfg)

        eta_min = 5e-5
        # During warmup, lr starts at 1e-5 which is below eta_min -- that is
        # expected.  The constraint applies to the cosine phase only.
        # Skip warmup
        for _ in range(500):
            scheduler.step()

        for step in range(500, 20500):
            lr = _get_lr(optimizer)
            assert lr >= eta_min - 1e-9, (
                f"LR below eta_min at step {step}: {lr} < {eta_min}"
            )
            scheduler.step()

    def test_lr_at_T_max_equals_eta_min(self):
        """At step warmup_steps + T_max, lr should be approximately eta_min."""
        optimizer = _make_optimizer(lr=1e-4)
        cfg = _make_default_cfg()
        scheduler = build_warmup_cosine_scheduler(optimizer, cfg)

        total_steps = 500 + 20000
        for _ in range(total_steps):
            scheduler.step()

        lr = _get_lr(optimizer)
        assert lr == pytest.approx(5e-5, rel=1e-4), (
            f"LR at T_max should be ~5e-5, got {lr}"
        )

    def test_warmup_linearity(self):
        """Warmup phase should produce a roughly linear lr increase."""
        optimizer = _make_optimizer(lr=1e-4)
        cfg = _make_default_cfg()
        scheduler = build_warmup_cosine_scheduler(optimizer, cfg)

        # Sample lr at 25%, 50%, 75% of warmup
        checkpoints = {125: None, 250: None, 375: None}
        for step in range(500):
            if step in checkpoints:
                checkpoints[step] = _get_lr(optimizer)
            scheduler.step()

        # Expected: linear from 1e-5 to 1e-4 over 500 steps
        for step, lr in checkpoints.items():
            expected = 1e-5 + (1e-4 - 1e-5) * (step / 500)
            assert lr == pytest.approx(expected, rel=0.05), (
                f"Warmup not linear at step {step}: expected ~{expected}, got {lr}"
            )


class TestConfigureOptimizersIntegration:
    """Verify that configure_optimizers correctly builds the scheduler."""

    def test_returns_scheduler_dict_for_warmup_cosine_safe(self):
        """configure_optimizers should return an lr_scheduler dict when
        scheduler config has type=warmup_cosine_safe."""
        from matcha.models.baselightningmodule import BaseLightningClass

        # Create a minimal concrete subclass
        class DummyModule(BaseLightningClass):
            def __init__(self):
                super().__init__()
                self._dummy = torch.nn.Linear(1, 1)
                self.save_hyperparameters(
                    {
                        "optimizer": None,
                        "scheduler": None,
                    },
                    logger=False,
                )

            def forward(self, x):
                return x

        module = DummyModule()

        # Patch hparams to match what Hydra would produce
        optimizer_partial = lambda params: torch.optim.AdamW(params, lr=1e-4, weight_decay=0.0)
        scheduler_cfg = SimpleNamespace(
            type="warmup_cosine_safe",
            warmup_steps=500,
            start_factor=0.1,
            T_max=20000,
            eta_min=5e-5,
        )
        module.hparams.optimizer = optimizer_partial
        module.hparams.scheduler = scheduler_cfg

        result = module.configure_optimizers()

        assert "optimizer" in result
        assert "lr_scheduler" in result
        lr_sched = result["lr_scheduler"]
        assert lr_sched["interval"] == "step"
        assert isinstance(
            lr_sched["scheduler"],
            torch.optim.lr_scheduler.SequentialLR,
        )

    def test_returns_no_scheduler_when_none(self):
        """configure_optimizers should return only optimizer when scheduler is
        None."""
        from matcha.models.baselightningmodule import BaseLightningClass

        class DummyModule(BaseLightningClass):
            def __init__(self):
                super().__init__()
                self._dummy = torch.nn.Linear(1, 1)
                self.save_hyperparameters(
                    {
                        "optimizer": None,
                        "scheduler": None,
                    },
                    logger=False,
                )

            def forward(self, x):
                return x

        module = DummyModule()
        optimizer_partial = lambda params: torch.optim.AdamW(params, lr=1e-4, weight_decay=0.0)
        module.hparams.optimizer = optimizer_partial

        result = module.configure_optimizers()

        assert "optimizer" in result
        assert "lr_scheduler" not in result


class TestCustomParameters:
    """Ensure non-default parameter values are respected."""

    def test_custom_warmup_steps(self):
        """Scheduler with warmup_steps=100 should reach peak at step 100."""
        optimizer = _make_optimizer(lr=1e-4)
        cfg = _make_default_cfg(warmup_steps=100)
        scheduler = build_warmup_cosine_scheduler(optimizer, cfg)

        for _ in range(100):
            scheduler.step()

        lr = _get_lr(optimizer)
        assert lr == pytest.approx(1e-4, rel=1e-5)

    def test_custom_eta_min(self):
        """Scheduler with eta_min=7e-5 should not drop below 7e-5 in cosine phase."""
        optimizer = _make_optimizer(lr=1e-4)
        cfg = _make_default_cfg(eta_min=7e-5, T_max=1000)
        scheduler = build_warmup_cosine_scheduler(optimizer, cfg)

        # Complete warmup + full cosine
        for _ in range(500 + 1000):
            scheduler.step()

        lr = _get_lr(optimizer)
        assert lr == pytest.approx(7e-5, rel=1e-4)
