"""Tests for training configuration optimizations.

Validates that jvs_fast.yaml and ddp_optimized.yaml contain the expected
optimized settings by directly parsing the YAML files.
"""

from pathlib import Path

import pytest
import yaml

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


@pytest.fixture
def jvs_fast_config():
    path = CONFIGS_DIR / "experiment" / "jvs_fast.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


@pytest.fixture
def ddp_optimized_config():
    path = CONFIGS_DIR / "trainer" / "ddp_optimized.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


class TestJvsFastConfig:
    """Validate jvs_fast.yaml optimization settings."""

    def test_gradient_checkpointing_disabled(self, jvs_fast_config):
        """gradient_checkpointing should be false to remove recomputation overhead."""
        assert jvs_fast_config["gradient_checkpointing"] is False

    def test_max_epochs(self, jvs_fast_config):
        """max_epochs should be 2500 (paper-equivalent ~240K steps for 100 speakers)."""
        assert jvs_fast_config["trainer"]["max_epochs"] == 2500

    def test_check_val_every_n_epoch(self, jvs_fast_config):
        """check_val_every_n_epoch should be 10 for effective early stopping."""
        assert jvs_fast_config["trainer"]["check_val_every_n_epoch"] == 10

    def test_precision_fp32(self, jvs_fast_config):
        """precision should be 32-true (FP16 degrades Duration Predictor quality)."""
        assert jvs_fast_config["trainer"]["precision"] == "32-true"

    def test_no_scheduler(self, jvs_fast_config):
        """No LR scheduler should be configured (paper-faithful: constant lr=1e-4)."""
        assert "scheduler" not in jvs_fast_config.get("model", {})

    def test_early_stopping_patience(self, jvs_fast_config):
        """patience should be 30 (effective 300 epochs with check_every=10)."""
        assert jvs_fast_config["callbacks"]["early_stopping"]["patience"] == 30

    def test_ema_start_epoch_10(self, jvs_fast_config):
        """EMA should start at epoch 10 (paper-faithful)."""
        assert jvs_fast_config["callbacks"]["ema"]["update_starting_at_epoch"] == 10


class TestDdpOptimizedConfig:
    """Validate ddp_optimized.yaml optimization settings."""

    def test_static_graph_enabled(self, ddp_optimized_config):
        """static_graph should be true for DDP communication optimization."""
        assert ddp_optimized_config["strategy"]["static_graph"] is True

    def test_gradient_as_bucket_view(self, ddp_optimized_config):
        """gradient_as_bucket_view should remain true."""
        assert ddp_optimized_config["strategy"]["gradient_as_bucket_view"] is True

    def test_find_unused_parameters_disabled(self, ddp_optimized_config):
        """find_unused_parameters should remain false (required for static_graph)."""
        assert ddp_optimized_config["strategy"]["find_unused_parameters"] is False
