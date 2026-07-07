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
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def ddp_optimized_config():
    path = CONFIGS_DIR / "trainer" / "ddp_optimized.yaml"
    with open(path, encoding="utf-8") as f:
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

    def test_precision_bf16_mixed(self, jvs_fast_config):
        """precision should be bf16-mixed (RTX 5090 validated; +11% steps/sec, quality equal).

        Note: bf16 != FP16. FP16 degrades the Duration Predictor, but bf16 has an 8-bit
        exponent so it does not overflow and matches FP32 quality (see docs/eval report).
        """
        assert jvs_fast_config["trainer"]["precision"] == "bf16-mixed"

    def test_optimizer_fused_disabled_for_mixed(self, jvs_fast_config):
        """bf16-mixed requires fused=false (fused AdamW + mixed + grad clipping crashes)."""
        assert jvs_fast_config["model"]["optimizer"]["fused"] is False

    def test_no_scheduler(self, jvs_fast_config):
        """No LR scheduler should be configured (paper-faithful: constant lr=1e-4)."""
        assert "scheduler" not in jvs_fast_config.get("model", {})

    def test_early_stopping_patience(self, jvs_fast_config):
        """patience should be 30 (effective 300 epochs with check_every=10)."""
        assert jvs_fast_config["callbacks"]["early_stopping"]["patience"] == 30

    def test_ema_start_epoch_10(self, jvs_fast_config):
        """EMA should start at epoch 10 (paper-faithful)."""
        assert jvs_fast_config["callbacks"]["ema"]["update_starting_at_epoch"] == 10


class TestJvsAlignedConfig:
    """Validate jvs_aligned.yaml configuration for duration-based training."""

    @pytest.fixture
    def jvs_aligned_config(self):
        path = CONFIGS_DIR / "experiment" / "jvs_aligned.yaml"
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)

    @pytest.fixture
    def jvs_precomputed_aligned_config(self):
        path = CONFIGS_DIR / "data" / "jvs_precomputed_aligned.yaml"
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)

    def test_compile_model_disabled(self, jvs_aligned_config):
        """compile_model should be false for DDP stability."""
        assert jvs_aligned_config["compile_model"] is False

    def test_gradient_checkpointing_disabled(self, jvs_aligned_config):
        """gradient_checkpointing should be false (static_graph compatible)."""
        assert jvs_aligned_config["gradient_checkpointing"] is False

    def test_max_epochs(self, jvs_aligned_config):
        """max_epochs should be 2500."""
        assert jvs_aligned_config["trainer"]["max_epochs"] == 2500

    def test_precision_bf16_mixed(self, jvs_aligned_config):
        """precision should be bf16-mixed (shipped 2500ep model trained this way, passed all gates)."""
        assert jvs_aligned_config["trainer"]["precision"] == "bf16-mixed"

    def test_optimizer_fused_disabled_for_mixed(self, jvs_aligned_config):
        """bf16-mixed requires fused=false (fused AdamW + mixed + grad clipping crashes)."""
        assert jvs_aligned_config["model"]["optimizer"]["fused"] is False

    def test_compile_regional_blocks_default_off(self, jvs_aligned_config):
        """A-2 regional compile must default OFF so the proven recipe stays byte-identical."""
        assert jvs_aligned_config["compile_regional_blocks"] is False

    def test_tags_include_aligned(self, jvs_aligned_config):
        """tags should include 'aligned'."""
        assert "aligned" in jvs_aligned_config["tags"]

    def test_data_load_durations_true(self, jvs_precomputed_aligned_config):
        """load_durations should be true in data config."""
        assert jvs_precomputed_aligned_config["load_durations"] is True

    def test_data_n_spks(self, jvs_precomputed_aligned_config):
        """n_spks should be 100 for JVS."""
        assert jvs_precomputed_aligned_config["n_spks"] == 100

    def test_model_config_has_use_precomputed_durations(self):
        """matcha.yaml should reference ${data.load_durations}."""
        path = CONFIGS_DIR / "model" / "matcha.yaml"
        with open(path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        assert config["use_precomputed_durations"] == "${data.load_durations}"

    def test_early_stopping_patience(self, jvs_aligned_config):
        """patience should be 30."""
        assert jvs_aligned_config["callbacks"]["early_stopping"]["patience"] == 30

    def test_ema_start_epoch_10(self, jvs_aligned_config):
        """EMA should start at epoch 10."""
        assert jvs_aligned_config["callbacks"]["ema"]["update_starting_at_epoch"] == 10


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
