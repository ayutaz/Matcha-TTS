"""Tests for training evaluation and monitoring scripts."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from check_training_health import check_health
from evaluate_durations import evaluate_durations


class TestEvaluateDurations:
    def _make_eval_data(self, tmp_path, n_samples=3):
        """Create dummy .pt files with durations for evaluation."""
        for i in range(n_samples):
            text_len = 11 + i * 2
            mel_len = 30 + i * 10
            text = torch.randint(0, 55, (text_len,), dtype=torch.int32)
            mel = torch.randn(80, mel_len)
            dur = torch.zeros(text_len, dtype=torch.long)
            n_ph = text_len // 2
            if n_ph > 0:
                base = mel_len // n_ph
                rem = mel_len - base * n_ph
                for j in range(n_ph):
                    dur[2 * j + 1] = base + (1 if j < rem else 0)
            torch.save(
                {
                    "mel": mel,
                    "text": text,
                    "spk": i % 10,
                    "cleaned_text": f"test_{i}",
                    "durations": dur,
                },
                tmp_path / f"sample_{i:04d}.pt",
            )

    def test_evaluate_with_valid_data(self, tmp_path):
        """Dummy data files are loadable and contain expected keys."""
        self._make_eval_data(tmp_path, 3)
        pt_files = sorted(tmp_path.glob("*.pt"))
        assert len(pt_files) == 3
        # Validate data structure without requiring a real model
        for pt_path in pt_files:
            data = torch.load(pt_path, weights_only=True)
            assert "durations" in data
            assert data["durations"].sum().item() > 0

    def test_data_dir_validation(self, tmp_path):
        """Non-existent data directory returns exit code 1."""
        from evaluate_durations import main

        ret = main(["--checkpoint", "dummy.ckpt", "--data-dir", "/nonexistent"])
        assert ret == 1

    def test_missing_checkpoint_skips_gracefully(self, tmp_path):
        """Missing checkpoint file prints message and returns 0."""
        self._make_eval_data(tmp_path, 1)
        from evaluate_durations import main

        ret = main(
            [
                "--checkpoint",
                str(tmp_path / "nonexistent.ckpt"),
                "--data-dir",
                str(tmp_path),
            ]
        )
        assert ret == 0

    def test_output_dir_creation(self, tmp_path):
        """Output directory is created when specified."""
        self._make_eval_data(tmp_path, 1)
        out_dir = tmp_path / "results" / "nested"
        from evaluate_durations import main

        main(
            [
                "--checkpoint",
                "dummy.ckpt",
                "--data-dir",
                str(tmp_path),
                "--output-dir",
                str(out_dir),
            ]
        )
        assert out_dir.exists()

    def test_duration_sum_consistency(self, tmp_path):
        """Duration arrays in generated test data sum to mel length."""
        self._make_eval_data(tmp_path, 5)
        for pt_path in sorted(tmp_path.glob("*.pt")):
            data = torch.load(pt_path, weights_only=True)
            dur_sum = data["durations"].sum().item()
            mel_len = data["mel"].shape[1]
            assert dur_sum == mel_len, f"{pt_path.name}: duration sum {dur_sum} != mel length {mel_len}"


class TestCheckTrainingHealth:
    def test_healthy_log_dir(self, tmp_path):
        """Fully healthy log directory reports no issues."""
        (tmp_path / "checkpoints").mkdir()
        (tmp_path / "checkpoints" / "last.ckpt").touch()
        (tmp_path / "checkpoints" / "epoch_10.ckpt").touch()
        (tmp_path / "tensorboard").mkdir()
        (tmp_path / "tensorboard" / "events.out.tfevents.12345").touch()

        results = check_health(tmp_path)
        assert results["checkpoints_found"] == 2
        assert results["last_ckpt_exists"] is True
        assert results["tensorboard_events"] == 1
        assert len(results["issues"]) == 0

    def test_missing_checkpoints(self, tmp_path):
        """Missing checkpoints directory is reported as an issue."""
        results = check_health(tmp_path)
        assert "No checkpoints directory found" in results["issues"]

    def test_missing_tensorboard(self, tmp_path):
        """Missing TensorBoard events are reported as an issue."""
        (tmp_path / "checkpoints").mkdir()
        (tmp_path / "checkpoints" / "last.ckpt").touch()
        results = check_health(tmp_path)
        assert "No TensorBoard event files found" in results["issues"]

    def test_nonexistent_dir(self):
        """Non-existent directory returns exit code 1."""
        from check_training_health import main

        ret = main(["--log-dir", "/nonexistent"])
        assert ret == 1

    def test_healthy_returns_0(self, tmp_path):
        """Healthy directory returns exit code 0."""
        (tmp_path / "checkpoints").mkdir()
        (tmp_path / "checkpoints" / "last.ckpt").touch()
        (tmp_path / "tensorboard").mkdir()
        (tmp_path / "tensorboard" / "events.out.tfevents.12345").touch()
        from check_training_health import main

        ret = main(["--log-dir", str(tmp_path)])
        assert ret == 0

    def test_events_in_root_dir(self, tmp_path):
        """TensorBoard events directly under log dir are also found."""
        (tmp_path / "checkpoints").mkdir()
        (tmp_path / "checkpoints" / "last.ckpt").touch()
        # Event file directly under log dir (no tensorboard/ subdir)
        (tmp_path / "events.out.tfevents.99999").touch()

        results = check_health(tmp_path)
        assert results["tensorboard_events"] == 1
        assert len(results["issues"]) == 0

    def test_multiple_checkpoints(self, tmp_path):
        """Multiple checkpoint files are correctly counted."""
        (tmp_path / "checkpoints").mkdir()
        (tmp_path / "checkpoints" / "last.ckpt").touch()
        (tmp_path / "checkpoints" / "epoch_100.ckpt").touch()
        (tmp_path / "checkpoints" / "epoch_200.ckpt").touch()
        (tmp_path / "tensorboard").mkdir()
        (tmp_path / "tensorboard" / "events.out.tfevents.12345").touch()

        results = check_health(tmp_path)
        assert results["checkpoints_found"] == 3
        assert results["last_ckpt_exists"] is True

    def test_no_last_ckpt(self, tmp_path):
        """Checkpoint dir exists but no last.ckpt is detected."""
        (tmp_path / "checkpoints").mkdir()
        (tmp_path / "checkpoints" / "epoch_50.ckpt").touch()
        (tmp_path / "tensorboard").mkdir()
        (tmp_path / "tensorboard" / "events.out.tfevents.12345").touch()

        results = check_health(tmp_path)
        assert results["checkpoints_found"] == 1
        assert results["last_ckpt_exists"] is False
