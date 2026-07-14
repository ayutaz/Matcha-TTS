"""Tests for training evaluation and monitoring scripts."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from check_training_health import check_health
from evaluate_durations import evaluate_durations
from transfer_from_english import main as transfer_main


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

        # Use a tmp_path-based path: a bare "/nonexistent" may actually exist
        # on some machines (e.g. C:\nonexistent left behind by other runs).
        ret = main(["--checkpoint", "dummy.ckpt", "--data-dir", str(tmp_path / "no_such_dir")])
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

    # ------------------------------------------------------------------
    # Core-metric tests using a stub model (no checkpoint required)
    # ------------------------------------------------------------------

    def _make_stub_model(self, pred_dur_by_len, n_spks=1):
        """Build a minimal stand-in for MatchaTTS exposing what evaluate_durations uses.

        ``pred_dur_by_len`` maps text length T to a 1-D tensor of desired
        predicted durations; the stub encoder returns ``log(durations)`` shaped
        (1, 1, T) so that ``round(exp(logw))`` reproduces them exactly.
        Every encoder call is recorded in ``model.calls``.
        """
        calls = []

        def encoder(x, x_lengths, spks=None):
            t = int(x.shape[1])
            calls.append({"length": t, "spks": spks})
            logw = torch.log(pred_dur_by_len[t].float()).view(1, 1, t)
            mu_x = torch.zeros(1, 2, t)
            x_mask = torch.ones(1, 1, t)
            return mu_x, logw, x_mask

        model = SimpleNamespace(n_spks=n_spks, encoder=encoder, calls=calls)
        if n_spks > 1:
            model.spk_emb = lambda spk_tensor: torch.zeros(1, 4)
        model.to = lambda device: model
        return model

    def _save_sample(self, path, text_len, durations, spk=0):
        """Save one precomputed .pt sample matching the evaluate_durations schema."""
        torch.save(
            {
                "mel": torch.randn(80, 20),
                "text": torch.randint(0, 55, (text_len,), dtype=torch.int32),
                "spk": spk,
                "cleaned_text": "stub",
                "durations": durations,
            },
            path,
        )

    def test_mae_hand_computed(self, tmp_path):
        """mae_mean matches a hand-computed MAE over phoneme (odd) indices."""
        model = self._make_stub_model({5: torch.tensor([1.0, 5.0, 1.0, 8.0, 1.0])})
        self._save_sample(tmp_path / "sample_0000.pt", 5, torch.tensor([0, 4, 0, 6, 0]))

        results = evaluate_durations(model, tmp_path)

        # Phoneme positions (odd indices): pred [5, 8] vs target [4, 6]
        # MAE = (|5 - 4| + |8 - 6|) / 2 = 1.5
        assert results["n_samples"] == 1
        assert results["mae_mean"] == pytest.approx(1.5)
        assert results["mae_median"] == pytest.approx(1.5)
        assert results["pred_duration_stats"]["mean"] == pytest.approx(6.5)
        assert results["target_duration_stats"]["mean"] == pytest.approx(5.0)
        assert results["degenerate_count"] == 0
        assert results["degenerate_rate"] == 0.0

    def test_degenerate_predictions_counted(self, tmp_path):
        """Predictions with >=80% of phonemes <=1 frame increment degenerate_count."""
        # Sample 0 (T=5): predicted phonemes [1, 1] -> 100% <= 1 frame -> degenerate
        # Sample 1 (T=7): predicted phonemes [6, 7, 8] -> 0% <= 1 frame -> healthy
        model = self._make_stub_model(
            {
                5: torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]),
                7: torch.tensor([1.0, 6.0, 1.0, 7.0, 1.0, 8.0, 1.0]),
            }
        )
        self._save_sample(tmp_path / "sample_0000.pt", 5, torch.tensor([0, 4, 0, 6, 0]))
        self._save_sample(tmp_path / "sample_0001.pt", 7, torch.tensor([0, 6, 0, 7, 0, 8, 0]))

        results = evaluate_durations(model, tmp_path)
        assert results["n_samples"] == 2
        assert results["degenerate_count"] == 1
        assert results["degenerate_rate"] == pytest.approx(0.5)

    def test_length_mismatch_silently_skipped(self, tmp_path):
        """A sample whose prediction length differs from target length is skipped."""
        model = self._make_stub_model({5: torch.tensor([1.0, 5.0, 1.0, 8.0, 1.0])})
        # Valid: text length 5, target durations length 5
        self._save_sample(tmp_path / "sample_0000.pt", 5, torch.tensor([0, 4, 0, 6, 0]))
        # Mismatched: text length 5 (pred length 5) but target durations length 7
        self._save_sample(tmp_path / "sample_0001.pt", 5, torch.tensor([0, 2, 0, 3, 0, 4, 0]))

        results = evaluate_durations(model, tmp_path)
        # Only the valid sample contributes to the metrics
        assert results["n_samples"] == 1
        assert results["mae_mean"] == pytest.approx(1.5)

    def test_none_durations_skipped(self, tmp_path):
        """A sample with durations=None is skipped before running the encoder."""
        model = self._make_stub_model({5: torch.tensor([1.0, 5.0, 1.0, 8.0, 1.0])})
        self._save_sample(tmp_path / "sample_0000.pt", 5, None)
        self._save_sample(tmp_path / "sample_0001.pt", 5, torch.tensor([0, 4, 0, 6, 0]))

        results = evaluate_durations(model, tmp_path)
        assert results["n_samples"] == 1
        # Encoder is only invoked for the sample that has durations
        assert len(model.calls) == 1

    def test_no_valid_samples_returns_error(self, tmp_path):
        """Zero valid samples yields the error dict instead of metrics."""
        model = self._make_stub_model({})
        self._save_sample(tmp_path / "sample_0000.pt", 5, None)

        results = evaluate_durations(model, tmp_path)
        assert results == {"error": "No valid samples found"}

    def test_single_speaker_encoder_gets_none_spks(self, tmp_path):
        """With n_spks == 1 the encoder is called with spks=None (no spk_emb lookup)."""
        model = self._make_stub_model({5: torch.tensor([1.0, 5.0, 1.0, 8.0, 1.0])}, n_spks=1)
        self._save_sample(tmp_path / "sample_0000.pt", 5, torch.tensor([0, 4, 0, 6, 0]), spk=3)

        evaluate_durations(model, tmp_path)
        assert len(model.calls) == 1
        assert model.calls[0]["spks"] is None


class TestTransferFromEnglish:
    """Tests for the English -> Japanese checkpoint vocab surgery script."""

    def _make_ckpt(self, path, n_vocab=178, n_channels=192, with_hparams=True):
        """Save a synthetic English checkpoint and return the in-memory dict."""
        ckpt = {
            "state_dict": {
                "encoder.emb.weight": torch.randn(n_vocab, n_channels),
                "decoder.some.weight": torch.randn(4, 4),
            },
        }
        if with_hparams:
            ckpt["hyper_parameters"] = {"n_vocab": n_vocab}
        torch.save(ckpt, path)
        return ckpt

    def _run_main(self, argv):
        """Run transfer_from_english.main() with an explicit argv list."""
        transfer_main(argv)

    def test_vocab_surgery_happy_path(self, tmp_path):
        """Embedding is resized to the new vocab; all other weights are preserved."""
        src = tmp_path / "en.ckpt"
        dst = tmp_path / "ja.ckpt"
        original = self._make_ckpt(src)

        self._run_main(["--source", str(src), "--target", str(dst), "--n-vocab-new", "55"])

        out = torch.load(dst, weights_only=True)
        assert out["state_dict"]["encoder.emb.weight"].shape == (55, 192)
        assert torch.equal(out["state_dict"]["decoder.some.weight"], original["state_dict"]["decoder.some.weight"])
        assert out["hyper_parameters"]["n_vocab"] == 55

    def test_missing_emb_key_raises_keyerror(self, tmp_path):
        """Checkpoint without encoder.emb.weight raises KeyError listing emb-like keys."""
        src = tmp_path / "en.ckpt"
        dst = tmp_path / "ja.ckpt"
        torch.save({"state_dict": {"encoder.token_emb.weight": torch.randn(178, 192)}}, src)

        with pytest.raises(KeyError) as excinfo:
            self._run_main(["--source", str(src), "--target", str(dst), "--n-vocab-new", "55"])
        msg = str(excinfo.value)
        assert "'encoder.emb.weight' not found" in msg
        assert "encoder.token_emb.weight" in msg
        assert not dst.exists()

    def test_n_channels_mismatch_raises_valueerror(self, tmp_path):
        """A --n-channels value conflicting with the checkpoint dim raises ValueError."""
        src = tmp_path / "en.ckpt"
        dst = tmp_path / "ja.ckpt"
        self._make_ckpt(src, n_channels=192)

        with pytest.raises(ValueError, match="does not match checkpoint"):
            self._run_main(
                ["--source", str(src), "--target", str(dst), "--n-vocab-new", "55", "--n-channels", "256"],
            )
        assert not dst.exists()

    def test_nonexistent_source_raises(self, tmp_path):
        """A missing --source checkpoint raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Source checkpoint not found"):
            self._run_main(
                ["--source", str(tmp_path / "missing.ckpt"), "--target", str(tmp_path / "out.ckpt")],
            )

    def test_missing_hyper_parameters_still_saves(self, tmp_path):
        """A checkpoint lacking hyper_parameters is still converted and saved."""
        src = tmp_path / "en.ckpt"
        dst = tmp_path / "ja.ckpt"
        self._make_ckpt(src, with_hparams=False)

        self._run_main(["--source", str(src), "--target", str(dst), "--n-vocab-new", "55"])

        out = torch.load(dst, weights_only=True)
        assert out["state_dict"]["encoder.emb.weight"].shape == (55, 192)
        assert "hyper_parameters" not in out


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

    def test_nonexistent_dir(self, tmp_path):
        """Non-existent directory returns exit code 1."""
        from check_training_health import main

        # tmp_path-based path for determinism (see test_data_dir_validation)
        ret = main(["--log-dir", str(tmp_path / "no_such_dir")])
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

    def test_empty_log_dir_returns_1(self, tmp_path):
        """Existing but empty log directory has issues and returns exit code 1."""
        from check_training_health import main

        ret = main(["--log-dir", str(tmp_path)])
        assert ret == 1

    def test_empty_tb_subdir_falls_back_to_root_events(self, tmp_path):
        """An empty tensorboard/ subdir must not hide root-level event files.

        check_health() prefers the tensorboard/ subdir, but when it exists and
        contains no event files, the search falls back to the whole log dir so
        root-level events are still counted (regression test for the false
        'No TensorBoard event files found' report).
        """
        (tmp_path / "checkpoints").mkdir()
        (tmp_path / "checkpoints" / "last.ckpt").touch()
        (tmp_path / "tensorboard").mkdir()  # exists but contains no event files
        (tmp_path / "events.out.tfevents.11111").touch()  # root-level events

        results = check_health(tmp_path)
        assert results["tensorboard_events"] == 1
        assert "No TensorBoard event files found" not in results["issues"]


class TestValidateJuliusMappingParseLab:
    """Tests for validate_julius_mapping.parse_lab_file line handling."""

    def test_three_and_one_token_lines_parsed(self, tmp_path):
        from validate_julius_mapping import parse_lab_file

        lab = tmp_path / "a.lab"
        lab.write_text("0.0000 0.1000 sil\n0.1000 0.2000 a\npau\n\n", encoding="utf-8")
        assert parse_lab_file(lab) == ["sil", "a", "pau"]

    def test_two_token_line_warns_instead_of_vanishing(self, tmp_path, capsys):
        """A 2-token line matches neither format; it must produce a warning, not silence."""
        from validate_julius_mapping import parse_lab_file

        lab = tmp_path / "b.lab"
        lab.write_text("0.0000 0.1000 sil\n0.2000 a\n0.3000 0.4000 o\n", encoding="utf-8")
        phonemes = parse_lab_file(lab)
        assert phonemes == ["sil", "o"]
        err = capsys.readouterr().err
        assert "Warning" in err
        assert "0.2000 a" in err
