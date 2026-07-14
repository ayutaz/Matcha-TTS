"""Tests for M5 evaluation metric scripts."""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


class TestEvalDurationAccuracy:
    def test_compute_accuracy(self):
        from eval_duration_accuracy import compute_accuracy

        pred = [0, 5, 0, 10, 0, 3, 0]
        gt = [0, 6, 0, 9, 0, 4, 0]
        result = compute_accuracy(pred, gt)
        assert result is not None
        assert result["mae"] < 2.0
        assert result["rmse"] < 2.0

    def test_compute_accuracy_identical(self):
        from eval_duration_accuracy import compute_accuracy

        dur = [0, 5, 0, 10, 0, 3, 0, 7, 0]
        result = compute_accuracy(dur, dur)
        assert result["mae"] == 0.0
        assert result["rmse"] == 0.0
        assert result["pearson_r"] == pytest.approx(1.0, abs=0.01)

    def test_main_missing_dir(self, tmp_path):
        from eval_duration_accuracy import main

        assert main(["--pred-dir", str(tmp_path / "does_not_exist")]) == 1

    def test_load_predicted_durations(self, tmp_path):
        from eval_duration_accuracy import load_predicted_durations

        (tmp_path / "spk_000").mkdir()
        dur_json = {"predicted_durations": [0, 5, 0, 10, 0], "speaker_id": 0}
        (tmp_path / "spk_000" / "text_00_dur.json").write_text(json.dumps(dur_json))
        results = load_predicted_durations(tmp_path)
        assert len(results) == 1
        assert results[0]["durations"] == [0, 5, 0, 10, 0]

    def test_main_without_gt_dir_returns_1(self, tmp_path):
        from eval_duration_accuracy import main

        (tmp_path / "spk_000").mkdir()
        pred = {"predicted_durations": [0, 5, 0, 10, 0], "speaker_id": 0}
        (tmp_path / "spk_000" / "text_00_dur.json").write_text(json.dumps(pred), encoding="utf-8")

        assert main(["--pred-dir", str(tmp_path)]) == 1

    def test_main_computes_accuracy_over_pairs(self, tmp_path):
        """main() runs compute_accuracy over prediction/ground-truth pairs and reports real metrics."""
        from eval_duration_accuracy import main

        pred_dir = tmp_path / "pred"
        gt_dir = tmp_path / "gt"
        (pred_dir / "spk_000").mkdir(parents=True)
        (gt_dir / "spk_000").mkdir(parents=True)

        pred = {"predicted_durations": [0, 5, 0, 10, 0, 3, 0], "speaker_id": 0}
        gt = {"durations": [0, 6, 0, 9, 0, 4, 0]}
        (pred_dir / "spk_000" / "text_00_dur.json").write_text(json.dumps(pred), encoding="utf-8")
        (gt_dir / "spk_000" / "text_00_dur.json").write_text(json.dumps(gt), encoding="utf-8")

        output = tmp_path / "report" / "duration_accuracy.json"
        rc = main(["--pred-dir", str(pred_dir), "--gt-dir", str(gt_dir), "--output", str(output)])
        assert rc == 0

        report = json.loads(output.read_text(encoding="utf-8"))
        assert report["status"] == "complete"
        assert report["n_pairs"] == 1
        # Phoneme positions: pred [5, 10, 3] vs gt [6, 9, 4] -> MAE = RMSE = 1.0
        assert report["mae_mean"] == pytest.approx(1.0)
        assert report["rmse_mean"] == pytest.approx(1.0)
        assert report["per_file"][0]["mae"] == pytest.approx(1.0)

    def test_main_no_pairs_returns_1(self, tmp_path):
        """Predictions without matching ground-truth files yield exit code 1, not a fake report."""
        from eval_duration_accuracy import main

        pred_dir = tmp_path / "pred"
        gt_dir = tmp_path / "gt"
        (pred_dir / "spk_000").mkdir(parents=True)
        gt_dir.mkdir()
        pred = {"predicted_durations": [0, 5, 0, 10, 0], "speaker_id": 0}
        (pred_dir / "spk_000" / "text_00_dur.json").write_text(json.dumps(pred), encoding="utf-8")

        assert main(["--pred-dir", str(pred_dir), "--gt-dir", str(gt_dir)]) == 1


class TestComputeAccuracy:
    """Edge cases for eval_duration_accuracy.compute_accuracy."""

    def test_different_lengths_use_min_len_overlap(self):
        from eval_duration_accuracy import compute_accuracy

        # min_len = 5 -> phoneme positions: pred [4, 6] vs gt [5, 7]
        pred = [0, 4, 0, 6, 0, 8, 0, 12, 0]
        gt = [0, 5, 0, 7, 0]
        result = compute_accuracy(pred, gt)
        assert result["mae"] == pytest.approx(1.0)
        assert result["rmse"] == pytest.approx(1.0)

    @pytest.mark.parametrize(
        "pred,gt",
        [
            ([], []),
            ([3], [3]),
            ([5], [0, 7, 0]),  # min_len = 1 -> no phoneme positions
        ],
    )
    def test_too_short_returns_none(self, pred, gt):
        from eval_duration_accuracy import compute_accuracy

        assert compute_accuracy(pred, gt) is None

    def test_constant_gt_pearson_nan(self):
        from eval_duration_accuracy import compute_accuracy

        pred = [0, 1, 0, 2, 0, 3, 0, 4, 0]
        gt = [0, 5, 0, 5, 0, 5, 0, 5, 0]
        result = compute_accuracy(pred, gt)
        assert math.isnan(result["pearson_r"])
        # MAE/RMSE remain well-defined
        assert result["mae"] == pytest.approx(2.5)

    def test_all_zero_gt_relative_error_nan(self):
        from eval_duration_accuracy import compute_accuracy

        pred = [0, 4, 0, 6, 0, 8, 0]
        gt = [0, 0, 0, 0, 0, 0, 0]
        result = compute_accuracy(pred, gt)
        assert math.isnan(result["relative_error"])

    def test_partial_zero_gt_excludes_zero_positions(self):
        from eval_duration_accuracy import compute_accuracy

        # gt phonemes [0, 3]: only index 1 counts -> rel_err = |6 - 3| / 3 = 1.0
        pred = [0, 4, 0, 6, 0]
        gt = [0, 0, 0, 3, 0]
        result = compute_accuracy(pred, gt)
        assert result["relative_error"] == pytest.approx(1.0)


class TestDurJsonUtf8Loading:
    """UTF-8 regression tests for _dur.json loaders."""

    @staticmethod
    def _write_dur_json(path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def test_japanese_text_field_utf8(self, tmp_path):
        from eval_degeneration_rate import load_durations_from_json
        from eval_duration_accuracy import load_predicted_durations

        payload = {
            "text": "こんにちは、日本語のテスト文です。",
            "predicted_durations": [0, 5, 0, 10, 0],
            "speaker_id": 3,
        }
        self._write_dur_json(tmp_path / "spk_003" / "text_00_dur.json", payload)

        acc_results = load_predicted_durations(tmp_path)
        assert len(acc_results) == 1
        assert acc_results[0]["durations"] == [0, 5, 0, 10, 0]

        deg_results = load_durations_from_json(tmp_path)
        assert len(deg_results) == 1
        np.testing.assert_array_equal(deg_results[0]["durations"], [0, 5, 0, 10, 0])

    def test_ascii_content_loads(self, tmp_path):
        from eval_degeneration_rate import load_durations_from_json
        from eval_duration_accuracy import load_predicted_durations

        payload = {"text": "hello world", "predicted_durations": [0, 3, 0, 7, 0], "speaker_id": 1}
        self._write_dur_json(tmp_path / "spk_001" / "text_00_dur.json", payload)

        acc_results = load_predicted_durations(tmp_path)
        assert len(acc_results) == 1
        assert acc_results[0]["speaker_id"] == 1

        deg_results = load_durations_from_json(tmp_path)
        assert len(deg_results) == 1
        assert deg_results[0]["durations"].dtype == np.int64

    def test_skips_files_without_predicted_durations(self, tmp_path):
        from eval_degeneration_rate import load_durations_from_json
        from eval_duration_accuracy import load_predicted_durations

        self._write_dur_json(tmp_path / "spk_000" / "text_00_dur.json", {"speaker_id": 0})

        assert load_predicted_durations(tmp_path) == []
        assert load_durations_from_json(tmp_path) == []


class TestEvalDegenerationRate:
    def test_healthy_durations(self, tmp_path):
        from eval_degeneration_rate import (
            compute_degeneration_report,
            load_durations_from_json,
        )

        (tmp_path / "spk_000").mkdir()
        dur = {"predicted_durations": [0, 10, 0, 8, 0, 12, 0, 6, 0], "speaker_id": 0}
        (tmp_path / "spk_000" / "text_00_dur.json").write_text(json.dumps(dur))

        all_dur = load_durations_from_json(tmp_path)
        report = compute_degeneration_report(all_dur)
        assert report["degenerate_rate"] == 0.0

    def test_degenerate_durations(self, tmp_path):
        from eval_degeneration_rate import (
            compute_degeneration_report,
            load_durations_from_json,
        )

        (tmp_path / "spk_000").mkdir()
        dur = {"predicted_durations": [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0], "speaker_id": 0}
        (tmp_path / "spk_000" / "text_00_dur.json").write_text(json.dumps(dur))

        all_dur = load_durations_from_json(tmp_path)
        report = compute_degeneration_report(all_dur)
        assert report["degenerate_rate"] == 1.0

    def test_main_missing_dir(self, tmp_path):
        from eval_degeneration_rate import main

        assert main(["--pred-dir", str(tmp_path / "does_not_exist")]) == 1

    def test_aggregate_stats_hand_computed(self, tmp_path):
        from eval_degeneration_rate import (
            compute_degeneration_report,
            load_durations_from_json,
        )

        (tmp_path / "spk_000").mkdir()
        # File 1: blank0=3, phonemes [10, 8]
        d1 = {"predicted_durations": [3, 10, 0, 8, 5], "speaker_id": 0}
        (tmp_path / "spk_000" / "text_00_dur.json").write_text(json.dumps(d1), encoding="utf-8")
        # File 2: blank0=7, phonemes [1, 2, 6]
        d2 = {"predicted_durations": [7, 1, 0, 2, 0, 6, 0], "speaker_id": 0}
        (tmp_path / "spk_000" / "text_01_dur.json").write_text(json.dumps(d2), encoding="utf-8")

        report = compute_degeneration_report(load_durations_from_json(tmp_path))
        assert report["total_samples"] == 2
        # All phonemes: [10, 8, 1, 2, 6]
        assert report["phoneme_le1_frame_rate"] == pytest.approx(1 / 5)
        assert report["phoneme_le2_frame_rate"] == pytest.approx(2 / 5)
        assert report["phoneme_median_duration"] == pytest.approx(6.0)
        assert report["blank0_duration_mean"] == pytest.approx(5.0)
        assert report["blank0_duration_max"] == 7

    def test_mixed_corpus_fractional_rate(self, tmp_path):
        from eval_degeneration_rate import (
            compute_degeneration_report,
            load_durations_from_json,
        )

        (tmp_path / "spk_000").mkdir()
        degenerate = {"predicted_durations": [0, 1, 0, 1, 0, 1, 0], "speaker_id": 0}
        healthy = {"predicted_durations": [0, 10, 0, 8, 0, 12, 0], "speaker_id": 0}
        (tmp_path / "spk_000" / "text_00_dur.json").write_text(json.dumps(degenerate), encoding="utf-8")
        for i in range(1, 4):
            (tmp_path / "spk_000" / f"text_{i:02d}_dur.json").write_text(json.dumps(healthy), encoding="utf-8")

        report = compute_degeneration_report(load_durations_from_json(tmp_path))
        assert report["total_samples"] == 4
        assert report["degenerate_count"] == 1
        assert report["degenerate_rate"] == pytest.approx(0.25)

    def test_empty_input_error(self):
        from eval_degeneration_rate import compute_degeneration_report

        assert compute_degeneration_report([]) == {"error": "No data"}

    def test_main_empty_dir_returns_1(self, tmp_path):
        from eval_degeneration_rate import main

        assert main(["--pred-dir", str(tmp_path)]) == 1


class TestEvalMCD:
    def test_compute_mcd_identical(self):
        from eval_mcd import compute_mcd

        mfcc = np.random.randn(100, 13)
        assert compute_mcd(mfcc, mfcc) == 0.0

    def test_compute_mcd_different(self):
        from eval_mcd import compute_mcd

        a = np.random.randn(100, 13)
        b = np.random.randn(100, 13)
        mcd = compute_mcd(a, b)
        assert mcd > 0.0

    def test_compute_mcd_constant_distance_coefficient(self):
        from eval_mcd import compute_mcd

        # Every frame differs in exactly one coefficient by d -> per-frame L2 = d
        d = 1.5
        synth = np.zeros((10, 13))
        synth[:, 0] = d
        ref = np.zeros((10, 13))
        expected = (10.0 * math.sqrt(2.0) / math.log(10.0)) * d
        assert compute_mcd(synth, ref) == pytest.approx(expected)

    def test_main_missing_dir(self, tmp_path):
        from eval_mcd import main

        assert main(["--synth-dir", str(tmp_path / "does_not_exist")]) == 1

    def test_main_without_ref_dir_returns_1(self, tmp_path):
        from eval_mcd import main

        np.save(tmp_path / "text_00.npy", np.zeros((10, 13)))
        assert main(["--synth-dir", str(tmp_path)]) == 1

    def test_main_computes_mcd_over_pairs(self, tmp_path):
        """main() runs compute_mcd over synth/ref pairs and reports real metrics."""
        from eval_mcd import main

        synth_dir = tmp_path / "synth"
        ref_dir = tmp_path / "ref"
        (synth_dir / "spk_000").mkdir(parents=True)
        (ref_dir / "spk_000").mkdir(parents=True)

        # Every frame differs in exactly one coefficient by d -> MCD = coeff * d
        d = 1.5
        synth = np.zeros((10, 13))
        synth[:, 0] = d
        np.save(synth_dir / "spk_000" / "text_00.npy", synth)
        np.save(ref_dir / "spk_000" / "text_00.npy", np.zeros((10, 13)))
        # Unpaired synth file is skipped, not silently folded into the mean
        np.save(synth_dir / "spk_000" / "text_01.npy", synth)

        output = tmp_path / "report" / "mcd.json"
        rc = main(["--synth-dir", str(synth_dir), "--ref-dir", str(ref_dir), "--output", str(output)])
        assert rc == 0

        report = json.loads(output.read_text(encoding="utf-8"))
        assert report["status"] == "complete"
        assert report["n_files"] == 2
        assert report["n_pairs"] == 1
        assert report["n_skipped"] == 1
        expected = (10.0 * math.sqrt(2.0) / math.log(10.0)) * d
        assert report["mcd_mean"] == pytest.approx(expected)

    def test_main_handles_batched_mel_layout(self, tmp_path):
        """(1, 80, T) mels as saved by generate_eval_samples are aligned frame-wise."""
        from eval_mcd import main

        synth_dir = tmp_path / "synth"
        ref_dir = tmp_path / "ref"
        synth_dir.mkdir()
        ref_dir.mkdir()

        mel = np.random.randn(1, 80, 50)
        np.save(synth_dir / "text_00.npy", mel)
        np.save(ref_dir / "text_00.npy", mel)

        output = tmp_path / "mcd.json"
        rc = main(["--synth-dir", str(synth_dir), "--ref-dir", str(ref_dir), "--output", str(output)])
        assert rc == 0
        report = json.loads(output.read_text(encoding="utf-8"))
        assert report["mcd_mean"] == pytest.approx(0.0)

    def test_main_no_pairs_returns_1(self, tmp_path):
        from eval_mcd import main

        synth_dir = tmp_path / "synth"
        ref_dir = tmp_path / "ref"
        synth_dir.mkdir()
        ref_dir.mkdir()
        np.save(synth_dir / "text_00.npy", np.zeros((10, 13)))

        assert main(["--synth-dir", str(synth_dir), "--ref-dir", str(ref_dir)]) == 1


class TestEvalUTMOS:
    @staticmethod
    def _write_wav(path, n_samples=1600, sr=16000):
        import soundfile as sf

        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), np.zeros(n_samples, dtype=np.float32), sr)

    def test_main_missing_dir(self, tmp_path):
        from eval_utmos import main

        assert main(["--wav-dir", str(tmp_path / "does_not_exist")]) == 1

    def test_main_empty_dir_returns_1(self, tmp_path):
        from eval_utmos import main

        assert main(["--wav-dir", str(tmp_path)]) == 1

    def test_main_scores_wavs_via_torch_hub(self, tmp_path, monkeypatch):
        """main() loads the UTMOS predictor from torch.hub and scores every wav."""
        from eval_utmos import main

        self._write_wav(tmp_path / "wavs" / "spk_000" / "text_00.wav")
        self._write_wav(tmp_path / "wavs" / "spk_000" / "text_01.wav")

        hub_calls = []

        def fake_hub_load(repo, model, **kwargs):
            hub_calls.append((repo, model))

            def predictor(wave, sr):
                assert wave.ndim == 2  # (batch, samples)
                return torch.tensor([3.5])

            return predictor

        monkeypatch.setattr(torch.hub, "load", fake_hub_load)

        output = tmp_path / "report" / "utmos.json"
        rc = main(["--wav-dir", str(tmp_path / "wavs"), "--output", str(output)])
        assert rc == 0
        assert hub_calls == [("tarepan/SpeechMOS:v1.2.0", "utmos22_strong")]

        report = json.loads(output.read_text(encoding="utf-8"))
        assert report["status"] == "complete"
        assert report["n_wav_files"] == 2
        assert len(report["per_file"]) == 2
        assert report["utmos_mean"] == pytest.approx(3.5)
        assert report["utmos_std"] == pytest.approx(0.0)

    def test_main_scores_baseline_dir(self, tmp_path, monkeypatch):
        from eval_utmos import main

        self._write_wav(tmp_path / "wavs" / "text_00.wav")
        self._write_wav(tmp_path / "baseline" / "text_00.wav")

        monkeypatch.setattr(torch.hub, "load", lambda *a, **kw: lambda wave, sr: torch.tensor([4.0]))

        output = tmp_path / "utmos.json"
        rc = main(
            [
                "--wav-dir",
                str(tmp_path / "wavs"),
                "--baseline-dir",
                str(tmp_path / "baseline"),
                "--output",
                str(output),
            ]
        )
        assert rc == 0
        report = json.loads(output.read_text(encoding="utf-8"))
        assert report["baseline_utmos_mean"] == pytest.approx(4.0)

    def test_main_hub_load_failure_returns_1(self, tmp_path, monkeypatch):
        from eval_utmos import main

        self._write_wav(tmp_path / "text_00.wav")

        def broken_hub_load(*args, **kwargs):
            raise RuntimeError("no network")

        monkeypatch.setattr(torch.hub, "load", broken_hub_load)
        assert main(["--wav-dir", str(tmp_path)]) == 1


class TestEvalReport:
    def test_generate_empty_report(self, tmp_path):
        from generate_eval_report import generate_report

        report = generate_report(tmp_path)
        assert report["summary"]["status"] == "no_data"

    def test_generate_with_degeneration(self, tmp_path):
        from generate_eval_report import generate_report

        deg = {"degenerate_rate": 0.0, "phoneme_median_duration": 5.2}
        (tmp_path / "degeneration.json").write_text(json.dumps(deg))
        report = generate_report(tmp_path)
        assert report["degeneration"]["degenerate_rate"] == 0.0
        assert report["summary"]["status"] == "complete"

    def test_main_missing_dir(self, tmp_path):
        from generate_eval_report import main

        assert main(["--report-dir", str(tmp_path / "does_not_exist")]) == 1

    def test_degeneration_missing_keys_na_summary(self, tmp_path):
        from generate_eval_report import generate_report

        (tmp_path / "degeneration.json").write_text(json.dumps({"total_samples": 4}), encoding="utf-8")
        report = generate_report(tmp_path)
        assert report["summary"]["degeneration_rate"] == "N/A"
        assert report["summary"]["phoneme_median_duration"] == "N/A"
        assert report["summary"]["status"] == "complete"

    def test_only_mcd_complete(self, tmp_path):
        from generate_eval_report import generate_report

        (tmp_path / "mcd.json").write_text(json.dumps({"mcd_mean": 5.4}), encoding="utf-8")
        report = generate_report(tmp_path)
        assert report["mcd"] == {"mcd_mean": 5.4}
        assert report["degeneration"] is None
        assert report["summary"]["status"] == "complete"

    def test_main_default_output_path(self, tmp_path):
        from generate_eval_report import main

        deg = {"degenerate_rate": 0.1, "phoneme_median_duration": 4.0}
        (tmp_path / "degeneration.json").write_text(json.dumps(deg), encoding="utf-8")

        assert main(["--report-dir", str(tmp_path)]) == 0
        default_output = tmp_path / "eval_report.json"
        assert default_output.exists()
        report = json.loads(default_output.read_text(encoding="utf-8"))
        assert report["degeneration"]["degenerate_rate"] == 0.1

    def test_japanese_metric_json_roundtrip(self, tmp_path):
        from generate_eval_report import generate_report, main

        note = "日本語メモ：退化率は低い"
        deg = {"degenerate_rate": 0.0, "phoneme_median_duration": 5.0, "note": note}
        (tmp_path / "degeneration.json").write_text(json.dumps(deg, ensure_ascii=False), encoding="utf-8")

        report = generate_report(tmp_path)
        assert report["degeneration"]["note"] == note

        out = tmp_path / "out" / "eval_report.json"
        assert main(["--report-dir", str(tmp_path), "--output", str(out)]) == 0
        written = json.loads(out.read_text(encoding="utf-8"))
        assert written["degeneration"]["note"] == note


class TestPrepareABTest:
    @staticmethod
    def _make_model_dirs(tmp_path, a_names, b_names):
        a_dir = tmp_path / "a"
        b_dir = tmp_path / "b"
        for d, names in [(a_dir, a_names), (b_dir, b_names)]:
            (d / "spk_000").mkdir(parents=True)
            for name in names:
                (d / "spk_000" / name).touch()
        return a_dir, b_dir

    def test_select_pairs(self, tmp_path):
        from prepare_ab_test import select_pairs

        a_dir = tmp_path / "a"
        b_dir = tmp_path / "b"
        for d in [a_dir, b_dir]:
            (d / "spk_000").mkdir(parents=True)
            for i in range(5):
                (d / "spk_000" / f"text_{i:02d}.wav").touch()

        pairs = select_pairs(a_dir, b_dir, n_pairs=3)
        assert len(pairs) == 3
        assert all("A" in p and "B" in p for p in pairs)

    def test_main_missing_dir(self, tmp_path):
        from prepare_ab_test import main

        missing = str(tmp_path / "does_not_exist")
        assert main(["--model-a-dir", missing, "--model-b-dir", missing, "--output-dir", str(tmp_path / "out")]) == 1

    def test_blind_pair_paths_match_labels(self, tmp_path):
        from prepare_ab_test import select_pairs

        # model_a_dir holds the julius samples, model_b_dir the mas samples
        names = [f"text_{i:02d}.wav" for i in range(12)]
        a_dir, b_dir = self._make_model_dirs(tmp_path, names, names)

        pairs = select_pairs(a_dir, b_dir, n_pairs=12, seed=42)
        assert len(pairs) == 12
        # Seed 42 exercises both orders
        assert {p["A_is"] for p in pairs} == {"julius", "mas"}

        for pair in pairs:
            if pair["A_is"] == "julius":
                assert pair["B_is"] == "mas"
                assert Path(pair["A"]).is_relative_to(a_dir)
                assert Path(pair["B"]).is_relative_to(b_dir)
            else:
                assert pair["A_is"] == "mas"
                assert pair["B_is"] == "julius"
                assert Path(pair["A"]).is_relative_to(b_dir)
                assert Path(pair["B"]).is_relative_to(a_dir)
            # A and B always point at the same utterance
            assert Path(pair["A"]).name == Path(pair["B"]).name

    def test_same_seed_reproducible(self, tmp_path):
        from prepare_ab_test import select_pairs

        names = [f"text_{i:02d}.wav" for i in range(12)]
        a_dir, b_dir = self._make_model_dirs(tmp_path, names, names)

        first = select_pairs(a_dir, b_dir, n_pairs=6, seed=42)
        second = select_pairs(a_dir, b_dir, n_pairs=6, seed=42)
        assert first == second  # identical selection AND A/B order

    def test_different_seed_differs(self, tmp_path):
        from prepare_ab_test import select_pairs

        names = [f"text_{i:02d}.wav" for i in range(12)]
        a_dir, b_dir = self._make_model_dirs(tmp_path, names, names)

        assert select_pairs(a_dir, b_dir, n_pairs=6, seed=42) != select_pairs(a_dir, b_dir, n_pairs=6, seed=43)

    def test_n_pairs_clamped_to_available(self, tmp_path):
        from prepare_ab_test import select_pairs

        names = [f"text_{i:02d}.wav" for i in range(5)]
        a_dir, b_dir = self._make_model_dirs(tmp_path, names, names)

        pairs = select_pairs(a_dir, b_dir, n_pairs=50)
        assert len(pairs) == 5

    def test_disjoint_dirs_return_empty(self, tmp_path):
        from prepare_ab_test import select_pairs

        a_dir, b_dir = self._make_model_dirs(tmp_path, ["only_a.wav"], ["only_b.wav"])
        assert select_pairs(a_dir, b_dir, n_pairs=10) == []

    def test_files_in_only_one_dir_excluded(self, tmp_path):
        from prepare_ab_test import select_pairs

        a_dir, b_dir = self._make_model_dirs(
            tmp_path,
            ["text_00.wav", "text_01.wav", "text_02.wav"],
            ["text_01.wav", "text_02.wav", "text_03.wav"],
        )
        pairs = select_pairs(a_dir, b_dir, n_pairs=10)
        assert len(pairs) == 2
        assert {Path(p["A"]).name for p in pairs} == {"text_01.wav", "text_02.wav"}


class TestAnalyzeABTest:
    def test_analyze_results(self):
        from analyze_ab_test import analyze_results

        results = [{"choice": "A"}, {"choice": "A"}, {"choice": "A"}, {"choice": "equal"}]
        pairs = [
            {"A_is": "julius", "B_is": "mas"},
            {"A_is": "mas", "B_is": "julius"},
            {"A_is": "julius", "B_is": "mas"},
            {"A_is": "julius", "B_is": "mas"},
        ]
        report = analyze_results(results, pairs)
        # Row 0: choice=A, A_is=julius -> julius_preferred
        # Row 1: choice=A, A_is=mas -> mas_preferred
        # Row 2: choice=A, A_is=julius -> julius_preferred
        # Row 3: choice=equal -> neither
        assert report["julius_preferred"] == 2
        assert report["mas_preferred"] == 1
        assert report["equal"] == 1

    def test_unanimous_julius_significant(self):
        from analyze_ab_test import analyze_results

        results = [{"choice": "A"}] * 15
        pairs = [{"A_is": "julius", "B_is": "mas"}] * 15
        report = analyze_results(results, pairs)
        assert report["julius_preferred"] == 15
        assert report["julius_preference_rate"] == pytest.approx(1.0)
        assert report["p_value"] < 0.01
        assert report["significant"] is True

    def test_balanced_not_significant(self):
        from analyze_ab_test import analyze_results

        results = [{"choice": "A"}] * 16
        pairs = [{"A_is": "julius", "B_is": "mas"}] * 8 + [{"A_is": "mas", "B_is": "julius"}] * 8
        report = analyze_results(results, pairs)
        assert report["julius_preferred"] == 8
        assert report["mas_preferred"] == 8
        assert report["p_value"] == pytest.approx(1.0, abs=0.01)
        assert report["significant"] is False

    def test_all_equal_no_decisive(self):
        from analyze_ab_test import analyze_results

        results = [{"choice": "equal"}] * 5
        pairs = [{"A_is": "julius", "B_is": "mas"}] * 5
        report = analyze_results(results, pairs)
        assert report["julius_preferred"] == 0
        assert report["mas_preferred"] == 0
        assert report["equal"] == 5
        assert report["julius_preference_rate"] == 0.5
        assert report["p_value"] == 1.0
        assert report["significant"] is False

    @pytest.mark.parametrize(
        "pair,expected_julius,expected_mas",
        [
            ({"A_is": "mas", "B_is": "julius"}, 1, 0),
            ({"A_is": "julius", "B_is": "mas"}, 0, 1),
        ],
    )
    def test_choice_b_credited_correctly(self, pair, expected_julius, expected_mas):
        from analyze_ab_test import analyze_results

        report = analyze_results([{"choice": "B"}], [pair])
        assert report["julius_preferred"] == expected_julius
        assert report["mas_preferred"] == expected_mas

    def test_main_round_trip(self, tmp_path):
        from analyze_ab_test import main

        results = [{"choice": "A"}, {"choice": "B"}, {"choice": "equal"}]
        pairs = [
            {"A_is": "julius", "B_is": "mas"},
            {"A_is": "mas", "B_is": "julius"},
            {"A_is": "julius", "B_is": "mas"},
        ]
        results_path = tmp_path / "results.json"
        pairs_path = tmp_path / "pairs.json"
        output_path = tmp_path / "report" / "ab_test.json"
        results_path.write_text(json.dumps(results), encoding="utf-8")
        pairs_path.write_text(json.dumps(pairs), encoding="utf-8")

        rc = main(["--results", str(results_path), "--pairs", str(pairs_path), "--output", str(output_path)])
        assert rc == 0
        assert output_path.exists()
        report = json.loads(output_path.read_text(encoding="utf-8"))
        assert report["total_pairs"] == 3
        assert report["julius_preferred"] == 2
        assert report["mas_preferred"] == 0
        assert report["equal"] == 1

    def test_main_missing_input_returns_1(self, tmp_path):
        from analyze_ab_test import main

        pairs_path = tmp_path / "pairs.json"
        pairs_path.write_text(json.dumps([]), encoding="utf-8")
        assert main(["--results", str(tmp_path / "missing.json"), "--pairs", str(pairs_path)]) == 1

    def test_length_mismatch_raises(self):
        """results/pairs of different lengths must not be silently zip-truncated."""
        from analyze_ab_test import analyze_results

        results = [{"choice": "A"}] * 5
        pairs = [{"A_is": "julius", "B_is": "mas"}] * 3
        with pytest.raises(ValueError, match="length mismatch"):
            analyze_results(results, pairs)

    def test_main_length_mismatch_returns_1(self, tmp_path):
        from analyze_ab_test import main

        results_path = tmp_path / "results.json"
        pairs_path = tmp_path / "pairs.json"
        results_path.write_text(json.dumps([{"choice": "A"}] * 2), encoding="utf-8")
        pairs_path.write_text(json.dumps([{"A_is": "julius", "B_is": "mas"}]), encoding="utf-8")

        assert main(["--results", str(results_path), "--pairs", str(pairs_path)]) == 1
