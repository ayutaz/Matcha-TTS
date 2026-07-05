"""Tests for M5 evaluation metric scripts."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

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

    def test_main_missing_dir(self):
        from eval_duration_accuracy import main

        assert main(["--pred-dir", "/nonexistent"]) == 1

    def test_load_predicted_durations(self, tmp_path):
        from eval_duration_accuracy import load_predicted_durations

        (tmp_path / "spk_000").mkdir()
        dur_json = {"predicted_durations": [0, 5, 0, 10, 0], "speaker_id": 0}
        (tmp_path / "spk_000" / "text_00_dur.json").write_text(json.dumps(dur_json))
        results = load_predicted_durations(tmp_path)
        assert len(results) == 1
        assert results[0]["durations"] == [0, 5, 0, 10, 0]


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

    def test_main_missing_dir(self):
        from eval_degeneration_rate import main

        assert main(["--pred-dir", "/nonexistent"]) == 1


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

    def test_main_missing_dir(self):
        from eval_mcd import main

        assert main(["--synth-dir", "/nonexistent"]) == 1


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

    def test_main_missing_dir(self):
        from generate_eval_report import main

        assert main(["--report-dir", "/nonexistent"]) == 1


class TestPrepareABTest:
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

    def test_main_missing_dir(self):
        from prepare_ab_test import main

        assert main(["--model-a-dir", "/nonexist", "--model-b-dir", "/nonexist", "--output-dir", "/tmp/out"]) == 1


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
