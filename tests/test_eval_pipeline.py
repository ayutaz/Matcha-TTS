"""Tests for M5 evaluation pipeline."""
import json
import pytest
import torch
from pathlib import Path


class TestSynthesiseClampParameter:
    """Test clamp_boundary_blanks parameter in synthesise()."""

    def test_synthesise_signature_has_clamp_param(self):
        """synthesise()にclamp_boundary_blanksパラメータが存在"""
        import inspect
        from matcha.models.matcha_tts import MatchaTTS
        sig = inspect.signature(MatchaTTS.synthesise)
        assert "clamp_boundary_blanks" in sig.parameters

    def test_clamp_default_is_true(self):
        """clamp_boundary_blanksのデフォルトがTrue"""
        import inspect
        from matcha.models.matcha_tts import MatchaTTS
        sig = inspect.signature(MatchaTTS.synthesise)
        assert sig.parameters["clamp_boundary_blanks"].default is True

    def test_durations_in_return_dict(self):
        """synthesise()の戻り値にdurationsキーが含まれること"""
        import inspect
        from matcha.models.matcha_tts import MatchaTTS
        # ソースコードでdurationsが返されることを確認
        source = inspect.getsource(MatchaTTS.synthesise)
        assert '"durations"' in source or "'durations'" in source


class TestEvalTexts:
    """Test evaluation text file."""

    def test_texts_file_exists(self):
        """eval/texts_ja.txtが存在すること"""
        path = Path(__file__).resolve().parent.parent / "eval" / "texts_ja.txt"
        assert path.exists(), f"Expected {path} to exist"

    def test_texts_file_has_10_lines(self):
        """10文が含まれること"""
        path = Path(__file__).resolve().parent.parent / "eval" / "texts_ja.txt"
        lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 10

    def test_texts_are_japanese(self):
        """テキストが日本語であること（ひらがな/カタカナ/漢字を含む）"""
        path = Path(__file__).resolve().parent.parent / "eval" / "texts_ja.txt"
        lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        for line in lines:
            has_ja = any(
                ('\u3040' <= ch <= '\u309F') or  # ひらがな
                ('\u30A0' <= ch <= '\u30FF') or  # カタカナ
                ('\u4E00' <= ch <= '\u9FFF')     # 漢字
                for ch in line
            )
            assert has_ja, f"Not Japanese: {line}"


class TestGenerateEvalSamples:
    """Test generate_eval_samples.py."""

    def test_parse_speaker_range(self):
        """話者範囲パース"""
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        from generate_eval_samples import parse_speaker_range

        assert parse_speaker_range("0-99") == list(range(100))
        assert parse_speaker_range("0,5,10") == [0, 5, 10]
        assert parse_speaker_range("0-4") == [0, 1, 2, 3, 4]

    def test_dry_run_without_checkpoint(self, tmp_path):
        """チェックポイントなしでdry-runが動作"""
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        from generate_eval_samples import main

        text_file = tmp_path / "texts.txt"
        text_file.write_text("テスト\n", encoding="utf-8")
        output_dir = tmp_path / "output"

        ret = main([
            "--checkpoint", "/nonexistent/model.ckpt",
            "--text-file", str(text_file),
            "--output-dir", str(output_dir),
        ])
        assert ret == 0
        # metadata.jsonが生成される
        meta_path = output_dir / "metadata.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["status"] == "dry_run"

    def test_missing_text_file_returns_error(self):
        """テキストファイルなしでエラー"""
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        from generate_eval_samples import main

        ret = main([
            "--checkpoint", "dummy.ckpt",
            "--text-file", "/nonexistent/texts.txt",
            "--output-dir", "/tmp/out",
        ])
        assert ret == 1
