"""Tests for validate_precomputed_durations.py"""
import torch
import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from validate_precomputed_durations import validate, main


def _make_valid_pt(path, text_len=11, mel_len=25, with_dur=True):
    """ダミーの正常な.ptファイル作成"""
    text = torch.randint(0, 55, (text_len,), dtype=torch.int32)
    mel = torch.randn(80, mel_len)
    d = {"mel": mel, "text": text, "spk": 0, "cleaned_text": "test"}
    if with_dur:
        dur = torch.zeros(text_len, dtype=torch.int32)
        dur[1::2] = mel_len // (text_len // 2)  # 均等分配（概算）
        # 合計をmel_lenに合わせる
        diff = mel_len - dur.sum().item()
        if diff != 0 and dur[1] > 0:
            dur[1] += diff
        d["durations"] = dur
    torch.save(d, path)


class TestValidate:
    def test_all_valid(self, tmp_path):
        """全ファイル正常"""
        _make_valid_pt(tmp_path / "s1.pt")
        _make_valid_pt(tmp_path / "s2.pt")
        stats = validate(tmp_path, expect_durations=True)
        assert stats["total"] == 2
        assert len(stats["errors"]) == 0
        assert stats["with_durations"] == 2

    def test_missing_durations_detected(self, tmp_path):
        """durations欠如が検出されること"""
        _make_valid_pt(tmp_path / "s1.pt", with_dur=False)
        stats = validate(tmp_path, expect_durations=True)
        assert stats["without_durations"] == 1
        assert any("Missing" in msg for _, msg in stats["errors"])

    def test_duration_sum_mismatch(self, tmp_path):
        """duration合計 != mel長が検出されること"""
        text = torch.randint(0, 55, (11,), dtype=torch.int32)
        mel = torch.randn(80, 100)
        dur = torch.ones(11, dtype=torch.int32)  # sum=11 != 100
        torch.save({"mel": mel, "text": text, "spk": 0,
                     "cleaned_text": "t", "durations": dur}, tmp_path / "bad.pt")
        stats = validate(tmp_path, expect_durations=True)
        assert stats["dur_sum_mismatches"] == 1

    def test_negative_duration_detected(self, tmp_path):
        """負のduration値が検出されること"""
        text = torch.randint(0, 55, (5,), dtype=torch.int32)
        dur = torch.tensor([0, -1, 0, 5, 0], dtype=torch.int32)
        torch.save({"mel": torch.randn(80, 4), "text": text, "spk": 0,
                     "cleaned_text": "t", "durations": dur}, tmp_path / "neg.pt")
        stats = validate(tmp_path, expect_durations=True)
        assert stats["negative_durations"] == 1

    def test_length_mismatch_detected(self, tmp_path):
        """duration長 != text長が検出されること"""
        text = torch.randint(0, 55, (11,), dtype=torch.int32)
        dur = torch.ones(7, dtype=torch.int32)  # 7 != 11
        torch.save({"mel": torch.randn(80, 100), "text": text, "spk": 0,
                     "cleaned_text": "t", "durations": dur}, tmp_path / "len.pt")
        stats = validate(tmp_path, expect_durations=True)
        assert any("Duration len" in msg for _, msg in stats["errors"])

    def test_no_expect_durations(self, tmp_path):
        """expect_durations=Falseで、durationsなしでもエラーにならない"""
        _make_valid_pt(tmp_path / "s1.pt", with_dur=False)
        stats = validate(tmp_path, expect_durations=False)
        assert len(stats["errors"]) == 0

    def test_empty_directory(self, tmp_path):
        """空ディレクトリ"""
        stats = validate(tmp_path, expect_durations=True)
        assert stats["total"] == 0
        assert len(stats["errors"]) == 0


class TestMainCLI:
    def test_nonexistent_dir_returns_error(self):
        """存在しないディレクトリでreturn 1"""
        ret = main(["--pt-dir", "/nonexistent/path"])
        assert ret == 1

    def test_valid_dir_returns_0(self, tmp_path):
        """正常ファイルでreturn 0"""
        _make_valid_pt(tmp_path / "s1.pt")
        ret = main(["--pt-dir", str(tmp_path)])
        assert ret == 0
