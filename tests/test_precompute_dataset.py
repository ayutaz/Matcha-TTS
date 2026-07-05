"""Tests for precompute_dataset.py duration embedding."""

import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

# Import from scripts
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from precompute_dataset import load_duration, parse_filelist


class TestLoadDuration:
    def test_valid_duration(self, tmp_path):
        """正常な.npyファイルでint64 Tensorが返ること"""
        dur = np.array([0, 5, 0, 3, 0, 7, 0], dtype=np.int64)
        np.save(tmp_path / "jvs001_UTT001.npy", dur)
        result = load_duration(tmp_path, "jvs001", "UTT001", 7)
        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.int64
        assert len(result) == 7

    def test_missing_file_returns_none(self, tmp_path):
        """存在しない.npyでNoneが返ること"""
        result = load_duration(tmp_path, "jvs001", "MISSING", 7)
        assert result is None

    def test_length_mismatch_raises(self, tmp_path):
        """text長と異なるdurationでValueError"""
        dur = np.array([0, 5, 0, 3, 0], dtype=np.int64)
        np.save(tmp_path / "jvs001_UTT001.npy", dur)
        with pytest.raises(ValueError, match="Duration length mismatch"):
            load_duration(tmp_path, "jvs001", "UTT001", 7)

    def test_dtype_is_int64(self, tmp_path):
        """int64 npyがint64 tensorとして返ること（collateのtorch.longと一致）"""
        dur = np.array([0, 10, 0], dtype=np.int64)
        np.save(tmp_path / "jvs001_UTT001.npy", dur)
        result = load_duration(tmp_path, "jvs001", "UTT001", 3)
        assert result.dtype == torch.int64

    def test_duration_sum_mismatch_raises(self, tmp_path):
        """duration合計がmel_framesと一致しない場合にValueError"""
        dur = np.array([0, 5, 0, 3, 0], dtype=np.int64)  # sum=8
        np.save(tmp_path / "jvs001_UTT001.npy", dur)
        with pytest.raises(ValueError, match="Duration sum mismatch"):
            load_duration(tmp_path, "jvs001", "UTT001", 5, mel_frames=100)

    def test_duration_sum_valid(self, tmp_path):
        """duration合計がmel_framesと一致する場合に正常動作"""
        dur = np.array([0, 5, 0, 3, 0, 7, 0], dtype=np.int64)  # sum=15
        np.save(tmp_path / "jvs001_UTT001.npy", dur)
        result = load_duration(tmp_path, "jvs001", "UTT001", 7, mel_frames=15)
        assert isinstance(result, torch.Tensor)
        assert result.sum().item() == 15

    def test_duration_sum_none_skips_check(self, tmp_path):
        """mel_frames=Noneでsum検証がスキップされること"""
        dur = np.array([0, 5, 0, 3, 0], dtype=np.int64)  # sum=8
        np.save(tmp_path / "jvs001_UTT001.npy", dur)
        # mel_frames=None（デフォルト）ではsumチェックなし → 正常に返る
        result = load_duration(tmp_path, "jvs001", "UTT001", 5)
        assert isinstance(result, torch.Tensor)
        assert len(result) == 5


class TestParseFilelist:
    def test_basic(self, tmp_path):
        fl = tmp_path / "test.txt"
        fl.write_text("path/a.wav|0|hello\npath/b.wav|1|world\n")
        entries = parse_filelist(str(fl))
        assert len(entries) == 2
        assert entries[0] == ["path/a.wav", "0", "hello"]

    def test_empty_lines_skipped(self, tmp_path):
        fl = tmp_path / "test.txt"
        fl.write_text("path/a.wav|0|hello\n\npath/b.wav|1|world\n")
        entries = parse_filelist(str(fl))
        assert len(entries) == 2


class TestProcessSampleWithDuration:
    """process_sample()のduration統合テスト（@pytest.mark.slow -- wav生成+mel計算が必要）"""

    @pytest.mark.slow
    def test_pt_contains_durations_key(self, tmp_path):
        """durations_dir指定時に.ptにdurationsキーが含まれること"""
        # ダミーwav作成
        wav_dir = tmp_path / "wavs" / "jvs001"
        wav_dir.mkdir(parents=True)
        wav_path = wav_dir / "UTT001.wav"
        audio_data = np.random.randn(22050).astype(np.float32) * 0.1
        sf.write(str(wav_path), audio_data, 22050)

        # テキスト処理でtext長を事前計算
        from precompute_dataset import process_sample

        from matcha.text import text_to_sequence
        from matcha.utils.utils import intersperse

        text = "こんにちは"
        text_seq, _ = text_to_sequence(text, ["japanese_cleaners"], language="ja")
        text_interspersed = intersperse(text_seq, 0)
        text_len = len(text_interspersed)

        # melフレーム数を事前計算（duration sumと一致させるため）
        audio_tensor = torch.from_numpy(audio_data).unsqueeze(0)
        from matcha.utils.audio import mel_spectrogram as _mel_spec

        mel_frames = _mel_spec(audio_tensor, 1024, 80, 22050, 256, 1024, 0.0, 8000, center=False).squeeze().shape[-1]

        # ダミーduration作成（sumがmel_framesと一致するよう分配）
        dur_dir = tmp_path / "durations"
        dur_dir.mkdir()
        dur = np.zeros(text_len, dtype=np.int64)
        n_phonemes = len(dur[1::2])
        base = mel_frames // n_phonemes
        remainder = mel_frames % n_phonemes
        dur[1::2] = base
        dur[1 : 2 * remainder : 2] += 1  # 余りを先頭音素に分配
        np.save(dur_dir / "jvs001_UTT001.npy", dur)

        # 実行
        out_dir = tmp_path / "output"
        out_dir.mkdir()
        result = process_sample(str(wav_path), 0, text, out_dir, -6.55, 2.38, dur_dir)
        out_path, skipped = result
        assert not skipped

        # 検証
        data = torch.load(out_path, weights_only=True)
        assert "durations" in data
        assert len(data["durations"]) == len(data["text"])

    @pytest.mark.slow
    def test_pt_without_durations_dir(self, tmp_path):
        """durations_dir=None時にdurationsキーが含まれないこと"""
        wav_dir = tmp_path / "wavs" / "jvs001"
        wav_dir.mkdir(parents=True)
        wav_path = wav_dir / "UTT001.wav"
        sf.write(str(wav_path), np.random.randn(22050).astype(np.float32) * 0.1, 22050)

        from precompute_dataset import process_sample

        out_dir = tmp_path / "output"
        out_dir.mkdir()
        result = process_sample(str(wav_path), 0, "こんにちは", out_dir, -6.55, 2.38, None)

        # 後方互換: durations_dir=Noneの場合
        if isinstance(result, tuple):
            out_path, skipped = result
        else:
            out_path = result

        data = torch.load(out_path, weights_only=True)
        assert "durations" not in data

    @pytest.mark.slow
    def test_skip_missing_duration(self, tmp_path):
        """対応.npyがない場合にskipped=Trueが返ること"""
        wav_dir = tmp_path / "wavs" / "jvs001"
        wav_dir.mkdir(parents=True)
        wav_path = wav_dir / "UTT001.wav"
        sf.write(str(wav_path), np.random.randn(22050).astype(np.float32) * 0.1, 22050)

        from precompute_dataset import process_sample

        out_dir = tmp_path / "output"
        out_dir.mkdir()
        dur_dir = tmp_path / "empty_durations"
        dur_dir.mkdir()
        result = process_sample(str(wav_path), 0, "こんにちは", out_dir, -6.55, 2.38, dur_dir)
        out_path, skipped = result
        assert skipped
