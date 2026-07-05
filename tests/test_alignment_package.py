"""matcha/alignment/ パッケージのテスト。"""

from pathlib import Path

import numpy as np
import pytest


class TestAlignmentPackageImports:
    def test_import_phoneme_mapping(self):
        """音素マッピングがパッケージからインポート可能"""
        from matcha.alignment import JULIUS_TO_PYOPENJTALK, map_julius_phoneme

        assert isinstance(JULIUS_TO_PYOPENJTALK, dict)
        assert map_julius_phoneme("a") == "a"

    def test_import_metrics(self):
        """品質指標がパッケージからインポート可能"""
        from matcha.alignment import compute_corpus_stats, is_degenerate

        assert callable(is_degenerate)
        assert callable(compute_corpus_stats)

    def test_import_constants(self):
        """定数がインポート可能"""
        from matcha.alignment.constants import HOP_LENGTH, JULIUS_TIME_UNIT, SAMPLE_RATE

        assert SAMPLE_RATE == 22050
        assert HOP_LENGTH == 256
        assert JULIUS_TIME_UNIT == 10_000_000

    def test_import_all_exports(self):
        """__all__ に列挙された全シンボルがインポート可能"""
        import matcha.alignment as alignment

        for name in alignment.__all__:
            assert hasattr(alignment, name), f"{name} is listed in __all__ but not importable"

    def test_import_prosody_symbols(self):
        """PROSODY_SYMBOLSがインポート可能"""
        from matcha.alignment import PROSODY_SYMBOLS

        assert isinstance(PROSODY_SYMBOLS, set)
        assert "^" in PROSODY_SYMBOLS

    def test_import_map_julius_sequence(self):
        """map_julius_sequenceがインポート可能で正しく動作"""
        from matcha.alignment import map_julius_sequence

        result = map_julius_sequence(["silB", "k", "a", "silE"])
        assert result == ["sil", "k", "a", "sil"]

    def test_import_get_unmapped_phonemes(self):
        """get_unmapped_phonemesがインポート可能で正しく動作"""
        from matcha.alignment import get_unmapped_phonemes

        assert get_unmapped_phonemes(["a", "k"]) == set()
        assert get_unmapped_phonemes(["a", "UNKNOWN"]) == {"UNKNOWN"}


class TestConstants:
    def test_frame_duration_consistency(self):
        """SAMPLE_RATE / HOP_LENGTH が期待通り"""
        from matcha.alignment.constants import HOP_LENGTH, SAMPLE_RATE

        frames_per_second = SAMPLE_RATE / HOP_LENGTH
        assert 86 <= frames_per_second <= 87  # approx 86.13

    def test_julius_constants(self):
        """Julius関連定数が正しい値"""
        from matcha.alignment.constants import JULIUS_SAMPLE_RATE, JULIUS_TIME_UNIT

        assert JULIUS_SAMPLE_RATE == 16000
        assert JULIUS_TIME_UNIT == 10_000_000

    def test_mel_constants(self):
        """メルスペクトログラム関連定数"""
        from matcha.alignment.constants import N_FEATS, N_FFT

        assert N_FFT == 1024
        assert N_FEATS == 80

    def test_jvs_mel_stats(self):
        """JVSメル統計量の定数"""
        from matcha.alignment.constants import JVS_MEL_MEAN, JVS_MEL_STD

        assert pytest.approx(-6.550095) == JVS_MEL_MEAN
        assert pytest.approx(2.383771) == JVS_MEL_STD

    def test_vocab_sizes(self):
        """語彙サイズ定数"""
        from matcha.alignment.constants import N_VOCAB_EN, N_VOCAB_JA

        assert N_VOCAB_JA == 55
        assert N_VOCAB_EN == 178

    def test_frame_adjustment_threshold(self):
        """端数調整警告閾値"""
        from matcha.alignment.constants import FRAME_ADJUSTMENT_WARN_THRESHOLD

        assert FRAME_ADJUSTMENT_WARN_THRESHOLD == 10


class TestBaseAlignerOutput:
    def _make_lab_file(self, tmp_path: Path) -> Path:
        """テスト用の.labファイルを作成するヘルパー"""
        lab_path = tmp_path / "test.lab"
        # Julius HTK形式: start end phoneme (100ns単位)
        lab_path.write_text(
            "0 30000000 silB\n"  # 0.0s - 3.0s
            "30000000 50000000 k\n"  # 3.0s - 5.0s
            "50000000 80000000 a\n"  # 5.0s - 8.0s
            "80000000 100000000 silE\n",  # 8.0s - 10.0s
            encoding="utf-8",
        )
        return lab_path

    def test_julius_output_from_lab(self, tmp_path):
        """JuliusOutput.from_lab_file が正しくパースできること"""
        from matcha.alignment.base import JuliusOutput

        lab_path = self._make_lab_file(tmp_path)
        output = JuliusOutput.from_lab_file(lab_path)

        assert len(output.get_phonemes()) == 4
        assert len(output.get_timings()) == 4

    def test_julius_output_get_phonemes(self, tmp_path):
        """get_phonemes() が音素リストを返すこと"""
        from matcha.alignment.base import JuliusOutput

        lab_path = self._make_lab_file(tmp_path)
        output = JuliusOutput.from_lab_file(lab_path)

        phonemes = output.get_phonemes()
        assert phonemes == ["silB", "k", "a", "silE"]

    def test_julius_output_get_timings(self, tmp_path):
        """get_timings() が(start,end)リストを返すこと"""
        from matcha.alignment.base import JuliusOutput

        lab_path = self._make_lab_file(tmp_path)
        output = JuliusOutput.from_lab_file(lab_path)

        timings = output.get_timings()
        assert len(timings) == 4
        # silB: 0.0s - 3.0s
        assert timings[0] == pytest.approx((0.0, 3.0))
        # k: 3.0s - 5.0s
        assert timings[1] == pytest.approx((3.0, 5.0))

    def test_julius_output_to_durations(self, tmp_path):
        """to_durations() がnp.ndarray(int64)を返すこと"""
        from matcha.alignment.base import JuliusOutput

        lab_path = self._make_lab_file(tmp_path)
        output = JuliusOutput.from_lab_file(lab_path)

        durations = output.to_durations(
            target_phonemes=["sil", "k", "a", "sil"],
            sample_rate=22050,
            hop_length=256,
        )

        assert isinstance(durations, np.ndarray)
        assert durations.dtype == np.int64
        assert len(durations) == 4
        assert all(d >= 0 for d in durations)
        # Total duration should be 10s worth of frames
        expected_total_frames = round(10.0 * 22050 / 256)
        assert sum(durations) == expected_total_frames

    def test_julius_output_empty_lab(self, tmp_path):
        """空の.labファイルを処理できること"""
        from matcha.alignment.base import JuliusOutput

        lab_path = tmp_path / "empty.lab"
        lab_path.write_text("", encoding="utf-8")
        output = JuliusOutput.from_lab_file(lab_path)

        assert output.get_phonemes() == []
        assert output.get_timings() == []
        durations = output.to_durations([], sample_rate=22050, hop_length=256)
        assert len(durations) == 0

    def test_julius_output_malformed_lines(self, tmp_path):
        """不正な行をスキップすること"""
        from matcha.alignment.base import JuliusOutput

        lab_path = tmp_path / "malformed.lab"
        lab_path.write_text(
            "0 30000000 silB\nbad line\n\n30000000 50000000 k\n",
            encoding="utf-8",
        )
        output = JuliusOutput.from_lab_file(lab_path)

        assert output.get_phonemes() == ["silB", "k"]

    def test_mfa_output_not_implemented(self):
        """MFAOutputがNotImplementedErrorを出すこと"""
        from matcha.alignment.base import MFAOutput

        with pytest.raises(NotImplementedError, match="MFA support"):
            MFAOutput(Path("/dummy/path.TextGrid"))

    def test_julius_aligner_name(self, tmp_path):
        """aligner_nameが'julius'を返すこと"""
        from matcha.alignment.base import JuliusOutput

        lab_path = self._make_lab_file(tmp_path)
        output = JuliusOutput.from_lab_file(lab_path)

        assert output.aligner_name == "julius"

    def test_julius_output_direct_construction(self):
        """直接コンストラクタからJuliusOutputを生成できること"""
        from matcha.alignment.base import JuliusOutput

        segments = [
            (0.0, 1.0, "silB"),
            (1.0, 1.5, "a"),
            (1.5, 2.0, "silE"),
        ]
        output = JuliusOutput(segments)

        assert output.get_phonemes() == ["silB", "a", "silE"]
        assert output.aligner_name == "julius"
