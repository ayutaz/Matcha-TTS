"""アライナー出力の抽象基底クラス。

将来的にJulius以外のアライナー（MFA, whisper-align等）に
切り替える際のインターフェース統一を提供する。

使用例:
    from matcha.alignment.base import BaseAlignerOutput, JuliusOutput

    # Julius実行結果のラップ
    output = JuliusOutput.from_lab_file(Path("utterance.lab"))
    durations = output.to_durations(pyopenjtalk_phonemes, sr=22050, hop=256)
"""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class BaseAlignerOutput(ABC):
    """アライナー出力の抽象基底クラス。

    全てのアライナー実装はこのクラスを継承し、
    to_durations() メソッドでMatcha-TTS互換のduration配列を返す。
    """

    @abstractmethod
    def get_phonemes(self) -> list[str]:
        """アライナーが出力した音素列を返す。"""
        ...

    @abstractmethod
    def get_timings(self) -> list[tuple[float, float]]:
        """各音素の (start_sec, end_sec) タプルリストを返す。"""
        ...

    @abstractmethod
    def to_durations(
        self,
        target_phonemes: list[str],
        sample_rate: int,
        hop_length: int,
    ) -> np.ndarray:
        """ターゲット音素列に対応するduration配列を生成する。

        Args:
            target_phonemes: pyopenjtalk等のターゲット音素列
            sample_rate: メルスペクトログラムのサンプリングレート
            hop_length: ホップ長

        Returns:
            duration配列 (dtype=int64)
        """
        ...

    @property
    @abstractmethod
    def aligner_name(self) -> str:
        """アライナーの名前（ログ出力用）。"""
        ...


class JuliusOutput(BaseAlignerOutput):
    """Julius segmentation-kitの出力をラップするクラス。"""

    def __init__(self, segments: list[tuple[float, float, str]]):
        """
        Args:
            segments: [(start_sec, end_sec, phoneme), ...] のリスト
        """
        self._segments = segments

    @classmethod
    def from_lab_file(cls, lab_path: Path) -> "JuliusOutput":
        """HTK形式の.labファイルから生成する。"""
        from matcha.alignment.constants import JULIUS_TIME_UNIT

        segments = []
        with open(lab_path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                try:
                    start = int(parts[0]) / JULIUS_TIME_UNIT
                    end = int(parts[1]) / JULIUS_TIME_UNIT
                except ValueError:
                    continue
                segments.append((start, end, parts[2]))
        return cls(segments)

    def get_phonemes(self) -> list[str]:
        return [seg[2] for seg in self._segments]

    def get_timings(self) -> list[tuple[float, float]]:
        return [(seg[0], seg[1]) for seg in self._segments]

    def to_durations(
        self,
        target_phonemes: list[str],
        sample_rate: int = 22050,
        hop_length: int = 256,
    ) -> np.ndarray:
        """Julius出力からtarget音素列に対応するduration配列を生成。

        NOTE: 実際の変換ロジックはscripts/convert_julius_to_durations.pyに実装済み。
        このメソッドは将来的なリファクタリングで統合予定。
        現時点では基本的なフレーム変換のみ実装。
        """
        durations = []
        for start, end, _ph in self._segments:
            start_frame = round(start * sample_rate / hop_length)
            end_frame = round(end * sample_rate / hop_length)
            durations.append(max(0, end_frame - start_frame))

        return np.array(durations, dtype=np.int64)

    @property
    def aligner_name(self) -> str:
        return "julius"


class MFAOutput(BaseAlignerOutput):
    """Montreal Forced Alignerの出力をラップするクラス（将来実装用スタブ）。"""

    def __init__(self, textgrid_path: Path):
        self._path = textgrid_path
        raise NotImplementedError("MFA support is planned for future implementation")

    def get_phonemes(self) -> list[str]:
        raise NotImplementedError

    def get_timings(self) -> list[tuple[float, float]]:
        raise NotImplementedError

    def to_durations(self, target_phonemes, sample_rate=22050, hop_length=256):
        raise NotImplementedError

    @property
    def aligner_name(self) -> str:
        return "mfa"
