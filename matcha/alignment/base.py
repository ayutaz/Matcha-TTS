"""アライナー出力の抽象基底クラス。

将来的にJulius以外のアライナー（MFA, whisper-align等）に
切り替える際のインターフェース統一を提供する。

使用例:
    from matcha.alignment.base import BaseAlignerOutput, JuliusOutput

    # Julius実行結果のラップ（target音素列はセグメントと1:1対応が必要）
    output = JuliusOutput.from_lab_file(Path("utterance.lab"))
    durations = output.to_durations(target_phonemes, sample_rate=22050, hop_length=256)
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# 無声化母音 (pyopenjtalk) → 有声形 (Julius)
_DEVOICED_TO_VOICED = {"A": "a", "I": "i", "U": "u", "E": "e", "O": "o"}


def _corresponds(target_ph: str, julius_ph: str) -> bool:
    """ターゲット音素がマッピング済みJulius音素と対応するか判定する。"""
    if target_ph == julius_ph:
        return True
    if _DEVOICED_TO_VOICED.get(target_ph) == julius_ph:
        return True
    # ヴ: Julius音響モデルにvが無いためバ行で整列される（normalize_vu_kana参照）
    if target_ph == "v" and julius_ph == "b":
        return True
    # ^ = utterance start, $ = declarative end, ? = interrogative end
    if target_ph in {"^", "$", "?"} and julius_ph == "sil":
        return True
    return target_ph == "_" and julius_ph == "pau"


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
        total_mel_frames: int | None = None,
    ) -> np.ndarray:
        """ターゲット音素列に対応するduration配列を生成する。

        Args:
            target_phonemes: pyopenjtalk等のターゲット音素列
            sample_rate: メルスペクトログラムのサンプリングレート
            hop_length: ホップ長
            total_mel_frames: メルフレーム総数。指定時は合計がこの値と
                一致するようduration配列を調整する

        Returns:
            target_phonemesと同じ長さのduration配列 (dtype=int64)

        Raises:
            ValueError: target_phonemesにアライナー出力と対応付けられない
                音素が含まれる場合
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
        """HTK形式（100ns整数）またはfloat秒形式の.labファイルから生成する。

        Julius segmentation-kitはfloat秒形式を出力する。タイムスタンプの
        フォーマットは行ごとに自動判別する（小数点の有無）。タイムスタンプが
        数値として解釈できない行は警告を出してスキップする。
        """
        from matcha.alignment.constants import JULIUS_TIME_UNIT

        segments = []
        with open(lab_path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                try:
                    # 小数点があればfloat秒、なければHTK 100ns整数単位
                    if "." in parts[0] or "." in parts[1]:
                        start = float(parts[0])
                        end = float(parts[1])
                    else:
                        start = int(parts[0]) / JULIUS_TIME_UNIT
                        end = int(parts[1]) / JULIUS_TIME_UNIT
                except ValueError:
                    logger.warning(
                        "Skipping invalid line %d in %s: %s",
                        line_num,
                        lab_path,
                        line.strip(),
                    )
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
        total_mel_frames: int | None = None,
    ) -> np.ndarray:
        """Julius出力からtarget音素列に対応するduration配列を生成。

        target_phonemesがJuliusセグメントと1:1対応する場合のみ変換する
        （完全一致、無声化母音A/I/U/E/O↔小文字母音、^/$/?↔sil、_↔pau）。
        韻律記号やblank挿入を含むpyopenjtalk完全列との対応付けには
        scripts/convert_julius_to_durations.pyのalign_julius_with_pyopenjtalk
        + build_duration_array_with_blanksを使用すること。

        Raises:
            ValueError: target_phonemesがセグメントと1:1対応しない場合
            KeyError: Julius音素がマッピングテーブルに存在しない場合
        """
        from matcha.alignment.constants import FRAME_ADJUSTMENT_WARN_THRESHOLD
        from matcha.text.julius_to_pyopenjtalk import map_julius_sequence

        if len(target_phonemes) != len(self._segments):
            raise ValueError(
                f"target_phonemes has {len(target_phonemes)} phonemes but this output has "
                f"{len(self._segments)} segments. to_durations() requires a 1:1 correspondence; "
                "use scripts/convert_julius_to_durations.py for full pyopenjtalk alignment "
                "(prosody symbols, blank intersperse)."
            )

        julius_mapped = map_julius_sequence(self.get_phonemes())
        for idx, (target_ph, julius_ph) in enumerate(zip(target_phonemes, julius_mapped)):
            if not _corresponds(target_ph, julius_ph):
                raise ValueError(
                    f"target_phonemes[{idx}]='{target_ph}' does not correspond to Julius "
                    f"phoneme '{julius_ph}'. to_durations() requires a 1:1 correspondence; "
                    "use scripts/convert_julius_to_durations.py for full pyopenjtalk alignment."
                )

        durations = []
        for start, end, _ph in self._segments:
            start_frame = round(start * sample_rate / hop_length)
            end_frame = round(end * sample_rate / hop_length)
            durations.append(max(0, end_frame - start_frame))

        arr = np.array(durations, dtype=np.int64)
        if len(arr) == 0:
            return arr

        # 隣接セグメント間のギャップ/オーバーラップ検出（フレームの欠落・重複）
        span_frames = round(self._segments[-1][1] * sample_rate / hop_length) - round(
            self._segments[0][0] * sample_rate / hop_length
        )
        if int(arr.sum()) != span_frames:
            logger.warning(
                "Segment gaps/overlaps detected in %s: duration sum %d != segment span %d frames",
                self.aligner_name,
                int(arr.sum()),
                span_frames,
            )

        # 合計フレーム数の調整（scripts/convert_julius_to_durations.pyと同じ不変条件）
        if total_mel_frames is not None:
            diff = total_mel_frames - int(arr.sum())
            if abs(diff) >= FRAME_ADJUSTMENT_WARN_THRESHOLD:
                logger.warning(
                    "Large frame adjustment: diff=%d (total_mel=%d, duration_sum=%d)",
                    diff,
                    total_mel_frames,
                    int(arr.sum()),
                )
            if diff != 0:
                nonzero = np.nonzero(arr)[0]
                if len(nonzero) > 0:
                    last = nonzero[-1]
                    arr[last] = max(0, arr[last] + diff)
                elif diff > 0:
                    arr[-1] = diff

        return arr

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

    def to_durations(self, target_phonemes, sample_rate=22050, hop_length=256, total_mel_frames=None):
        raise NotImplementedError

    @property
    def aligner_name(self) -> str:
        return "mfa"
