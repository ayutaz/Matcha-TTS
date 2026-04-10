"""M1 duration output and matcha_tts.py use_precomputed_durations path integration tests.

Tests that int64 duration arrays produced by convert_julius_to_durations.py
are correctly consumed by generate_path() after float conversion, covering:
  - int64 -> float32 type conversion
  - blank (duration=0) rows in attention matrix
  - duration sum == mel length invariant
  - batch processing with padding
  - full .npy load -> generate_path pipeline
"""

import numpy as np
import pytest
import torch

from matcha.utils.model import generate_path, sequence_mask


def _make_attn_mask(x_lengths, y_lengths, max_x, max_y):
    """Build float attn_mask matching matcha_tts.py forward() convention.

    In the model, x_mask and y_mask are float tensors (encoder output dtype),
    so attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2) is also float.
    """
    x_mask = sequence_mask(x_lengths, max_x).unsqueeze(1).float()  # (B,1,T_x)
    y_mask = sequence_mask(y_lengths, max_y).unsqueeze(1).float()  # (B,1,T_y)
    attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)  # (B,1,T_x,T_y)
    return attn_mask


class TestDurationToGeneratePath:
    def test_int64_durations_produce_valid_attn(self):
        """M1 output int64 durations produce correct attn after float conversion."""
        # blank intersperse済みduration: [0, 10, 0, 8, 0, 12, 0]
        durations = torch.tensor([[0, 10, 0, 8, 0, 12, 0]], dtype=torch.int64)
        total_frames = 30  # 10+8+12

        # float変換（matcha_tts.py forward()での処理と同一）
        durations_f = durations.float()

        # attn_mask作成
        x_lengths = torch.tensor([7])
        y_lengths = torch.tensor([total_frames])
        attn_mask = _make_attn_mask(x_lengths, y_lengths, 7, total_frames)

        attn = generate_path(durations_f.squeeze(1), attn_mask.squeeze(1))

        # 検証
        assert attn.shape == (1, 7, 30)
        assert attn.sum() == total_frames  # 全フレームが割り当てられている
        assert (attn >= 0).all()  # 非負
        assert (attn <= 1).all()  # 0/1バイナリ

    def test_blank_zero_durations_produce_empty_rows(self):
        """Blank positions (even index) with duration=0 produce zero rows in attn."""
        durations = torch.tensor([[0, 5, 0, 5, 0]], dtype=torch.int64)
        total_frames = 10

        x_lengths = torch.tensor([5])
        y_lengths = torch.tensor([total_frames])
        attn_mask = _make_attn_mask(x_lengths, y_lengths, 5, total_frames)

        attn = generate_path(durations.float().squeeze(1), attn_mask.squeeze(1))

        # blank行(0,2,4)は全て0
        assert attn[0, 0, :].sum() == 0  # blank[0]
        assert attn[0, 2, :].sum() == 0  # blank[1]
        assert attn[0, 4, :].sum() == 0  # blank[2]
        # phoneme行(1,3)は非ゼロ
        assert attn[0, 1, :].sum() == 5
        assert attn[0, 3, :].sum() == 5

    def test_duration_sum_matches_mel_length(self):
        """Duration total matches mel length and attn is fully assigned."""
        # 実際のM1出力に近い形式
        durations = torch.tensor(
            [[0, 3, 0, 7, 0, 2, 0, 12, 0, 5, 0, 1, 0]], dtype=torch.int64
        )
        total = durations.sum().item()  # 30

        x_len = durations.shape[1]
        x_lengths = torch.tensor([x_len])
        y_lengths = torch.tensor([total])
        attn_mask = _make_attn_mask(x_lengths, y_lengths, x_len, total)

        attn = generate_path(durations.float().squeeze(1), attn_mask.squeeze(1))

        assert attn.sum() == total

    def test_batch_processing(self):
        """Batch of padded durations with different lengths is handled correctly."""
        # 2つの発話：長さの異なるduration
        dur1 = [0, 5, 0, 5, 0]  # 5+5=10 frames, text_len=5
        dur2 = [0, 3, 0, 7, 0, 2, 0]  # 3+7+2=12 frames, text_len=7

        # パディング
        max_text = 7
        max_mel = 12
        durations = torch.zeros(2, max_text, dtype=torch.int64)
        durations[0, :5] = torch.tensor(dur1)
        durations[1, :7] = torch.tensor(dur2)

        x_lengths = torch.tensor([5, 7])
        y_lengths = torch.tensor([10, 12])
        attn_mask = _make_attn_mask(x_lengths, y_lengths, max_text, max_mel)

        attn = generate_path(durations.float().squeeze(1), attn_mask.squeeze(1))

        assert attn.shape == (2, max_text, max_mel)

    def test_m1_npy_to_generate_path_pipeline(self, tmp_path):
        """Full pipeline: M1 .npy output -> torch.from_numpy -> generate_path."""
        # M1出力のシミュレーション
        duration_npy = np.array(
            [0, 10, 0, 5, 0, 15, 0, 8, 0, 2, 0], dtype=np.int64
        )
        npy_path = tmp_path / "test.npy"
        np.save(npy_path, duration_npy)

        # ロード → テンソル変換
        loaded = np.load(npy_path)
        durations = torch.from_numpy(loaded).unsqueeze(0)  # (1, 2N+1)
        total = durations.sum().item()

        x_len = durations.shape[1]
        x_lengths = torch.tensor([x_len])
        y_lengths = torch.tensor([total])
        attn_mask = _make_attn_mask(x_lengths, y_lengths, x_len, total)

        attn = generate_path(durations.float().squeeze(1), attn_mask.squeeze(1))

        assert attn.shape == (1, x_len, total)
        assert attn.sum().item() == total
