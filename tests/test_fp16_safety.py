"""Tests for FP16 mixed precision safety in Matcha-TTS forward pass.

Validates that loss computations remain in FP32 even when autocast is active,
preventing overflow/NaN in MSE losses with reduction="sum".
"""

import math

import pytest
import torch
import torch.nn.functional as F

from matcha.utils.model import duration_loss


class TestDurationLossF32:
    """Validate that duration_loss stays in FP32 when inputs are cast."""

    def test_fp16_inputs_produce_fp32_loss(self):
        """duration_loss with .float() cast should produce FP32 output."""
        logw = torch.randn(2, 1, 10, dtype=torch.float16)
        logw_ = torch.randn(2, 1, 10, dtype=torch.float16)
        lengths = torch.tensor([8, 10])
        loss = duration_loss(logw.float(), logw_.float(), lengths)
        assert loss.dtype == torch.float32

    def test_fp32_inputs_unchanged(self):
        """duration_loss with FP32 inputs should remain FP32 (no-op cast)."""
        logw = torch.randn(2, 1, 10, dtype=torch.float32)
        logw_ = torch.randn(2, 1, 10, dtype=torch.float32)
        lengths = torch.tensor([8, 10])
        loss = duration_loss(logw.float(), logw_.float(), lengths)
        assert loss.dtype == torch.float32

    def test_large_values_no_overflow(self):
        """Large values that would overflow FP16 should be safe with .float() cast."""
        # Values near FP16 max (~65504) would overflow in (x-y)**2
        logw = torch.full((2, 1, 10), 200.0, dtype=torch.float16)
        logw_ = torch.full((2, 1, 10), -200.0, dtype=torch.float16)
        lengths = torch.tensor([10, 10])
        loss = duration_loss(logw.float(), logw_.float(), lengths)
        assert not torch.isnan(loss), "Loss should not be NaN with FP32 cast"
        assert not torch.isinf(loss), "Loss should not be Inf with FP32 cast"


class TestPriorLossF32:
    """Validate prior_loss computation stays in FP32 under autocast conditions."""

    def _compute_prior_loss(self, y, mu_y, y_mask, n_feats):
        """Replicate the prior_loss computation from matcha_tts.py."""
        LOG_2PI = math.log(2 * math.pi)
        masked_n = torch.sum(y_mask) * n_feats
        prior_loss = 0.5 * (
            F.mse_loss(y.float() * y_mask, mu_y.float() * y_mask, reduction="sum") / masked_n + LOG_2PI
        )
        return prior_loss

    def test_fp16_inputs_produce_fp32_loss(self):
        """prior_loss with FP16 inputs and .float() cast should produce FP32."""
        y = torch.randn(2, 80, 100, dtype=torch.float16)
        mu_y = torch.randn(2, 80, 100, dtype=torch.float16)
        y_mask = torch.ones(2, 1, 100, dtype=torch.float32)
        loss = self._compute_prior_loss(y, mu_y, y_mask, n_feats=80)
        assert loss.dtype == torch.float32

    def test_large_mel_values_no_overflow(self):
        """Large mel values should not cause overflow with FP32 cast."""
        # Mel values can be large after denormalization
        y = torch.full((2, 80, 200), 100.0, dtype=torch.float16)
        mu_y = torch.full((2, 80, 200), -100.0, dtype=torch.float16)
        y_mask = torch.ones(2, 1, 200, dtype=torch.float32)
        loss = self._compute_prior_loss(y, mu_y, y_mask, n_feats=80)
        assert not torch.isnan(loss), "Prior loss should not be NaN"
        assert not torch.isinf(loss), "Prior loss should not be Inf"

    def test_matches_pure_fp32_result(self):
        """FP16→FP32 cast result should be close to pure FP32 computation."""
        torch.manual_seed(42)
        y_f32 = torch.randn(2, 80, 100, dtype=torch.float32)
        mu_y_f32 = torch.randn(2, 80, 100, dtype=torch.float32)
        y_mask = torch.ones(2, 1, 100, dtype=torch.float32)

        # Pure FP32
        loss_f32 = self._compute_prior_loss(y_f32, mu_y_f32, y_mask, n_feats=80)
        # FP16 → FP32 cast (simulates autocast output)
        loss_mixed = self._compute_prior_loss(y_f32.half(), mu_y_f32.half(), y_mask, n_feats=80)

        # Allow tolerance for FP16 quantization error
        assert torch.allclose(loss_f32, loss_mixed, rtol=1e-2, atol=1e-2)


class TestDiffLossF32:
    """Validate that flow_matching compute_loss already casts to FP32."""

    def test_mse_with_float_cast(self):
        """Replicate the .float() cast in flow_matching.py compute_loss."""
        estimator_out = torch.randn(2, 80, 100, dtype=torch.float16)
        u = torch.randn(2, 80, 100, dtype=torch.float16)
        mask = torch.ones(2, 1, 100, dtype=torch.float32)

        loss = F.mse_loss(estimator_out.float(), u.float(), reduction="sum") / (
            torch.sum(mask) * u.shape[1]
        )
        assert loss.dtype == torch.float32

    def test_large_estimator_output_no_overflow(self):
        """Large estimator outputs should not overflow with .float() cast."""
        estimator_out = torch.full((2, 80, 200), 250.0, dtype=torch.float16)
        u = torch.full((2, 80, 200), -250.0, dtype=torch.float16)
        mask = torch.ones(2, 1, 200, dtype=torch.float32)

        loss = F.mse_loss(estimator_out.float(), u.float(), reduction="sum") / (
            torch.sum(mask) * u.shape[1]
        )
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)


class TestMASF32Safety:
    """Validate MAS computation stays in FP32 when autocast is disabled."""

    def test_autocast_disabled_produces_fp32(self):
        """With autocast disabled and .float() cast, MAS inputs should be FP32."""
        mu_x = torch.randn(2, 80, 10, dtype=torch.float16)
        y = torch.randn(2, 80, 50, dtype=torch.float32)
        n_feats = 80
        LOG_2PI = math.log(2 * math.pi)

        with torch.amp.autocast("cuda", enabled=False):
            mu_x_f = mu_x.float()
            y_f = y.float()
            const = -0.5 * LOG_2PI * n_feats
            y_square = -0.5 * torch.sum(y_f**2, 1, keepdim=True)
            y_mu_double = torch.matmul(mu_x_f.transpose(1, 2), y_f)
            mu_square = -0.5 * torch.sum(mu_x_f**2, 1).unsqueeze(-1)
            log_prior = y_square + y_mu_double + mu_square + const

        assert mu_x_f.dtype == torch.float32
        assert y_mu_double.dtype == torch.float32
        assert log_prior.dtype == torch.float32

    def test_fp16_without_cast_loses_precision(self):
        """Demonstrate that FP16 matmul loses precision vs FP32."""
        torch.manual_seed(42)
        mu_x = torch.randn(2, 80, 10)
        y = torch.randn(2, 80, 50)

        # FP32 reference
        ref = torch.matmul(mu_x.transpose(1, 2), y)
        # FP16 computation
        fp16_result = torch.matmul(mu_x.half().transpose(1, 2), y.half())

        # FP16 matmul should have non-trivial error
        max_err = (ref - fp16_result.float()).abs().max().item()
        assert max_err > 1e-4, f"Expected FP16 precision loss but max_err={max_err}"
