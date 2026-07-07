#!/usr/bin/env bash
# A-1: kernel-level profiling of the jvs_aligned training step to locate the bottleneck.
#
# Runs 1 short epoch (35 batches) with torch.profiler active on rank 0, then prints a
# key_averages() table + exports a chrome trace to <output_dir>/profiler/trace_rank0.json.
# Zero quality risk: no weights are kept (EMA/early-stopping/checkpoints are off).
#
# Read the table to decide the A-2 / B-1 investment:
#   GEMM (addmm/mm/bmm) dominates      -> compute-bound, compile/SDPA give little
#   many tiny kernels, CPU >> CUDA     -> kernel-launch-bound  => A-2 regional compile
#   ncclDevKernel_AllReduce* large     -> comm-bound           => C-1 / batch size
#   copy_ / idle gaps at step edges    -> data-bound           => B-1 frame batching
#
# Usage:
#   bash scripts/profile_training.sh                                   # single GPU (device 0)
#   MATCHA_PROFILE_DEVICES="[0,1,2,3]" bash scripts/profile_training.sh  # 4-GPU DDP profile
set -euo pipefail

DEVICES="${MATCHA_PROFILE_DEVICES:-[0]}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# 4-GPU profile needs the same NCCL transport as real training (C-1)
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

echo "=== config dry-run (--cfg job) — abort here if compose fails, before spending GPU time ==="
uv run python matcha/train.py experiment=jvs_aligned_profile trainer.devices="${DEVICES}" --cfg job \
    | grep -E 'precision|fused|limit_train_batches|max_epochs|TorchProfiler|check_val' || true

echo "=== profiling run (devices=${DEVICES}) ==="
exec uv run python matcha/train.py experiment=jvs_aligned_profile trainer.devices="${DEVICES}"
