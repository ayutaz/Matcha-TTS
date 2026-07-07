#!/usr/bin/env bash
# Production launcher for jvs_aligned on 4x RTX 5090.
#
# Sets the NCCL / allocator env (C-1) that must NOT live in train.py (it is host/transport
# specific and would be wrong on an IB-equipped node). bf16-mixed + optimizer.fused=false are
# now the committed config default (see configs/experiment/jvs_aligned.yaml), so this launcher
# needs no precision override.
#
# Usage:
#   bash scripts/train_jvs_aligned.sh
#   bash scripts/train_jvs_aligned.sh ckpt_path=logs/.../last.ckpt   # resume (args pass through)
set -euo pipefail

# --- C-1: NCCL transport for RTX 5090 (GeForce -> GPU P2P disabled at driver level).
# NCCL already stages through host memory; making it explicit avoids probe hangs and is
# side-effect-free (P2P was never available). IB is absent on typical vast.ai boxes.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Uncomment once to confirm the transport (C-1 check): look for 'P2P/direct disabled' + 'via SHM'
# export NCCL_DEBUG=INFO

exec uv run python matcha/train.py \
    experiment=jvs_aligned \
    data.batch_size=32 data.num_workers=0 +data.preload_to_memory=true \
    test=false "$@"
