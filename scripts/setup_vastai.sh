#!/usr/bin/env bash
# vast.ai インスタンス初期化スクリプト（docs/next-steps-plan.md Phase 0）
#
# 使い方（インスタンスにssh後、tmux内で実行）:
#   export HF_TOKEN=hf_xxx   # writeスコープのトークン
#   curl -fsSL https://raw.githubusercontent.com/ayutaz/Matcha-TTS/feature/japanese-support/scripts/setup_vastai.sh | bash
# repo clone済みなら: bash scripts/setup_vastai.sh
#
# 環境変数で上書き可能:
#   MATCHA_BRANCH (default: feature/japanese-support)
#   MATCHA_REPO   (default: https://github.com/ayutaz/Matcha-TTS.git)
#   JVS_DATASET   (default: ayousanz/jvs-ver1-raw — HF privateデータセット)
set -euo pipefail

BRANCH="${MATCHA_BRANCH:-feature/japanese-support}"
REPO_URL="${MATCHA_REPO:-https://github.com/ayutaz/Matcha-TTS.git}"
JVS_DATASET="${JVS_DATASET:-ayousanz/jvs-ver1-raw}"

echo "== [0/6] connectivity check =="
# 一部のvast.aiホストはhuggingface.coをDNS/SNIレベルで遮断している（2026-07に遭遇）。
# HFはデータセット取得とcheckpointバックアップの生命線なので、重い処理の前に検査する
if ! curl -fsS --max-time 15 -o /dev/null "https://huggingface.co/api/models?limit=1"; then
    echo "ERROR: huggingface.co に接続できません。このホストはHFを遮断している可能性があります。"
    echo "       別ホストのインスタンスに乗り換えてください（vastai search offers で host_id が異なるもの）"
    exit 2
fi
echo "huggingface.co OK"

echo "== [1/6] apt packages =="
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq julius perl git curl tmux unzip make build-essential espeak-ng

echo "== [2/6] uv =="
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

echo "== [3/6] repository + dependencies =="
if [ ! -d Matcha-TTS ] && [ ! -f pyproject.toml ]; then
    git clone -b "$BRANCH" "$REPO_URL"
    cd Matcha-TTS
elif [ -d Matcha-TTS ]; then
    cd Matcha-TTS
fi
# 再実行時は最新コミットへ更新（一時的なネットワークエラーに備えて1回リトライ）
git fetch origin "$BRANCH" || { sleep 5; git fetch origin "$BRANCH"; }
git checkout "$BRANCH"
git merge --ff-only "origin/$BRANCH"
uv sync --all-groups
bash scripts/setup_julius.sh

echo "== [4/6] HF CLI =="
uv tool install "huggingface_hub[cli]" >/dev/null 2>&1 || true
export PATH="$HOME/.local/bin:$PATH"
if [ -n "${HF_TOKEN:-}" ]; then
    hf auth login --token "$HF_TOKEN" >/dev/null
fi
hf auth whoami || echo "WARNING: HF未ログイン。バックアップ前に hf auth login を実行してください"

echo "== [5/6] JVS corpus =="
if [ ! -d data/jvs_ver1 ]; then
    mkdir -p data
    if hf download "$JVS_DATASET" jvs_ver1.zip --repo-type dataset --local-dir /tmp/jvs_dl; then
        unzip -q /tmp/jvs_dl/jvs_ver1.zip -d data/   # zipルートは jvs_ver1/
        rm -rf /tmp/jvs_dl
        echo "JVS corpus -> data/jvs_ver1"
    else
        echo "WARNING: $JVS_DATASET のダウンロードに失敗。data/jvs_ver1 に手動配置してください"
    fi
fi

echo "== [6/6] verification =="
nvidia-smi || echo "WARNING: nvidia-smi failed（GPU未検出）"
julius -help >/dev/null 2>&1 && echo "julius OK"
df -h /dev/shm
PHONEMIZER_ESPEAK_LIBRARY=/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1 make test

echo "== [opt] GPU runtime tuning (C-1 / C-2, opt-in) =="
# C-2: persistence mode keeps the driver resident -> lower kernel-launch latency + steadier
# clocks. Reversible and safe. The power limit is applied ONLY when MATCHA_POWER_LIMIT is set,
# so the default training path stays at the stock 575W (byte-identical). All guarded so a
# restricted host never aborts setup (`set -e` safe via `&& ... || ...`).
nvidia-smi -pm 1 >/dev/null 2>&1 && echo "persistence mode ON" || echo "persistence mode unavailable (host restriction)"
if [ -n "${MATCHA_POWER_LIMIT:-}" ]; then
    nvidia-smi -pl "$MATCHA_POWER_LIMIT" >/dev/null 2>&1 \
        && echo "power limit -> ${MATCHA_POWER_LIMIT}W" \
        || echo "power limit set failed (host restriction); continuing at default"
else
    echo "power limit: default 575W (set MATCHA_POWER_LIMIT=450 to cap; keep A-1/A-2 before/after on the SAME value)"
fi
nvidia-smi --query-gpu=index,persistence_mode,power.limit,power.default_limit --format=csv 2>/dev/null || true
# C-1: RTX 5090 has no GPU P2P (GeForce driver) -> NCCL stages via host SHM. The production
# launcher scripts/train_jvs_aligned.sh exports NCCL_P2P_DISABLE=1 / NCCL_IB_DISABLE=1 so NCCL
# never probes P2P (side-effect-free; P2P was never available). Verify the transport by running
# training once with NCCL_DEBUG=INFO and confirming 'P2P/direct disabled' + 'via SHM' in the log.
# Building nccl-tests is off by default (needs nvcc/libnccl-dev, usually absent in torch images).
if [ "${MATCHA_BUILD_NCCL_TESTS:-0}" = "1" ]; then
    echo "building nccl-tests (MATCHA_BUILD_NCCL_TESTS=1)..."
    NCCL_HOME="$(python -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"lib"))' 2>/dev/null || echo /usr)"
    { git clone --depth 1 https://github.com/NVIDIA/nccl-tests /tmp/nccl-tests \
        && make -C /tmp/nccl-tests NCCL_HOME="$NCCL_HOME" >/dev/null 2>&1 \
        && echo "nccl-tests -> /tmp/nccl-tests/build (run: build/all_reduce_perf -b 8 -e 128M -f 2 -g 4)"; } \
        || echo "nccl-tests build failed (nvcc/libnccl-dev missing); use NCCL_DEBUG=INFO instead"
fi

echo ""
echo "setup complete. 次: docs/next-steps-plan.md Phase 1（データ準備）"
echo "  uv run python scripts/prepare_jvs.py --jvs-dir data/jvs_ver1 --output-dir data/jvs \\"
echo "    --julius-output-dir data/julius_work/wav --num-workers 8"
