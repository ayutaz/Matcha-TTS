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

echo "== [1/6] apt packages =="
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq julius perl git curl tmux unzip make build-essential

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
make test

echo ""
echo "setup complete. 次: docs/next-steps-plan.md Phase 1（データ準備）"
echo "  uv run python scripts/prepare_jvs.py --jvs-dir data/jvs_ver1 --output-dir data/jvs \\"
echo "    --julius-output-dir data/julius_work/wav --num-workers 8"
