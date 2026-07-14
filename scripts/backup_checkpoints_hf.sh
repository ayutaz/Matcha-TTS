#!/usr/bin/env bash
# 学習中のcheckpointを定期的にHF privateリポジトリへアップロードする常駐スクリプト
# （docs/next-steps-plan.md §4.3。vast.aiインスタンス消滅への保険）
#
# 使い方（tmuxの別ペインで起動しっぱなしにする）:
#   bash scripts/backup_checkpoints_hf.sh                       # jvs_aligned を3時間ごと
#   bash scripts/backup_checkpoints_hf.sh jvs_fast              # ベースライン学習用
#   bash scripts/backup_checkpoints_hf.sh jvs_aligned ayousanz/matcha-tts-jvs-ja 3600
set -u

EXP="${1:-jvs_aligned}"
REPO="${2:-ayousanz/matcha-tts-jvs-ja}"
INTERVAL="${3:-10800}"   # 秒（default: 3時間）

echo "[backup] experiment=$EXP repo=$REPO interval=${INTERVAL}s"
while true; do
    RUN_DIR=$(ls -td "logs/train/$EXP/runs/"* 2>/dev/null | head -1)
    CKPT="${RUN_DIR:-}/checkpoints/last.ckpt"
    if [ -n "${RUN_DIR:-}" ] && [ -f "$CKPT" ]; then
        if hf upload "$REPO" "$CKPT" "$EXP/last.ckpt" >/dev/null 2>&1; then
            echo "[backup] $(date -u +%FT%TZ) uploaded $CKPT"
        else
            echo "[backup] $(date -u +%FT%TZ) upload FAILED（次サイクルで再試行）"
        fi
    else
        echo "[backup] $(date -u +%FT%TZ) checkpointがまだ無い（$EXP）"
    fi
    sleep "$INTERVAL"
done
