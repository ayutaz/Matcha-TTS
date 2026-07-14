#!/bin/bash
# Setup /dev/shm cache for Matcha-TTS JVS preprocessing
# Copies source data and intermediate files to tmpfs for fast I/O
#
# Usage:
#   bash scripts/setup_shm_cache.sh [--source-only | --full | --cleanup]
#
# Options:
#   --source-only  Copy only JVS source wavs (3.5GB)
#   --full         Copy source wavs + julius_work (6GB total)
#   --cleanup      Remove all cached data from /dev/shm
#   (default)      Show current cache status

set -euo pipefail

# リポジトリルート（env MATCHA_ROOT で上書き可能。デフォルトはこのスクリプトの親ディレクトリ）
MATCHA_ROOT="${MATCHA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

SHM_BASE="/dev/shm"
JVS_WAVS_SRC="${MATCHA_ROOT}/data/jvs/wavs"
JVS_WAVS_CACHE="${SHM_BASE}/jvs_wavs"
JULIUS_WORK_SRC="${MATCHA_ROOT}/data/julius_work"
JULIUS_WORK_CACHE="${SHM_BASE}/julius_work"
PRECOMPUTED_ALIGNED_SRC="${MATCHA_ROOT}/data/jvs_precomputed_aligned"
PRECOMPUTED_ALIGNED_CACHE="${SHM_BASE}/jvs_precomputed_aligned"

show_status() {
    echo "=== /dev/shm Cache Status ==="
    echo ""
    df -h /dev/shm | tail -1 | awk '{printf "  Total: %s  Used: %s  Available: %s  Usage: %s\n", $2, $3, $4, $5}'
    echo ""
    echo "  Cached directories:"
    for dir in "${JVS_WAVS_CACHE}" "${JULIUS_WORK_CACHE}" "${PRECOMPUTED_ALIGNED_CACHE}" "${SHM_BASE}/jvs_precomputed"; do
        if [ -d "$dir" ]; then
            size=$(du -sh "$dir" 2>/dev/null | cut -f1)
            count=$(find "$dir" -type f 2>/dev/null | wc -l)
            echo "    $dir: ${size} (${count} files)"
        fi
    done
    echo ""
}

cache_source() {
    echo "=== Caching JVS source wavs to /dev/shm ==="
    if [ -d "$JVS_WAVS_CACHE" ]; then
        echo "  Already cached: $JVS_WAVS_CACHE"
        du -sh "$JVS_WAVS_CACHE"
    else
        if [ ! -d "$JVS_WAVS_SRC" ]; then
            echo "  ERROR: Source not found: $JVS_WAVS_SRC"
            exit 1
        fi
        echo "  Copying $JVS_WAVS_SRC -> $JVS_WAVS_CACHE ..."
        time cp -r "$JVS_WAVS_SRC" "$JVS_WAVS_CACHE"
        echo "  Done."
        du -sh "$JVS_WAVS_CACHE"
    fi
}

cache_julius() {
    echo "=== Caching Julius work directory to /dev/shm ==="
    if [ -d "$JULIUS_WORK_CACHE" ]; then
        echo "  Already cached: $JULIUS_WORK_CACHE"
        du -sh "$JULIUS_WORK_CACHE"
    else
        if [ ! -d "$JULIUS_WORK_SRC" ]; then
            echo "  Julius work not found (will be created during pipeline): $JULIUS_WORK_SRC"
            mkdir -p "$JULIUS_WORK_CACHE/wav" "$JULIUS_WORK_CACHE/durations"
            echo "  Created empty: $JULIUS_WORK_CACHE"
        else
            echo "  Copying $JULIUS_WORK_SRC -> $JULIUS_WORK_CACHE ..."
            time cp -r "$JULIUS_WORK_SRC" "$JULIUS_WORK_CACHE"
            echo "  Done."
            du -sh "$JULIUS_WORK_CACHE"
        fi
    fi
}

cache_precomputed() {
    echo "=== Caching precomputed aligned .pt to /dev/shm ==="
    if [ -d "$PRECOMPUTED_ALIGNED_CACHE" ]; then
        echo "  Already cached: $PRECOMPUTED_ALIGNED_CACHE"
        du -sh "$PRECOMPUTED_ALIGNED_CACHE"
    else
        if [ -d "$PRECOMPUTED_ALIGNED_SRC" ]; then
            echo "  Copying $PRECOMPUTED_ALIGNED_SRC -> $PRECOMPUTED_ALIGNED_CACHE ..."
            time cp -r "$PRECOMPUTED_ALIGNED_SRC" "$PRECOMPUTED_ALIGNED_CACHE"
            echo "  Done."
            du -sh "$PRECOMPUTED_ALIGNED_CACHE"
        else
            echo "  Source not found (will be created during pipeline): $PRECOMPUTED_ALIGNED_SRC"
            mkdir -p "$PRECOMPUTED_ALIGNED_CACHE/train" "$PRECOMPUTED_ALIGNED_CACHE/val"
        fi
    fi
}

cleanup() {
    echo "=== Cleaning up /dev/shm cache ==="
    for dir in "$JVS_WAVS_CACHE" "$JULIUS_WORK_CACHE"; do
        if [ -d "$dir" ]; then
            echo "  Removing $dir ..."
            rm -rf "$dir"
            echo "  Done."
        fi
    done
    echo "  Note: Keeping precomputed .pt caches (needed for training)"
    show_status
}

# Parse arguments
case "${1:-status}" in
    --source-only)
        cache_source
        show_status
        ;;
    --full)
        cache_source
        cache_julius
        cache_precomputed
        show_status
        ;;
    --cleanup)
        cleanup
        ;;
    status|--status|*)
        show_status
        ;;
esac
