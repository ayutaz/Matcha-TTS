#!/bin/bash
# Julius segmentation-kit setup script.
# Clones the segmentation-kit repository into tools/ directory.
# Idempotent: re-running does not overwrite existing installation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
TOOLS_DIR="${PROJECT_DIR}/tools"
SEGKIT_DIR="${TOOLS_DIR}/segmentation-kit"

if [ -d "${SEGKIT_DIR}" ]; then
    echo "segmentation-kit already exists at ${SEGKIT_DIR}"
    exit 0
fi

mkdir -p "${TOOLS_DIR}"
echo "Cloning julius-speech/segmentation-kit into ${SEGKIT_DIR} ..."
git clone https://github.com/julius-speech/segmentation-kit.git "${SEGKIT_DIR}"

# Verify the julius binary is available
if ! command -v julius &> /dev/null; then
    echo ""
    echo "WARNING: julius command not found in PATH."
    echo "Install via one of:"
    echo "  Ubuntu/Debian: sudo apt-get install julius"
    echo "  From source:   https://github.com/julius-speech/julius"
    echo ""
fi

echo "Setup complete: ${SEGKIT_DIR}"
