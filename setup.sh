#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="$SCRIPT_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
python3 -m venv "$VENV_DIR" || { echo "venv 모듈이 필요합니다. (Ubuntu: sudo apt-get install -y python3-venv)"; exit 1; }
fi

. "$VENV_DIR/bin/activate"

pip install --upgrade pip setuptools wheel

# PyTorch Packages
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchtext==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# Other Packages
pip install -r requirements.txt

if command -v apt-get >/dev/null 2>&1; then
sudo apt-get update && sudo apt-get install -y ffmpeg libgl1
fi

echo "Environment setup complete."