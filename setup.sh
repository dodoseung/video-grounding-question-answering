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

# VidSTG Dataset
# Download VidSTG dataset (Place the dataset in data/vidstg/)
# Download annotations from: https://huggingface.co/Gstar666/TASTVG/resolve/main/data.tar
# Download videos from: https://disk.pku.edu.cn/link/AA93DEAF3BBC694E52ACC5A23A9DC3D03B

# Pretrained models
# Download InternVideo2.5-Chat-8B for QA (Place the model files in checkpoints/qa/InternVideo2_5_Chat_8B/)
# Download from: https://huggingface.co/OpenGVLab/InternVideo2_5-Chat-8B

# RoBERTa base
# Download from https://huggingface.co/FacebookAI/roberta-base
# Place in checkpoints/pretrained/roberta-base/

# ResNet101 (optional, for different backbone)
wget https://zenodo.org/record/4721981/files/pretrained_resnet101_checkpoint.pth -P checkpoints/pretrained/

# Swin Tiny (optional, for different backbone)
wget https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_tiny_patch244_window877_kinetics400_1k.pth -P checkpoints/pretrained/

if command -v apt-get >/dev/null 2>&1; then
sudo apt-get update && sudo apt-get install -y ffmpeg libgl1
fi

echo "Environment setup complete."