# VGQA: Video Grounding and Question Answering

비디오 기반 시공간 객체 로컬라이제이션 및 질의응답 프레임워크

## Features

- Spatio-Temporal Video Grounding: 자연어 쿼리 기반 시공간 객체 탐지
- Video Question Answering: InternVideo2.5-Chat-8B 기반 비디오 QA
- CLI 및 Python API 지원
- FastAPI 웹 인터페이스

## Installation

```bash
# 환경 설정
source setup.sh

# 또는 수동 설치
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 체크포인트 다운로드

```bash
# 디렉토리 생성
mkdir -p checkpoints/{grounding,qa,pretrained/roberta-base}

# Grounding 모델: checkpoints/grounding/vidstg.pth
# InternVideo2.5-Chat-8B: checkpoints/qa/InternVideo2_5_Chat_8B/
# RoBERTa: checkpoints/pretrained/roberta-base/
# ResNet101: checkpoints/pretrained/pretrained_resnet101_checkpoint.pth
# Swin: checkpoints/pretrained/swin_tiny_patch244_window877_kinetics400_1k.pth
```

## Usage

### Python API

```python
from vgqa.inference import grounding, qa

# Grounding
result = grounding.predict(
    video_path="videos/sample.mp4",
    query="a person jumping",
    cfg_path="configs/grounding_vidstg.yaml",
    ckpt_path="checkpoints/grounding/vidstg.pth"
)

# QA
answer = qa.predict(
    video_path="videos/sample.mp4",
    question="What is happening?",
    model_dir="checkpoints/qa/InternVideo2_5_Chat_8B"
)
```

### CLI

```bash
# Grounding
python tools/infer_grounding.py \
    --video videos/sample.mp4 \
    --query "a dancing girl" \
    --cfg configs/grounding_vidstg.yaml \
    --ckpt checkpoints/grounding/vidstg.pth

# QA
python tools/infer_qa.py \
    --video videos/sample.mp4 \
    --question "What is happening?" \
    --model-dir checkpoints/qa/InternVideo2_5_Chat_8B

# Training
python tools/train.py --config-file configs/grounding_vidstg.yaml

# Evaluation
python tools/evaluate.py --config-file configs/grounding_vidstg.yaml
```

### Web Interface

```bash
python app/server.py
# http://localhost:8000/app/static/index.html
```

## Project Structure

```
VGQA/
├── vgqa/                    # 메인 패키지
│   ├── config/              # 설정 관리
│   ├── core/                # 모델 아키텍처
│   │   ├── grounding_net.py # Grounding Net
│   │   ├── vision/          # ResNet101/Swin Tiny
│   │   ├── language/        # RoBERTa/LSTM
│   │   ├── decoder/         # Transformer decoder
│   │   └── loss.py          # VideoSTGLoss
│   ├── data/                # 데이터 처리
│   ├── inference/           # 추론 API
│   │   ├── grounding.py
│   │   └── qa.py
│   ├── training/            # 학습 유틸리티
│   └── utils/               # 공통 유틸리티
├── tools/                   # CLI 도구
│   ├── train.py
│   ├── evaluate.py
│   ├── infer_grounding.py
│   └── infer_qa.py
├── app/                     # FastAPI 웹 앱
│   ├── server.py
│   └── static/
├── configs/                 # 설정 파일
├── data/vidstg/             # VidSTG 데이터셋
├── checkpoints/             # 모델 체크포인트
├── output/                  # 학습 출력
└── videos/                  # 데모 비디오
```

## Models

### Grounding
- Input: Video + 텍스트 쿼리
- Output: 시간 구간 (start/end) + 프레임별 bounding box
- Architecture: ResNet101/Swin + RoBERTa + Transformer decoder

### QA (InternVideo2.5-Chat-8B)
- Input: Video + 질문
- Output: 자연어 답변
- 8B 파라미터 멀티모달 LLM

## Training & Evaluation

```bash
# 학습
python tools/train.py --config-file configs/grounding_vidstg.yaml

# 평가
python tools/evaluate.py --config-file configs/grounding_vidstg.yaml

# TensorBoard 모니터링
tensorboard --logdir output/vidstg/tensorboard
```

## Configuration

주요 설정: [configs/grounding_vidstg.yaml](configs/grounding_vidstg.yaml)

```yaml
MODEL:
  VISION_BACKBONE:
    NAME: resnet101
  TEXT_ENCODER:
    NAME: roberta
    NAME_OR_PATH: checkpoints/pretrained/roberta-base
  VIDEO_SWIN:
    MODEL_NAME: video_swin_t_p4w7
  VSTG:
    QUERY_DIM: 512
    DEC_LAYERS: 6

INPUT:
  RESOLUTION: 224
  NUM_FRAMES: 64

SOLVER:
  MAX_EPOCH: 11
  BASE_LR: 0.0001
  BATCH_SIZE: 1

DATASETS:
  TRAIN: ["vidstg_train"]
  TEST: ["vidstg_test"]
```

## API Reference

### Grounding

```python
from vgqa.inference.grounding import predict

result = predict(
    video_path="video.mp4",
    query="person running",
    cfg_path="configs/grounding_vidstg.yaml",
    ckpt_path="checkpoints/grounding/vidstg.pth"
)
# Returns: {"temporal": {"start", "end", "score"}, "tube": [{"frame", "bbox", "score"}]}
```

### QA

```python
from vgqa.inference.qa import predict

answer = predict(
    video_path="video.mp4",
    question="What is happening?",
    model_dir="checkpoints/qa/InternVideo2_5_Chat_8B",
    num_frames=32,
    bound=(2.0, 5.0)  # Optional: focus on time range
)
# Returns: {"answer": str}
```

## Dataset Setup (VidSTG)

```bash
mkdir -p data/vidstg/{videos,annos,data_cache}

# VidSTG 비디오 및 어노테이션 다운로드
# data/vidstg/videos/     - 비디오 파일
# data/vidstg/annos/      - train.json, test.json
```

## Dependencies

주요 패키지 ([requirements.txt](requirements.txt)):
- PyTorch 1.9+
- transformers 4.37.2
- decord 0.6.0
- opencv-python 4.10.0.84
- yacs, einops, timm
- FastAPI, uvicorn

## Contact

seungwon.do@etri.re.kr
