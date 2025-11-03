# VGQA: Video Grounding and Question Answering

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

A unified, production-ready framework for **spatio-temporal video grounding** and **video question answering**. VGQA combines state-of-the-art models with an intuitive API and web interface for real-world video understanding tasks.

## Features

- **Spatio-Temporal Video Grounding**: Localize objects in space and time using natural language queries
- **Video Question Answering**: Answer questions about video content using InternVideo2.5-Chat-8B
- **Dual Inference Modes**: CLI tools and Python API for flexible integration
- **Web Interface**: Interactive demo with real-time visualization
- **Production Ready**: Clean architecture, comprehensive documentation, and extensive testing

## Quick Start

### System Requirements

- Python 3.8 or higher
- CUDA 11.8+ (for GPU support)
- 16GB+ RAM recommended
- 40GB+ disk space for models and checkpoints

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/vgqa.git
cd vgqa

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package
pip install -e .

# For web interface support
pip install -e ".[web]"

# For development
pip install -e ".[dev]"

# Everything
pip install -e ".[all]"
```

### GPU Support

For CUDA 11.8:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

For CUDA 12.1:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Download Pre-trained Models

```bash
# Create checkpoints directory
mkdir -p checkpoints/grounding checkpoints/qa checkpoints/pretrained

# Download grounding model checkpoint
# wget -O checkpoints/grounding/tastvg_vidstg.pth <CHECKPOINT_URL>

# Download InternVideo2.5-Chat-8B for QA
# Place the model files in checkpoints/qa/InternVideo2_5_Chat_8B/

# Download pretrained backbones
# wget https://huggingface.co/FacebookAI/roberta-base -P ./checkpoints/pretrained/
# wget https://zenodo.org/record/4721981/files/pretrained_resnet101_checkpoint.pth -P ./checkpoints/pretrained/
# wget https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_tiny_patch244_window877_kinetics400_1k.pth -P ./checkpoints/pretrained/

```

### Usage

#### Python API

```python
from vgqa.inference import grounding, qa

# Spatio-temporal grounding
result = grounding.predict(
    video_path="videos/sample1.mp4",
    query="a person jumping",
    cfg_path="configs/grounding_vidstg.yaml",
    ckpt_path="checkpoints/grounding/tastvg_vidstg.pth"
)
print(f"Temporal bounds: {result['temporal']}")
print(f"Spatial tubes: {result['tube'][:5]}")  # First 5 frames

# Video question answering
answer = qa.predict(
    video_path="videos/sample1.mp4",
    question="What is happening in this video?",
    model_dir="checkpoints/qa/InternVideo2_5_Chat_8B"
)
print(f"Answer: {answer['answer']}")

# QA with temporal bounds (focus on specific time range)
answer = qa.predict(
    video_path="videos/sample1.mp4",
    question="What color is the shirt?",
    bound=(2.0, 5.0),  # Only analyze frames between 2s and 5s
    num_frames=16,     # Use fewer frames for faster inference
)

# QA with custom generation parameters
answer = qa.predict(
    video_path="videos/sample1.mp4",
    question="Describe what happens in detail.",
    max_new_tokens=256,  # Generate longer answers
    temperature=0.7,     # More creative responses
    top_p=0.95,
)
```

#### Command Line Interface

```bash
# Spatio-temporal grounding
python tools/infer_grounding.py \
    --video videos/sample1.mp4 \
    --query "a dancing girl" \
    --cfg configs/grounding_vidstg.yaml \
    --ckpt checkpoints/grounding/tastvg_vidstg.pth

# Video question answering
python tools/infer_qa.py \
    --video videos/sample1.mp4 \
    --question "What is happening?" \
    --model-dir checkpoints/qa/InternVideo2_5_Chat_8B
```

#### Web Interface

```bash
# Start the web server
python app/server.py

# Open browser and navigate to:
# http://localhost:8000/app/index.html
```

**Using REST API Endpoints:**

```python
import requests

# List available videos
response = requests.get("http://localhost:8000/api/videos")
videos = response.json()["videos"]

# Run grounding inference
response = requests.post(
    "http://localhost:8000/api/predict",
    json={
        "video_name": "sample1.mp4",
        "query": "person dancing"
    }
)
result = response.json()

# Run QA inference
response = requests.post(
    "http://localhost:8000/api/qa",
    json={
        "video_name": "sample1.mp4",
        "question": "What is happening?"
    }
)
answer = response.json()
```

## Architecture

```
VGQA/
├── vgqa/                       # Main package
│   ├── core/                   # Model architectures
│   │   ├── grounding_net.py   # Grounding model
│   │   ├── vision/            # Vision encoders
│   │   ├── language/          # Text encoders
│   │   └── decoder/           # Grounding decoder
│   ├── data/                  # Dataset handling
│   ├── inference/             # Inference APIs
│   │   ├── grounding.py       # Grounding inference
│   │   ├── qa.py              # QA inference
│   │   └── video_utils.py     # Shared utilities
│   ├── training/              # Training utilities
│   └── utils/                 # Common utilities
├── tools/                     # CLI tools
│   ├── infer_grounding.py     # Grounding CLI
│   ├── infer_qa.py            # QA CLI
│   ├── train.py               # Training script
│   └── evaluate.py            # Evaluation script
├── app/                       # Web application
│   ├── server.py              # FastAPI server
│   └── static/                # Frontend
├── configs/                   # Configuration files
├── checkpoints/               # Model weights
└── videos/                    # Example videos
```

## Model Details

### Spatio-Temporal Grounding

The grounding model localizes objects in videos based on natural language queries:

- **Input**: Video + text query (e.g., "a person waving")
- **Output**:
  - Temporal bounds (start/end time)
  - Spatial bounding boxes per frame
  - Confidence scores

**Architecture**: Transformer-based model with vision-language fusion

### Video Question Answering

The QA model answers questions about video content:

- **Input**: Video + question (e.g., "What color is the shirt?")
- **Output**: Natural language answer

**Model**: InternVideo2.5-Chat-8B (8 billion parameters)

## Training

```bash
# Train grounding model
python tools/train.py \
    --config-file configs/grounding_vidstg_mini.yaml \
    OUTPUT_DIR output/experiment

# Evaluate on test set
python tools/evaluate.py \
    --config-file configs/grounding_vidstg_mini.yaml
```

## Configuration

All models and training parameters are configured via YAML files in `configs/`:

```yaml
# configs/grounding_vidstg.yaml
MODEL:
  VISION_BACKBONE:
    NAME: resnet101
  TEXT_ENCODER:
    NAME_OR_PATH: checkpoints/pretrained/roberta-base
  TASTVG:
    QUERY_DIM: 512
    DEC_LAYERS: 6

INPUT:
  RESOLUTION: 420
  MAX_VIDEO_LEN: 200

SOLVER:
  MAX_EPOCH: 11
  BASE_LR: 0.0001
  BATCH_SIZE: 1
```

## API Reference

### Grounding API

```python
from vgqa.inference.grounding import predict

result = predict(
    video_path: str,              # Path to video file
    query: str,                   # Natural language query
    cfg_path: str,                # Path to config YAML
    ckpt_path: str,               # Path to checkpoint
    device_str: Optional[str],    # "cuda" or "cpu"
)
# Returns: {"temporal": {...}, "tube": [{frame, bbox, score}, ...]}
```

### QA API

```python
from vgqa.inference.qa import predict

answer = predict(
    video_path: str,              # Path to video file
    question: str,                # Question about the video
    model_dir: str,               # Path to model directory
    num_frames: int = 32,         # Number of frames to sample
    temperature: float = 0.2,     # Sampling temperature
)
# Returns: {"answer": str}
```

### Visualizing Grounding Results

```python
import cv2
from vgqa.inference.grounding import predict

# Run inference
result = predict(video_path="video.mp4", query="person running")

# Load video
cap = cv2.VideoCapture("video.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)

# Draw bounding boxes
for tube_item in result["tube"]:
    frame_idx = tube_item["frame"]
    bbox = tube_item["bbox"]

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()

    # Draw bbox
    x1, y1, x2, y2 = [int(c) for c in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Grounding Result", frame)
    cv2.waitKey(int(1000/fps))

cap.release()
cv2.destroyAllWindows()
```

### Batch Processing QA

```python
from vgqa.inference.qa import predict
import json

questions = [
    "What is happening?",
    "How many people are in the video?",
    "What is the weather like?",
]

results = []
for question in questions:
    answer = predict(
        video_path="videos/sample1.mp4",
        question=question
    )
    results.append({
        "question": question,
        "answer": answer["answer"]
    })

# Save results
with open("qa_results.json", "w") as f:
    json.dump(results, f, indent=2)
```

## Advanced Usage

### Custom Configurations

Create custom config for different model settings:

```yaml
# configs/my_config.yaml
MODEL:
  VISION_BACKBONE:
    NAME: resnet101
    PRETRAINED_PATH: checkpoints/pretrained/resnet101.pth
  TEXT_ENCODER:
    NAME_OR_PATH: checkpoints/pretrained/roberta-base
  TASTVG:
    QUERY_DIM: 512
    DEC_LAYERS: 6

INPUT:
  RESOLUTION: 420
  MAX_VIDEO_LEN: 200
```

Use it:

```python
from vgqa.inference.grounding import predict

result = predict(
    video_path="video.mp4",
    query="person",
    cfg_path="configs/my_config.yaml"
)
```

### Environment Variables

```bash
# Set custom video directory for web app
export VGQA_VIDEOS_DIR=/path/to/videos

# Set custom model cache directory
export TRANSFORMERS_CACHE=/path/to/cache

python app/server.py
```

### Performance Optimization

```python
# For faster inference, reduce frames
result = grounding.predict(
    video_path="long_video.mp4",
    query="person",
    # Model will internally sample fewer frames
)

# For QA, use fewer frames and smaller tokens
answer = qa.predict(
    video_path="video.mp4",
    question="What?",
    num_frames=8,        # Down from 32
    max_new_tokens=64,   # Down from 128
)
```

### Verification

Test your installation:

```bash
# Test imports
python -c "import vgqa; print(vgqa.__version__)"

# Test grounding inference
python tools/infer_grounding.py --help

# Test QA inference
python tools/infer_qa.py --help
```

## Troubleshooting

### Installation Issues

**CUDA Out of Memory:**
- Reduce batch size in config
- Use smaller models
- Enable gradient checkpointing

**Import Errors:**
```bash
# Make sure project root is in PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/path/to/vgqa
```

**Decord Installation Issues:**
```bash
# On Ubuntu/Debian
sudo apt-get install ffmpeg

# Then reinstall decord
pip uninstall decord
pip install decord==0.6.0
```

### Runtime Issues

**Out of Memory:**
- Reduce `num_frames` parameter
- Use CPU instead of GPU for small videos
- Close other GPU applications

**Slow Inference:**
- Ensure you're using GPU (`device_str="cuda"`)
- Reduce video resolution
- Use fewer frames

**Poor Results:**
- Try different text queries (be more specific)
- Adjust temperature for QA tasks
- Check if video quality is sufficient

### Docker Installation (Optional)

```dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.8-cudnn8-runtime

WORKDIR /workspace
COPY . /workspace/vgqa

RUN pip install -e /workspace/vgqa[all]

CMD ["python", "app/server.py"]
```

Build and run:

```bash
docker build -t vgqa:latest .
docker run -p 8000:8000 --gpus all vgqa:latest
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

## Citation

If you use VGQA in your research, please cite:

```bibtex
@software{vgqa2024,
  title={VGQA: Video Grounding and Question Answering},
  author={VGQA Team},
  year={2024},
  url={https://github.com/your-username/vgqa}
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- InternVideo2.5 team for the video QA model
- Original TA-STVG implementation for grounding architecture inspiration
- VidSTG dataset creators

## Contact

For questions and issues, please open a GitHub issue or contact: vgqa-team@example.com

---

**Note**: This is a research project. Model weights and datasets should be obtained separately and placed in the appropriate directories as described in the installation instructions.
