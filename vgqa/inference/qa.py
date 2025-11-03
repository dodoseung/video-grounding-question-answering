"""Video Question Answering using InternVideo2.5-Chat-8B."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from PIL import Image

from .video_utils import (
    load_video_reader,
    get_video_info,
    get_frame_indices_with_bound,
    IMAGENET_MEAN,
    IMAGENET_STD,
)


# Default model directory
DEFAULT_MODEL_DIR = "checkpoints/qa/InternVideo2_5_Chat_8B"


def _get_device() -> torch.device:
    """Get the appropriate device (CUDA if available, else CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _build_transform(input_size: int = 448):
    """Build image transformation pipeline for InternVideo2.5."""
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def _find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: List[Tuple[int, int]],
    width: int,
    height: int,
    image_size: int
) -> Tuple[int, int]:
    """Find the closest aspect ratio from target ratios."""
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height

    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio

    return best_ratio


def _dynamic_preprocess(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = 6,
    image_size: int = 448,
    use_thumbnail: bool = True
) -> List[Image.Image]:
    """Dynamically preprocess image into multiple patches."""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # Calculate target ratios
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # Find closest aspect ratio
    target_aspect_ratio = _find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # Calculate target dimensions
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # Resize and split image
    resized_img = image.resize((target_width, target_height))
    processed_images = []

    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    assert len(processed_images) == blocks

    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)

    return processed_images


def _load_video(
    video_path: str,
    bound: Optional[Tuple[float, float]] = None,
    input_size: int = 448,
    max_num: int = 1,
    num_segments: int = 32
) -> Tuple[torch.Tensor, List[int]]:
    """Load and preprocess video for InternVideo2.5.

    Returns:
        pixel_values: Concatenated tensor of all frame patches
        num_patches_list: List of number of patches per frame
    """
    vr = load_video_reader(video_path)
    total_frames, fps = get_video_info(vr)
    max_frame = total_frames - 1

    print(f"Video info: {total_frames} frames, {fps:.2f} fps")

    pixel_values_list = []
    num_patches_list = []
    transform = _build_transform(input_size=input_size)

    frame_indices = get_frame_indices_with_bound(bound, fps, max_frame, num_segments=num_segments)
    print(f"Sampling {len(frame_indices)} frames from video...")

    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")
        img_patches = _dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img_patches]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)

    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list


def _load_offline_model(model_dir: str = DEFAULT_MODEL_DIR):
    """Load InternVideo2.5-Chat-8B model from local directory.

    Args:
        model_dir: Path to the local model directory

    Returns:
        Tuple of (model, tokenizer)

    Raises:
        FileNotFoundError: If the model directory doesn't exist
    """
    if not os.path.exists(model_dir):
        raise FileNotFoundError(
            f"InternVideo2.5-Chat-8B local directory not found: {model_dir}"
        )

    # Clear CUDA cache before loading model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"CUDA memory before loading: {torch.cuda.memory_allocated()/1024**3:.2f} GB allocated, "
              f"{torch.cuda.memory_reserved()/1024**3:.2f} GB reserved")

    print(f"Loading tokenizer from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, use_fast=False)

    print(f"Loading model from {model_dir}...")
    device = _get_device()

    # Load model with device_map for automatic memory management
    model = AutoModel.from_pretrained(
        model_dir,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
        device_map="auto",  # Automatically distribute model across available devices
    )
    model.eval()

    if torch.cuda.is_available():
        print(f"CUDA memory after loading: {torch.cuda.memory_allocated()/1024**3:.2f} GB allocated, "
              f"{torch.cuda.memory_reserved()/1024**3:.2f} GB reserved")

    return model, tokenizer


def predict(
    video_path: str,
    question: str,
    bound: Optional[Tuple[float, float]] = None,
    model_dir: str = DEFAULT_MODEL_DIR,
    num_frames: int = 32,
    max_new_tokens: int = 128,
    temperature: float = 0.2,
    top_p: float = 0.9,
    input_size: int = 448,
    max_num: int = 1,
) -> Dict[str, Any]:
    """Run InternVideo2.5-Chat-8B offline QA.

    Args:
        video_path: Path to the video file
        question: Question to ask about the video
        bound: Optional temporal bounds (start_time, end_time) in seconds
        model_dir: Path to the model directory
        num_frames: Number of frames to sample from video
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        input_size: Input image size (default 448)
        max_num: Maximum number of image patches per frame (default 1)

    Returns:
        Dictionary with "answer" key containing the model's response

    Raises:
        FileNotFoundError: If video or model directory doesn't exist
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Load model
    model, tokenizer = _load_offline_model(model_dir)
    device = _get_device()

    # Load and transform video
    print(f"Loading video from {video_path}...")
    pixel_values, num_patches_list = _load_video(
        video_path, bound=bound, input_size=input_size,
        max_num=max_num, num_segments=num_frames
    )
    pixel_values = pixel_values.to(torch.bfloat16).to(device)

    # Construct question with video prefix
    video_prefix = "".join([f"Frame{i+1}: <image>\n" for i in range(len(num_patches_list))])
    full_question = video_prefix + question

    print(f"Generating answer for question: '{question}'")
    print(f"Total frames: {len(num_patches_list)}, Total patches: {sum(num_patches_list)}")

    # Generate answer using the model's chat method
    generation_config = dict(
        do_sample=temperature > 0,
        temperature=max(temperature, 0.01),  # Avoid 0 temperature
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        num_beams=1
    )

    with torch.no_grad():
        answer, _ = model.chat(
            tokenizer,
            pixel_values,
            full_question,
            generation_config,
            num_patches_list=num_patches_list,
            history=None,
            return_history=True
        )

    print(f"Answer generated successfully")
    return {"answer": str(answer)}
