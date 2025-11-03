"""Shared video loading and preprocessing utilities."""

import numpy as np
from typing import List, Optional, Tuple
from decord import VideoReader, cpu
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode


# ImageNet normalization constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_video_reader(video_path: str) -> VideoReader:
    """Load video using decord VideoReader.

    Args:
        video_path: Path to video file

    Returns:
        VideoReader object
    """
    return VideoReader(video_path, ctx=cpu(0), num_threads=1)


def get_video_info(vr: VideoReader) -> Tuple[int, float]:
    """Get video metadata.

    Args:
        vr: VideoReader object

    Returns:
        Tuple of (total_frames, fps)
    """
    total_frames = len(vr)
    fps = float(vr.get_avg_fps()) if hasattr(vr, "get_avg_fps") else 30.0
    return total_frames, fps


def uniform_sample_indices(total_frames: int, target_frames: int) -> List[int]:
    """Uniformly sample frame indices.

    Args:
        total_frames: Total number of frames in video
        target_frames: Number of frames to sample

    Returns:
        List of frame indices
    """
    target_frames = max(1, min(int(target_frames), int(total_frames)))
    if target_frames == total_frames:
        return list(range(total_frames))
    return [int(round(i * (total_frames - 1) / (target_frames - 1))) for i in range(target_frames)]


def load_frames(vr: VideoReader, indices: List[int]) -> List[np.ndarray]:
    """Load frames from video at specified indices.

    Args:
        vr: VideoReader object
        indices: List of frame indices to load

    Returns:
        List of numpy arrays (frames)
    """
    try:
        batch = vr.get_batch(indices).asnumpy()
        return [batch[i] for i in range(batch.shape[0])]
    except Exception:
        return [vr[i].asnumpy() for i in indices]


def build_transform(input_size: int = 448,
                    mean: Tuple[float, float, float] = IMAGENET_MEAN,
                    std: Tuple[float, float, float] = IMAGENET_STD) -> T.Compose:
    """Build image transformation pipeline.

    Args:
        input_size: Target image size
        mean: Normalization mean
        std: Normalization std

    Returns:
        Composition of transforms
    """
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])


def get_frame_indices_with_bound(
    bound: Optional[Tuple[float, float]],
    fps: float,
    max_frame: int,
    num_segments: int = 32
) -> np.ndarray:
    """Get frame indices based on temporal bounds.

    Args:
        bound: Optional tuple of (start_time, end_time) in seconds
        fps: Video frames per second
        max_frame: Maximum frame index
        num_segments: Number of segments to sample

    Returns:
        Array of frame indices
    """
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000

    start_idx = max(0, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments

    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])

    return frame_indices
