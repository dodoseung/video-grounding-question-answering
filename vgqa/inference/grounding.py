"""Spatio-temporal video grounding inference."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import numpy as np
from torchvision import transforms

from vgqa.utils.misc import NestedTensor
from vgqa.training.evaluator import single_forward, linear_interp, linear_interp_conf
from vgqa.core import build_postprocessors
from .video_utils import load_video_reader, get_video_info, uniform_sample_indices, load_frames


# Default paths
DEFAULT_CONFIG_PATH = "configs/grounding_vidstg.yaml"
DEFAULT_CHECKPOINT_PATH = "checkpoints/grounding/tastvg_vidstg.pth"


def _load_yaml_config(config_path: str):
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    from vgqa.config import cfg as default_cfg
    cfg = default_cfg.clone()
    cfg.merge_from_file(config_path)
    cfg.freeze()
    return cfg


def _bind_local_model_zoo(cfg):
    """Bind local model zoo paths to configuration."""
    cfg.defrost()
    model_zoo_dir = "checkpoints/pretrained"
    if hasattr(cfg, 'MODEL') and hasattr(cfg.MODEL, 'VISION_BACKBONE'):
        vision_cfg = cfg.MODEL.VISION_BACKBONE
        if hasattr(vision_cfg, 'NAME'):
            if vision_cfg.NAME == 'resnet101' and hasattr(vision_cfg, 'PRETRAINED_PATH'):
                if not vision_cfg.PRETRAINED_PATH:
                    vision_cfg.PRETRAINED_PATH = os.path.join(model_zoo_dir, "resnet101.pth")
            elif vision_cfg.NAME == 'swin_tiny' and hasattr(vision_cfg, 'PRETRAINED_PATH'):
                if not vision_cfg.PRETRAINED_PATH:
                    vision_cfg.PRETRAINED_PATH = os.path.join(model_zoo_dir, "swin_tiny_patch4_window7_224.pth")
    # Handle both TEXT_ENCODER and TEXT_MODEL naming conventions
    if hasattr(cfg, 'MODEL'):
        if hasattr(cfg.MODEL, 'TEXT_ENCODER'):
            if hasattr(cfg.MODEL.TEXT_ENCODER, 'NAME_OR_PATH'):
                if not cfg.MODEL.TEXT_ENCODER.NAME_OR_PATH:
                    cfg.MODEL.TEXT_ENCODER.NAME_OR_PATH = os.path.join(model_zoo_dir, "roberta-base")
        elif hasattr(cfg.MODEL, 'TEXT_MODEL'):
            # Set NAME to the full path for inference
            if hasattr(cfg.MODEL.TEXT_MODEL, 'NAME'):
                if cfg.MODEL.TEXT_MODEL.NAME == 'roberta-base':
                    cfg.MODEL.TEXT_MODEL.NAME = os.path.join(model_zoo_dir, "roberta-base")
    cfg.freeze()
    return cfg


def _get_device(explicit: Optional[str] = None) -> torch.device:
    """Get the appropriate device."""
    if explicit:
        return torch.device(explicit)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _build_transforms(cfg) -> transforms.Compose:
    """Build image transformation pipeline from config."""
    input_size = int(getattr(cfg.INPUT, 'RESOLUTION', 224))
    mean = [float(x) for x in getattr(cfg.INPUT, 'PIXEL_MEAN', [0.485, 0.456, 0.406])]
    std = [float(x) for x in getattr(cfg.INPUT, 'PIXEL_STD', [0.229, 0.224, 0.225])]
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def _load_model(cfg, ckpt_path: str, device: torch.device) -> torch.nn.Module:
    """Load the grounding model from checkpoint."""
    # Try to import build_model from vgqa.core
    from vgqa.core import build_model

    result = build_model(cfg)
    model = result[0] if isinstance(result, tuple) else result
    model.to(device)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = None
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state_dict = ckpt["state_dict"]
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            state_dict = ckpt["model"]
    if state_dict is None and isinstance(ckpt, dict):
        if all(isinstance(k, str) for k in ckpt.keys()):
            state_dict = ckpt
    if state_dict is None and isinstance(ckpt, torch.nn.Module):
        model = ckpt
        model.to(device)
        model.eval()
        return model
    if state_dict is None:
        raise ValueError("Unrecognized checkpoint format. Expected 'state_dict' or a full Module.")

    model.load_state_dict(state_dict, strict=False)

    # Ensure verb_label2 exists for compatibility
    if not hasattr(model, 'verb_label2') or not model.verb_label2:
        model.verb_label2 = {}
    if '0' not in model.verb_label2:
        model.verb_label2['0'] = {'sub': '', 'verb_index_list': [], 'adj_index_list': []}

    model.eval()
    return model


def _preprocess_frames(frames: List[np.ndarray], transform: transforms.Compose) -> torch.Tensor:
    """Preprocess frames with the given transform."""
    processed: List[torch.Tensor] = []
    for frame_np in frames:
        processed.append(transform(frame_np))
    if not processed:
        raise ValueError("No frames read from video.")
    return torch.stack(processed, dim=0)


def predict(
    video_path: str,
    query: str,
    cfg_path: str = DEFAULT_CONFIG_PATH,
    ckpt_path: str = DEFAULT_CHECKPOINT_PATH,
    device_str: Optional[str] = None,
    batch_size: int = 32,
) -> Dict[str, Any]:
    """Run spatio-temporal video grounding inference.

    Args:
        video_path: Path to the video file
        query: Text query for grounding
        cfg_path: Path to config YAML file
        ckpt_path: Path to checkpoint file
        device_str: Device string (e.g., "cuda", "cpu")
        batch_size: Batch size (unused, kept for compatibility)

    Returns:
        Dictionary with "temporal" and "tube" keys containing predictions

    Raises:
        FileNotFoundError: If video, config, or checkpoint not found
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cfg = _load_yaml_config(cfg_path)
    cfg = _bind_local_model_zoo(cfg)
    device = _get_device(device_str)
    model = _load_model(cfg, ckpt_path, device)

    vr = load_video_reader(video_path)
    total_frames, fps = get_video_info(vr)

    try:
        target_base = int(cfg.INPUT.TRAIN_SAMPLE_NUM)
    except Exception:
        target_base = 64
    target_T = max(2, target_base * 2)
    frame_ids = uniform_sample_indices(total_frames, target_T)
    raw_frames = load_frames(vr, frame_ids)
    print(f"Loaded video: {total_frames} frames, {fps:.2f} fps; using {len(raw_frames)} sampled frames")

    h0, w0 = raw_frames[0].shape[0], raw_frames[0].shape[1]

    transform = _build_transforms(cfg)
    video_tensor = _preprocess_frames(raw_frames, transform)

    Tn, C, H, W = video_tensor.shape
    mask = torch.zeros((Tn, H, W), dtype=torch.bool, device='cpu')
    nested_video = NestedTensor(video_tensor, mask, [Tn])

    videos1 = nested_video.subsample(2, start_idx=0)
    videos2 = nested_video.subsample(2, start_idx=1)
    fids1 = frame_ids[0::2]
    fids2 = frame_ids[1::2]
    print(f"Subsampling video into 2 passes: even={len(fids1)}, odd={len(fids2)}")

    targets1 = [{
        'item_id': 0,
        'vid': 'dummy_video',
        'ori_size': (h0, w0),
        'qtype': 'declar',
        'frame_ids': fids1,
        'actioness': torch.ones(len(fids1), dtype=torch.float32, device=device),
    }]
    targets2 = [{
        'item_id': 0,
        'vid': 'dummy_video',
        'ori_size': (h0, w0),
        'qtype': 'declar',
        'frame_ids': fids2,
        'actioness': torch.ones(len(fids2), dtype=torch.float32, device=device),
    }]

    postprocessor = build_postprocessors()
    with torch.no_grad():
        videos1 = videos1.to(device)
        bbox_pred1, att_pred1, temp_pred1, kf_pred1 = single_forward(
            cfg, model, videos1, [query], targets1, device, postprocessor
        )
        videos2 = videos2.to(device)
        bbox_pred2, att_pred2, temp_pred2, kf_pred2 = single_forward(
            cfg, model, videos2, [query], targets2, device, postprocessor
        )

    vid_key = 0
    bbox_dict = bbox_pred1[vid_key]
    bbox_dict.update(bbox_pred2[vid_key])
    bbox_full = linear_interp(bbox_dict)

    att_dict = att_pred1[vid_key]
    att_dict.update(att_pred2[vid_key])
    att_full = linear_interp_conf(att_dict)

    sted1 = temp_pred1[vid_key]['sted']
    sted2 = temp_pred2[vid_key]['sted']
    merged_sted = [min(sted1[0], sted2[0]), max(sted1[1], sted2[1])]

    temporal = {
        "start": float(merged_sted[0]) / max(fps, 1e-6),
        "end": float(merged_sted[1]) / max(fps, 1e-6),
        "score": 1.0,
    }

    tube = []
    for fid in sorted(bbox_full.keys()):
        bbox = bbox_full[fid][0]
        conf_v = att_full.get(fid, 1.0)
        score = float(conf_v[0] if isinstance(conf_v, list) else conf_v)
        tube.append({
            "frame": int(fid),
            "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
            "score": score,
        })

    return {"temporal": temporal, "tube": tube}
