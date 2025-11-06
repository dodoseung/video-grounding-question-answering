import torch
import numpy as np
from torchvision.ops.boxes import box_area
from typing import Tuple


# -------------------------
# NumPy helpers (xyxy)
# -------------------------
def np_box_area(boxes: np.ndarray) -> np.ndarray:
    """Area of boxes in xyxy format.

    boxes: (N, 4) with columns [x_min, y_min, x_max, y_max]
    """
    assert boxes.ndim == 2 and boxes.shape[-1] == 4
    widths = (boxes[:, 2] - boxes[:, 0])
    heights = (boxes[:, 3] - boxes[:, 1])
    return widths * heights


def _box_inter_union(boxes1: np.ndarray, boxes2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Intersection and union areas for two sets of xyxy boxes (NumPy)."""
    area1 = np_box_area(boxes1)
    area2 = np_box_area(boxes2)

    top_left = np.maximum(boxes1[:, None, :2], boxes2[:, :2])   # [N,M,2]
    bottom_right = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (bottom_right - top_left).clip(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    union = area1[:, None] + area2 - inter
    return inter, union


def np_box_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """IoU for two sets of xyxy boxes (NumPy)."""
    inter, union = _box_inter_union(boxes1, boxes2)
    return inter / union


# -------------------------
# Torch helpers
# -------------------------
def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    """(cx, cy, w, h) -> (x_min, y_min, x_max, y_max)."""
    cx, cy, w, h = x.unbind(-1)
    x_min = cx - 0.5 * w
    y_min = cy - 0.5 * h
    x_max = cx + 0.5 * w
    y_max = cy + 0.5 * h
    return torch.stack([x_min, y_min, x_max, y_max], dim=-1)


def box_xyxy_to_cxcywh(x: torch.Tensor) -> torch.Tensor:
    """(x_min, y_min, x_max, y_max) -> (cx, cy, w, h)."""
    x_min, y_min, x_max, y_max = x.unbind(-1)
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    w = (x_max - x_min)
    h = (y_max - y_min)
    return torch.stack([cx, cy, w, h], dim=-1)


def _torch_inter_union(boxes1: torch.Tensor, boxes2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Internal: intersection, union, enclosing area for xyxy boxes (torch)."""
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    top_left = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    bottom_right = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (bottom_right - top_left).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter

    # enclosing rectangle (for GIoU)
    enc_tl = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    enc_br = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    enc_wh = (enc_br - enc_tl).clamp(min=0)
    enc_area = enc_wh[:, :, 0] * enc_wh[:, :, 1]

    return inter, union, enc_area


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """IoU matrix and union for xyxy boxes (torch)."""
    inter, union, _ = _torch_inter_union(boxes1, boxes2)
    return inter / union, union


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Generalized IoU for xyxy boxes (torch)."""
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)
    _, _, enc_area = _torch_inter_union(boxes1, boxes2)
    return iou - (enc_area - union) / enc_area


def masks_to_boxes(masks: torch.Tensor) -> torch.Tensor:
    """Binary mask -> xyxy boxes (torch)."""
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]
    y = torch.arange(0, h, dtype=torch.float32, device=masks.device)
    x = torch.arange(0, w, dtype=torch.float32, device=masks.device)
    # torch >=1.10 default indexing='ij'; be explicit for older versions
    y, x = torch.meshgrid(y, x, indexing='ij') if 'indexing' in torch.meshgrid.__code__.co_varnames else torch.meshgrid(y, x)

    x_mask = masks * x.unsqueeze(0)
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = masks * y.unsqueeze(0)
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], dim=1)


