import torch
import numpy as np
from typing import Any, Dict, List, Tuple
from vgqa.utils.training_utils import NestedTensor


def collate_fn(batch: List[Tuple[torch.Tensor, str, Dict[str, Any]]]) -> Dict[str, Any]:
    """Collate batch items into nested tensors for dataloader."""
    videos, texts, targets = list(zip(*batch))

    result: Dict[str, Any] = {}
    result["durations"] = [video.shape[0] for video in videos]
    result["videos"] = NestedTensor.from_tensor_list(list(videos))
    result["texts"] = list(texts)
    result["targets"] = list(targets)

    return result
    

