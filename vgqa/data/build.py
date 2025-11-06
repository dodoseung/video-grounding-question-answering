import os
import math
import bisect
import copy


import torch
import torch.utils.data
from typing import Any, Iterable, List
from vgqa.utils.distributed import get_world_size

from torch.utils.data import DistributedSampler

from . import samplers
from . import transforms as T
from .vidstg_dataset import VidSTGDataset
from .video_batch_collator import collate_fn


def build_transforms(cfg: Any, is_train: bool = True):
    """Build image transformation pipeline for training or evaluation."""
    target_short_side = cfg.INPUT.RESOLUTION
    max_long_side = 720

    if is_train:
        flip_prob = cfg.INPUT.FLIP_PROB_TRAIN

        if cfg.INPUT.AUG_SCALE:
            resize_candidates: List[int] = [target_short_side - 32 * i for i in range(4)]
        else:
            resize_candidates = [target_short_side]

        transform = T.Compose(
            [
                T.RandomHorizontalFlip(flip_prob),
                T.RandomSelect(
                    T.RandomResize(resize_candidates, max_size=max_long_side),
                    T.Compose(
                        [
                            T.RandomResize([400, 500, 600]),
                            T.RandomSizeCrop(384, 600),
                            T.RandomResize(resize_candidates, max_size=max_long_side),
                        ]
                    ),
                ),
                T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            ]
        )
        return transform

    transform = T.Compose(
        [
            T.RandomResize(target_short_side, max_size=max_long_side),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        ]
    )
    return transform


def build_dataset(cfg: Any, split: str, transforms):
    """Build VidSTG dataset for given split."""
    return VidSTGDataset(cfg, split, transforms)


def make_data_sampler(dataset, shuffle: bool, distributed: bool):
    """Create data sampler for distributed or single-process training."""
    if distributed:
        return DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        return torch.utils.data.sampler.RandomSampler(dataset)
    return torch.utils.data.sampler.SequentialSampler(dataset)


def _quantize(values: Iterable[float], bins: Iterable[float]):
    buckets = copy.copy(list(bins))
    buckets = sorted(buckets)
    return [bisect.bisect_right(buckets, val) for val in values]


def _compute_aspect_ratios(dataset) -> List[float]:
    ratios: List[float] = []
    for i in range(len(dataset)):
        info = dataset.get_video_info(i)
        ratios.append(float(info["height"]) / float(info["width"]))
    return ratios


def _count_frame_size(dataset):
    sizes = {}
    for i in range(len(dataset)):
        info = dataset.get_video_info(i)
        key = (info["width"], info["height"])
        sizes[key] = sizes.get(key, 0) + 1


def make_batch_data_sampler(
    dataset,
    sampler,
    aspect_grouping,
    batch_size,
    num_iters=None,
    start_iter: int = 0,
    is_train: bool = True,
):
    if aspect_grouping:
        if not isinstance(aspect_grouping, (list, tuple)):
            aspect_grouping = [aspect_grouping]
        aspect_ratios = _compute_aspect_ratios(dataset)
        group_ids = _quantize(aspect_ratios, aspect_grouping)
        batch_sampler = samplers.GroupedBatchSampler(
            sampler, group_ids, batch_size, drop_uneven=False
        )
    else:
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, batch_size, drop_last=True if is_train else False
        )

    if num_iters is not None:
        batch_sampler = samplers.IterationBasedBatchSampler(
            batch_sampler, num_iters, start_iter
        )
    return batch_sampler


def make_data_loader(cfg: Any, mode: str = "train", is_distributed: bool = False, start_iter: int = 0):
    assert mode in {"train", "val", "test"}
    num_gpus = get_world_size()
    is_train = mode == "train"

    transforms = build_transforms(cfg, is_train)
    dataset = build_dataset(cfg, mode, transforms)

    if cfg.SOLVER.BATCH_SIZE != 1:
        # The pipeline assumes one video per device per step
        raise AssertionError("Each GPU should only take 1 video.")

    videos_per_device = cfg.SOLVER.BATCH_SIZE
    shuffle = is_train

    if is_train:
        global_batch_size = videos_per_device * num_gpus
        num_epochs = cfg.SOLVER.MAX_EPOCH
        total_iters = num_epochs * math.ceil(len(dataset) / global_batch_size)
    else:
        total_iters = None
        start_iter = 0

    aspect_grouping = [1] if cfg.DATALOADER.ASPECT_RATIO_GROUPING else []

    sampler = make_data_sampler(dataset, shuffle, is_distributed)
    batch_sampler = make_batch_data_sampler(
        dataset,
        sampler,
        aspect_grouping,
        videos_per_device,
        total_iters,
        start_iter,
        is_train=is_train,
    )
    num_workers = cfg.DATALOADER.NUM_WORKERS

    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
    )

    return data_loader