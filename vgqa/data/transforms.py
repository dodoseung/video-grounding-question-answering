import random

import torch
import torchvision
from torchvision.transforms import functional as F
import torchvision.transforms as T
from typing import Any, Dict, Tuple
from vgqa.utils.bounding_boxes import BoxList


class Compose(object):
    """Compose multiple transforms together."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Apply all transforms in sequence."""
        for transform in self.transforms:
            input_dict = transform(input_dict)
        return input_dict

    def __repr__(self) -> str:
        formatted = [f"    {t}" for t in self.transforms]
        return f"{self.__class__.__name__}(\n" + "\n".join(formatted) + "\n)"


class ColorJitter(object):
    """Apply random color jittering to video frames."""

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )

    def __call__(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Apply color jitter with 80% probability."""
        if random.random() < 0.8:
            frames = input_dict["frames"]
            input_dict["frames"] = self.color_jitter(frames)
        return input_dict


class RandomHorizontalFlip(object):
    """Randomly flip video frames and bounding boxes horizontally."""

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Flip frames and boxes, and swap left/right in text."""
        if random.random() < self.prob:
            frames = input_dict["frames"]
            boxs: BoxList = input_dict["boxs"]
            text: str = input_dict["text"]

            frames = F.hflip(frames)
            boxs = boxs.transpose(0)
            text = (
                text.replace("right", "*&^special^&*")
                .replace("left", "right")
                .replace("*&^special^&*", "left")
            )

            input_dict["frames"] = frames
            input_dict["boxs"] = boxs
            input_dict["text"] = text

        return input_dict


class RandomSelect(object):
    """Randomly select between two transform pipelines."""

    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        if random.random() < self.p:
            return self.transforms1(input_dict)
        return self.transforms2(input_dict)


class RandomResize(object):
    """Randomly resize frames while maintaining aspect ratio."""

    def __init__(self, min_size, max_size=None):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size

    def get_size(self, image_size: Tuple[int, int]) -> Tuple[int, int]:
        """Compute target size (h, w) based on constraints."""
        h, w = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        frames = input_dict["frames"]
        boxs: BoxList = input_dict["boxs"]
        current_h, current_w = frames.shape[2], frames.shape[3]
        target_h, target_w = self.get_size((current_h, current_w))

        frames = F.resize(frames, (target_h, target_w), antialias=True)
        boxs = boxs.resize((target_w, target_h))
        input_dict["frames"] = frames
        input_dict["boxs"] = boxs
        return input_dict


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int, max_try: int = 50):
        self.min_size = min_size
        self.max_size = max_size
        self.max_try = max_try

    def __call__(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        frames = input_dict["frames"]
        boxs: BoxList = input_dict["boxs"]

        for _ in range(self.max_try):
            h = frames.shape[2]
            w = frames.shape[3]
            tw = random.randint(self.min_size, min(w, self.max_size))
            th = random.randint(self.min_size, min(h, self.max_size))

            region = T.RandomCrop.get_params(frames, [th, tw])  # [i, j, th, tw]
            if boxs.check_crop_valid(region):
                frames = F.crop(frames, *region)
                boxs = boxs.crop(region)
                input_dict["frames"] = frames
                input_dict["boxs"] = boxs
                return input_dict

        return input_dict


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        frames = input_dict["frames"]
        boxs: BoxList = input_dict["boxs"]
        frames = F.normalize(frames, mean=self.mean, std=self.std)
        assert boxs.size == (frames.shape[3], frames.shape[2])  # (w, h)
        boxs = boxs.normalize()
        input_dict["frames"] = frames
        input_dict["boxs"] = boxs
        return input_dict


class NormalizeAndPad(object):
    def __init__(self, mean, std, size, aug_translate=False):
        self.mean = mean
        self.std = std
        self.size = size
        self.aug_translate = aug_translate

    def __call__(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        frames = input_dict["frames"]
        frames = F.normalize(frames, mean=self.mean, std=self.std)

        t, _, h, w = frames.shape
        dw = self.size - w
        dh = self.size - h

        if self.aug_translate:
            top = random.randint(0, dh)
            left = random.randint(0, dw)
        else:
            top = round(dh / 2.0 - 0.1)
            left = round(dw / 2.0 - 0.1)

        out_frames = torch.zeros((t, 3, self.size, self.size)).float()
        out_mask = torch.ones((self.size, self.size)).int()

        out_frames[:, :, top : top + h, left : left + w] = frames
        out_mask[top : top + h, left : left + w] = 0

        input_dict["frames"] = out_frames
        input_dict["mask"] = out_mask

        if "boxs" in input_dict.keys():
            boxs: BoxList = input_dict["boxs"]
            boxs = boxs.shift((self.size, self.size), left, top)
            input_dict["boxs"] = boxs

        return input_dict