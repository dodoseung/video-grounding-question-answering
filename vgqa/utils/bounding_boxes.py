import torch

# transpose
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1


class BoxList(object):
    """Bounding boxes with image size and coordinate mode.

    - mode: "xyxy" or "xywh"
    - size: (width, height)
    """

    def __init__(self, bbox, image_size, mode="xyxy"):
        device = bbox.device if isinstance(bbox, torch.Tensor) else torch.device("cpu")
        tensor = torch.as_tensor(bbox, dtype=torch.float32, device=device)
        if tensor.ndimension() != 2:
            raise ValueError("bbox should have 2 dimensions, got {}".format(tensor.ndimension()))
        if tensor.size(-1) != 4:
            raise ValueError("last dimension of bbox should be 4, got {}".format(tensor.size(-1)))
        if mode not in ("xyxy", "xywh"):
            raise ValueError("mode should be 'xyxy' or 'xywh'")

        self.bbox = tensor
        self.size = image_size  # (image_width, image_height)
        self.mode = mode

    def __len__(self):
        return self.bbox.shape[0]

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"num_boxes={len(self)}, "
            f"image_width={self.size[0]}, "
            f"image_height={self.size[1]}, "
            f"mode={self.mode})"
        )

    # -------------------------
    # conversions and views
    # -------------------------
    def _split_into_xyxy(self):
        if self.mode == "xyxy":
            x_min, y_min, x_max, y_max = self.bbox.split(1, dim=-1)
            return x_min, y_min, x_max, y_max
        if self.mode == "xywh":
            x_c, y_c, w, h = self.bbox.split(1, dim=-1)
            return (x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h)
        raise RuntimeError("Unsupported mode: {}".format(self.mode))

    def convert(self, mode):
        """Convert bounding box format between xyxy and xywh."""
        if mode not in ("xyxy", "xywh"):
            raise ValueError("mode should be 'xyxy' or 'xywh'")
        if mode == self.mode:
            return self

        x_min, y_min, x_max, y_max = self._split_into_xyxy()
        if mode == "xyxy":
            out = torch.cat((x_min, y_min, x_max, y_max), dim=-1)
            return BoxList(out, self.size, mode="xyxy")

        # to xywh (center based)
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        w = (x_max - x_min)
        h = (y_max - y_min)
        out = torch.cat((cx, cy, w, h), dim=-1)
        return BoxList(out, self.size, mode="xywh")

    # -------------------------
    # geometry ops
    # -------------------------
    def area(self):
        """Compute area of bounding boxes."""
        if self.mode == "xyxy":
            return (self.bbox[:, 2] - self.bbox[:, 0]) * (self.bbox[:, 3] - self.bbox[:, 1])
        if self.mode == "xywh":
            return self.bbox[:, 2] * self.bbox[:, 3]
        raise RuntimeError("Unsupported mode: {}".format(self.mode))

    def resize(self, size, *args, **kwargs):
        """Resize bounding boxes to a new image size."""
        scale_w = float(size[0]) / float(self.size[0])
        scale_h = float(size[1]) / float(self.size[1])

        if scale_w == scale_h:
            scaled = self.bbox * scale_w
            return BoxList(scaled, size, mode=self.mode)

        x_min, y_min, x_max, y_max = self._split_into_xyxy()
        x_min = x_min * scale_w
        x_max = x_max * scale_w
        y_min = y_min * scale_h
        y_max = y_max * scale_h
        stacked = torch.cat((x_min, y_min, x_max, y_max), dim=-1)
        return BoxList(stacked, size, mode="xyxy").convert(self.mode)

    def transpose(self, method):
        """Flip bounding boxes horizontally or vertically."""
        if method not in (FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM):
            raise NotImplementedError("Only FLIP_LEFT_RIGHT and FLIP_TOP_BOTTOM implemented")

        width, height = self.size
        x_min, y_min, x_max, y_max = self._split_into_xyxy()
        if method == FLIP_LEFT_RIGHT:
            x_min_t = width - x_max
            x_max_t = width - x_min
            y_min_t = y_min
            y_max_t = y_max
        else:  # FLIP_TOP_BOTTOM
            x_min_t = x_min
            x_max_t = x_max
            y_min_t = height - y_max
            y_max_t = height - y_min

        stacked = torch.cat((x_min_t, y_min_t, x_max_t, y_max_t), dim=-1)
        return BoxList(stacked, self.size, mode="xyxy").convert(self.mode)

    def crop(self, region):
        """Crop bounding boxes to the specified rectangular region.

        region: (top, left, height, width) as returned by torchvision RandomCrop
        """
        top, left, height, width = region
        x_min, y_min, x_max, y_max = self._split_into_xyxy()
        x_min_c = (x_min - left).clamp(min=0, max=width)
        y_min_c = (y_min - top).clamp(min=0, max=height)
        x_max_c = (x_max - left).clamp(min=0, max=width)
        y_max_c = (y_max - top).clamp(min=0, max=height)

        stacked = torch.cat((x_min_c, y_min_c, x_max_c, y_max_c), dim=-1)
        return BoxList(stacked, (width, height), mode="xyxy").convert(self.mode)

    def check_crop_valid(self, region):
        """Return True if no box becomes degenerate after cropping."""
        top, left, height, width = region
        x_min, y_min, x_max, y_max = self._split_into_xyxy()
        x_min_c = (x_min - left).clamp(min=0, max=width)
        y_min_c = (y_min - top).clamp(min=0, max=height)
        x_max_c = (x_max - left).clamp(min=0, max=width)
        y_max_c = (y_max - top).clamp(min=0, max=height)

        degenerate = (x_min_c == x_max_c) | (y_min_c == y_max_c)
        return (~degenerate).all().item()

    def normalize(self):
        """Normalize coordinates to [0, 1] relative to image size.

        Returns boxes converted to xywh format for downstream usage.
        """
        x_min, y_min, x_max, y_max = self._split_into_xyxy()
        width, height = self.size
        x_min = x_min / width
        y_min = y_min / height
        x_max = x_max / width
        y_max = y_max / height
        stacked = torch.cat((x_min, y_min, x_max, y_max), dim=-1)
        return BoxList(stacked, self.size, mode="xyxy").convert("xywh")

    # -------------------------
    # tensor-y helpers
    # -------------------------
    def to(self, device):
        return BoxList(self.bbox.to(device), self.size, self.mode)

    def __getitem__(self, item):
        return BoxList(self.bbox[item], self.size, self.mode)

    def copy(self):
        return BoxList(self.bbox, self.size, self.mode)


if __name__ == "__main__":
    bbox = BoxList([[0, 0, 10, 10], [0, 0, 5, 5]], (10, 10))
    s_bbox = bbox.resize((5, 5))
    print(s_bbox)
    print(s_bbox.bbox)

    t_bbox = bbox.transpose(0)
    print(t_bbox)
    print(t_bbox.bbox)


