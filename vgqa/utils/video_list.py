from __future__ import division

import torch


class NestedTensor(object):
    """Container for tensors with associated masks"""
    def __init__(self, tensors, mask):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return 'NestedTensor{}'.format(self.tensors.shape)


class TargetTensor(object):
    """Container for training target tensors including heatmaps and offsets"""

    def __init__(self, target : dict) -> None:
        self.spatial_hm = target['spatial_heatmap']
        self.wh = target['wh']
        self.offset = target['offset']
        self.actioness = target['actioness']
        self.start_hm = target['start_heatmap']
        self.end_hm = target['end_heatmap']
        self.target = target

    def to(self,device):
        return TargetTensor(
            {
                'spatial_heatmap' : self.spatial_hm.to(device),
                'wh' : self.wh.to(device),
                'offset' : self.offset.to(device),
                'actioness' : self.actioness.to(device),
                'start_heatmap' : self.spatial_hm.to(device),
                'end_heatmap' : self.end_hm.to(device)
            }
        )


class VideoList(object):
    """Container for batch of videos with padding to handle varying sizes"""

    def __init__(self, tensors, video_sizes):
        self.tensors = tensors
        self.video_sizes = video_sizes

    def to(self, *args, **kwargs):
        cast_tensor = self.tensors.to(*args, **kwargs)
        return VideoList(cast_tensor, self.video_sizes)


def to_video_list(tensors, size_divisible=0):
    """Convert tensors to VideoList with padding for uniform batch size"""

    if isinstance(tensors, torch.Tensor) and size_divisible > 0:
        tensors = [tensors]

    if isinstance(tensors, VideoList):
        return tensors
    elif isinstance(tensors, torch.Tensor):
        # single tensor shape can be inferred
        if tensors.dim() == 4:   # T * C * H * W
            tensors = tensors[None]
        assert tensors.dim() == 5
        image_sizes = [tensor.shape[-2:] for tensor in tensors]
        return VideoList(tensors, image_sizes)
    
    elif isinstance(tensors, (tuple, list)):
        max_size = tuple(max(s) for s in zip(*[img.shape for img in tensors]))

        # input sizs
        if size_divisible > 0:
            import math

            stride = size_divisible
            max_size = list(max_size)
            max_size[2] = int(math.ceil(max_size[2] / stride) * stride)
            max_size[3] = int(math.ceil(max_size[3] / stride) * stride)
            max_size = tuple(max_size)
        
        batch_shape = (len(tensors),) + max_size
        batched_imgs = tensors[0].new(*batch_shape).zero_()
        for img, pad_img in zip(tensors, batched_imgs):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2], : img.shape[3]].copy_(img)

        image_sizes = [im.shape[-2:] for im in tensors]

        return VideoList(batched_imgs, image_sizes)
    else:
        raise TypeError("Unsupported type for to_video_list: {}".format(type(tensors)))
