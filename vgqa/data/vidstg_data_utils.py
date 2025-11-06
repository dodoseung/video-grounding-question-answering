import torch
import random
import math
import re
import numpy as np
from copy import copy
from typing import Any, Dict, List, Tuple
from pytorch_pretrained_bert.tokenization import BertTokenizer

from .gaussian_heatmap import gaussian_radius, draw_umich_gaussian


SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')


def _choose_temporal_slice(candidate_indices: List[int], min_len: int, max_len: int) -> Tuple[int, int]:
    """Choose a [start, end] index pair from candidate indices that satisfies a min length.

    If valid range cannot be sampled, returns the widest possible range.
    """
    if not candidate_indices:
        return 0, max_len - 1

    start_idx = min(candidate_indices)
    end_idx = max(candidate_indices)
    if end_idx - start_idx + 1 >= min_len:
        return start_idx, end_idx

    # Expand to meet min length
    center = (start_idx + end_idx) // 2
    half = (min_len - 1) // 2
    start = max(0, center - half)
    end = min(max_len - 1, start + min_len - 1)
    return start, end


def _build_sampled_item(meta: Dict[str, Any], frame_ids: np.ndarray, mask: np.ndarray,
                        start_hm: np.ndarray, end_hm: np.ndarray) -> Dict[str, Any]:
    item = {
        'item_id' : meta['item_id'],
        'vid' : meta['vid'],
        'width' : meta['width'],
        'height' : meta['height'],
        'qtype' : meta['qtype'],
        'description' : meta['description'],
        'object' : meta['object'],
        'bboxs' :  meta['bboxs'],
        'gt_temp_bound' : meta['gt_temp_bound'],
        'segment_bound' : meta['segment_bound']
    }
    item.update({
        'frame_ids': frame_ids,
        'actioness': mask,
        'start_heatmap': start_hm,
        'end_heatmap': end_hm,
    })
    return item


def crop_for_2d_map(cfg: Any, video_data: Dict[str, Any]) -> Dict[str, Any]:
    """Crop video data for 2D temporal map processing."""
    if random.random() < 1 - cfg.INPUT.TEMP_CROP_PROB:
        return video_data

    if len(video_data['frame_ids']) <= cfg.MODEL.TEMPFORMER.MAX_MAP_SIZE:
        return video_data

    frames = copy(video_data['frame_ids'])
    mask = video_data['actioness'].copy()
    start_hm = video_data['start_heatmap'].copy()
    end_hm = video_data['end_heatmap'].copy()

    action_indices = list(np.where(mask)[0])
    before = [i for i in range(len(frames)) if i < action_indices[0]] if action_indices else []
    after = [i for i in range(len(frames)) if i > action_indices[-1]] if action_indices else []

    max_try = 30
    for _ in range(max_try):
        s = random.choice(before) if before else 0
        e = random.choice(after) if after else len(frames) - 1
        if e - s + 1 >= cfg.MODEL.TEMPFORMER.MAX_MAP_SIZE:
            sl = slice(s, e + 1)
            return _build_sampled_item(video_data, frames[sl], mask[sl], start_hm[sl], end_hm[sl])

    return video_data


def make_vidstg_input_clip(cfg: Any, split: str, video_data: Dict[str, Any]) -> Dict[str, Any]:
    """Select a temporal clip and optionally crop for training; sub-sample to desired length."""
    input_frame_num = cfg.INPUT.TRAIN_SAMPLE_NUM if split == 'train' else cfg.INPUT.TRAIN_SAMPLE_NUM * 2

    do_crop = split == 'train' and (random.random() < cfg.INPUT.TEMP_CROP_PROB)

    frames = copy(video_data['frame_ids'])
    mask = video_data['actioness'].copy()
    start_hm = video_data['start_heatmap'].copy()
    end_hm = video_data['end_heatmap'].copy()

    if do_crop:
        action_indices = list(np.where(mask)[0])
        if len(action_indices) == 0:
            do_crop = False
            selected_indices = list(range(0, len(frames)))
        else:
            starts = [i for i in range(len(frames)) if i < action_indices[0]]
            ends = [i for i in range(len(frames)) if i > action_indices[-1]]

            start_idx = random.choice(starts) if starts else 0
            end_idx = random.choice(ends) if ends else len(frames) - 1
            selected_indices = list(range(start_idx, end_idx + 1))

    if not do_crop:
        selected_indices = list(range(0, len(frames)))

    # Subsample to requested length
    if len(selected_indices) > input_frame_num:
        linspace_idx = np.linspace(0, len(selected_indices) - 1, num=input_frame_num)
        idxs = [int(i) for i in linspace_idx]
        assert len(set(idxs)) == len(idxs)
        selected_indices = [selected_indices[i] for i in idxs]

    return _build_sampled_item(
        video_data,
        np.array([frames[i] for i in selected_indices]),
        mask[selected_indices],
        start_hm[selected_indices],
        end_hm[selected_indices],
    )


def crop_clip(cfg: Any, video_data: Dict[str, Any]) -> Dict[str, Any]:
    """Random crop a video clip while preserving its groundtruth."""
    if random.random() < 1 - cfg.INPUT.TEMP_CROP_PROB:
        return video_data

    frames = copy(video_data['frame_ids'])
    mask = video_data['actioness'].copy()
    start_hm = video_data['start_heatmap'].copy()
    end_hm = video_data['end_heatmap'].copy()

    action_indices = list(np.where(mask)[0])
    starts = [i for i in range(len(frames)) if i < action_indices[0]] if action_indices else [0]
    ends = [i for i in range(len(frames)) if i > action_indices[-1]] if action_indices else [len(frames) - 1]

    start_idx = random.choice(starts) if starts else 0
    end_idx = random.choice(ends) if ends else len(frames) - 1

    sl = slice(start_idx, end_idx + 1)
    return _build_sampled_item(video_data, frames[sl], mask[sl], start_hm[sl], end_hm[sl])


def iou(candidates: torch.Tensor, gt: List[int]) -> torch.Tensor:
    start, end = candidates[:, 0], candidates[:, 1]
    s, e = torch.tensor([gt[0]]).float(), torch.tensor([gt[1]]).float()
    inter = end.min(e) - start.max(s)
    union = end.max(e) - start.min(s)
    return inter.clamp(min=0) / union


def score2d_to_moments_scores(score2d: torch.Tensor, num_clips: int, duration: int):
    grids = score2d.nonzero()
    scores = score2d[grids[:, 0], grids[:, 1]]
    grids[:, 1] += 1
    moments = grids * duration / num_clips
    return moments, scores


def make_2dmap(cfg: Any, video_data: Dict[str, Any]):
    num_clips = cfg.MODEL.TEMPFORMER.MAX_MAP_SIZE
    iou2d = torch.ones(num_clips, num_clips)
    duration = video_data['frame_ids'][-1] - video_data['frame_ids'][0] + 1
    candidates, _ = score2d_to_moments_scores(iou2d, num_clips, duration)
    moment = video_data['gt_temp_bound']
    iou2d = iou(candidates, moment).reshape(num_clips, num_clips)
    return iou2d, candidates


def sample_clip(cfg: Any, video_data: Dict[str, Any]) -> Dict[str, Any]:
    """Sample a small video clip and its groundtruth (used for visualization/aux tasks)."""
    data_item = {
        'gt_file' : video_data['gt_file'],
        'width' : video_data['width'],
        'height' : video_data['height'],
        'description' : video_data['description'],
        'object' : video_data['object']
    }
    boxs = video_data['bboxs'].copy()
    video_frames = copy(video_data['frame_names'])
    gt_mask = video_data['actioness'].copy()
    start_heatmap = video_data['start_heatmap'].copy()
    end_heatmap = video_data['end_heatmap'].copy()

    gt_temp_length = boxs.shape[0]
    clip_length = cfg.DATASET.NUM_CLIP_FRAMES
    min_gt_num = min(cfg.DATASET.MIN_GT_FRAME, gt_temp_length)

    video_length = len(video_frames)
    assert gt_mask.shape[0] == video_length

    action_span = np.where(gt_mask)[0]
    min_start_idx = max(0, action_span[0] + min_gt_num - clip_length)
    max_start_idx = min(max(0, video_length - clip_length), action_span[-1] - min_gt_num + 1)

    start_idx = random.choice(list(range(min_start_idx, max_start_idx + 1)))
    sample_slice = slice(start_idx, start_idx + clip_length)
    bbox_slice = slice(max(0, start_idx - action_span[0]), start_idx + clip_length - action_span[0])
    data_item.update({
        'frame_names' : video_frames[sample_slice],
        'actioness' : gt_mask[sample_slice],
        'start_heatmap' : start_heatmap[sample_slice],
        'end_heatmap' : end_heatmap[sample_slice],
        'bboxs' : boxs[bbox_slice]}
    )
    assert np.where(data_item['actioness'])[0].shape[0] == data_item['bboxs'].shape[0]

    return data_item
    

def make_heatmap(cfg: Any, input_dict: Dict[str, Any]):
    """Generate the Gaussian heatmap, width/height, and center offset targets per frame."""
    video_clip = input_dict['frames']
    bboxs = input_dict['boxs'].bbox
    gt_mask = input_dict['actioness']
    action_span = np.where(gt_mask)[0]

    input_t = video_clip.shape[0]
    input_h = video_clip.shape[-2]
    input_w = video_clip.shape[-1]
    output_h = input_h // cfg.MODEL.DOWN_RATIO
    output_w = input_w // cfg.MODEL.DOWN_RATIO

    hm = np.zeros((input_t, output_h, output_w), dtype=np.float32)
    wh = np.zeros((input_t, 2), dtype=np.float32)
    offset = np.zeros((input_t, 2), dtype=np.float32)

    for box_idx in range(len(bboxs)):
        bbox = bboxs[box_idx].numpy()
        bbox[0] = bbox[0] * (output_w / input_w)
        bbox[1] = bbox[1] * (output_h / input_h)
        bbox[2] = bbox[2] * (output_w / input_w)
        bbox[3] = bbox[3] * (output_h / input_h)

        h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
        assert h > 0 and w > 0

        radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, int(radius))
        ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
        ct_int = ct.astype(np.int32)

        assert 0 <= ct_int[0] and ct_int[0] <= output_w and 0 <= ct_int[1] and ct_int[1] <= output_h

        frame_idx = action_span[box_idx]
        draw_umich_gaussian(hm[frame_idx], ct_int, radius)
        wh[frame_idx] = (1.0 * w, 1.0 * h)
        offset[frame_idx] = ct - ct_int

    return hm, wh, offset


## Bert text encoding
class InputExample(object):
    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b

class InputFeatures(object):
    """A single set of BERT input features."""
    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


def read_examples(input_line: str, unique_id: int):
    """Create a list with a single `InputExample` from raw text."""
    examples = []
    line = input_line.strip()
    text_a = None
    text_b = None
    m = re.match(r"^(.*) \|\|\| (.*)$", line)
    if m is None:
        text_a = line
    else:
        text_a = m.group(1)
        text_b = m.group(2)

    examples.append(InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
    return examples


def convert_examples_to_features(examples: List[InputExample], seq_length: int, tokenizer):
    """Convert examples into BERT-compatible features (single-sentence mode)."""
    features: List[InputFeatures] = []
    for (_, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Multi-sentence mode not supported in this project yet
            raise NotImplementedError
        else:
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length
        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids,
            )
        )
    return features


def make_word_tokens(cfg: Any, sentence: str, index: int, vocab=None):
    max_query_len = cfg.INPUT.MAX_QUERY_LEN

    if cfg.MODEL.USE_LSTM:
        words = sentence.strip().split()
        word_idx = torch.tensor([vocab.stoi.get(w.lower(), 400000) for w in words], dtype=torch.long)
        padded_word_idx = torch.zeros(max_query_len, dtype=torch.long)
        padded_word_idx[: word_idx.shape[0]] = word_idx
        word_mask = torch.zeros(max_query_len, dtype=torch.long)
        word_mask[: word_idx.shape[0]] = 1
        word_idx = padded_word_idx
    else:
        bert_model = 'bert-base-uncased'
        tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)
        examples = read_examples(sentence, index)
        features = convert_examples_to_features(
            examples=examples, seq_length=max_query_len, tokenizer=tokenizer
        )
        word_idx = torch.tensor(features[0].input_ids, dtype=torch.long)
        word_mask = torch.tensor(features[0].input_mask, dtype=torch.long)

    return word_idx, word_mask


