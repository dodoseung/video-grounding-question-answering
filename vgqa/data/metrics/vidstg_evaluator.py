import os
import json
from functools import reduce
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch

from vgqa.utils.distributed import is_main_process, all_gather
from vgqa.utils.box_ops import np_box_iou

def save_json(path: str, data: Dict[str, Any]) -> None:
    """Save a dictionary to a JSON file."""
    with open(path, "w") as f:
        json.dump(data, f)

class VidSTGiouEvaluator:
    """Evaluator for VidSTG temporal and spatial IoU metrics."""

    def __init__(self, vidstg_path: str, subset: str = "test", iou_thresholds: Optional[List[float]] = None):
        assert subset in ["train", "test", "val"], f"Wrong VidSTG subset {subset}"

        cache_dir = os.path.join(vidstg_path, "data_cache")
        dataset_cache = os.path.join(cache_dir, f"vidstd-{subset}-anno.cache")
        gt_data = torch.load(dataset_cache)

        self.vid2steds: Dict[int, List[int]] = {}
        self.vid2box: Dict[int, Dict[int, List[List[float]]]] = {}
        self.vid2names: Dict[int, Any] = {}
        self.vid2sents: Dict[int, str] = {}

        for data_item in gt_data:
            item_id = data_item["item_id"]
            temp_gt = data_item["gt_temp_bound"]
            self.vid2names[item_id] = data_item["item_id"]  # kept behavior
            self.vid2sents[item_id] = data_item["description"]
            box_dict = data_item["bboxs"]
            self.vid2box[item_id] = {fid: [box_dict[fid]] for fid in box_dict}
            self.vid2steds[item_id] = temp_gt

        self.iou_thresholds = iou_thresholds or [0.3, 0.5]

    def evaluate(
        self,
        predictions: Dict[int, Dict[int, List[List[float]]]],
        video_predictions: Dict[int, Dict[str, Any]],
        pred_conf: Dict[int, Any],
        pred_kf: Dict[int, Tuple[float, float]],
    ):
        """Evaluate predictions and compute temporal and spatial IoU metrics."""
        vid_metrics: Dict[int, Dict[str, Any]] = {}
        for video_id, video_pred in video_predictions.items():
            if video_id in vid_metrics:
                print(f"Warning, multiple predictions found for video {video_id}")
                continue
            
            gt_sted = self.vid2steds[video_id]
            pred_sted = video_pred["sted"]
            qtype = video_pred["qtype"]

            # compute temporal iou
            max_start = max(gt_sted[0], pred_sted[0])
            min_end = min(gt_sted[1], pred_sted[1])
            min_start = min(gt_sted[0], pred_sted[0])
            max_end = max(gt_sted[1], pred_sted[1])
            if min_end <= max_start:
                tiou = 0
            else:
                intersection = min_end - max_start
                gt_span = gt_sted[1] - gt_sted[0]
                pred_span = pred_sted[1] - pred_sted[0]
                union = gt_span + pred_span - intersection
                tiou = intersection / union

            # compute viou and gt_viou
            vid_metrics[video_id] = {
                "gt_sted": gt_sted,
                "pred_sted": pred_sted,
                "tiou": tiou,
                "qtype": qtype,
                "img_metrics": {},
            }

            union_predgt = set([
                frame_id for frame_id in range(min_start, max_end)
            ])
            inter_predgt = set(
                [frame_id for frame_id in range(max_start, min_end)]
            )

            viou = 0
            gt_viou = 0
            prediction = predictions.get(video_id, {})

            for fid in self.vid2box[video_id].keys():
                if fid not in prediction:
                    continue
                pred_boxes = prediction[fid]
                gt_boxes = self.vid2box[video_id][fid]
                iou = np_box_iou(np.array(pred_boxes), np.array(gt_boxes))[0][0]
                if fid in inter_predgt:
                    viou += iou
                gt_viou += iou

            viou = viou / max(len(union_predgt), 1)
            vid_metrics[video_id]["viou"] = viou
            recalls = {th: 0 for th in self.iou_thresholds}
            for th in self.iou_thresholds:
                if viou > th:
                    recalls[th] += 1
            vid_metrics[video_id].update(
                {
                    f"viou@{th}": recalls[th]
                    for th in self.iou_thresholds
                }
            )

            # compute gt_viou@R
            gt_viou = gt_viou / max(len(self.vid2box[video_id]), 1)
            vid_metrics[video_id]["gt_viou"] = gt_viou
            gt_recalls = {th: 0 for th in self.iou_thresholds}
            for th in self.iou_thresholds:
                if gt_viou > th:
                    gt_recalls[th] += 1
            vid_metrics[video_id].update(
                {
                    f"gt_viou@{th}": gt_recalls[th]
                    for th in self.iou_thresholds
                }
            )


        for vid, kf_pr in pred_kf.items():
            vid_metrics[vid]['kf_pr'] = kf_pr

        return vid_metrics, self.vid2names, self.vid2sents


class VidSTGEvaluator(object):
    def __init__(
        self,
        logger,
        vidstg_path: str,
        subset: str,
        iou_thresholds: List[float],
        save_pred: bool = False,
        save_dir: Optional[str] = None,
    ):
        """High-level evaluator wrapper aggregating predictions across processes."""
        self.evaluator = VidSTGiouEvaluator(vidstg_path, subset=subset, iou_thresholds=iou_thresholds)
        self.predictions: Dict[int, Dict[int, List[List[float]]]] = {}
        self.att_predictions: Dict[int, Any] = {}
        self.confs: Dict[int, Any] = {}
        self.video_predictions: Dict[int, Dict[str, Any]] = {}
        self.video_cross_attn: Dict[int, Any] = {}
        self.kf_pred: Dict[int, Tuple[float, float]] = {}
        self.results: Optional[Dict[int, Dict[str, Any]]] = None
        self.iou_thresholds = iou_thresholds
        self.save_pred = save_pred
        self.save_dir = save_dir
        self.logger = logger

        self.tsa_weights: Dict[int, Any] = {}
        self.text_weights: Dict[int, Any] = {}
        self.spatial_weights: Dict[int, Any] = {}
        self.pred_sted: Dict[int, Any] = {}

    def accumulate(self):
        return None

    def update(self, predictions: Dict[int, Dict[int, List[List[float]]]]):
        self.predictions.update(predictions)

    def update_att(self, predictions: Dict[int, Any]):
        self.att_predictions.update(predictions)

    def update_conf(self, confs: Dict[int, Any]):
        self.confs.update(confs)

    def update_kf_pr(self, kf_pr: Dict[int, Tuple[float, float]]):
        self.kf_pred.update(kf_pr)

    def update_cross_attn(self, cross_weights: Dict[int, Any]):
        self.video_cross_attn.update(cross_weights)

    def video_update(self, video_predictions: Dict[int, Dict[str, Any]]):
        self.video_predictions.update(video_predictions)

    def synchronize_between_processes(self):
        all_predictions = all_gather(self.predictions)
        self.predictions = reduce(lambda a, b: a.update(b) or a, all_predictions, {})
        all_predictions = all_gather(self.att_predictions)
        self.att_predictions = reduce(lambda a, b: a.update(b) or a, all_predictions, {})
        all_confs = all_gather(self.confs)
        self.confs = reduce(lambda a, b: a.update(b) or a, all_confs, {})
        all_kf_preds = all_gather(self.kf_pred)
        self.kf_pred = reduce(lambda a, b: a.update(b) or a, all_kf_preds, {})
        all_video_predictions = all_gather(self.video_predictions)
        self.video_predictions = reduce(lambda a, b: a.update(b) or a, all_video_predictions, {})

    def summarize(self):
        if is_main_process():
            self.logger.info("#######  Start Calculating the metrics  ########")
            self.results, vid2names, vid2sents = self.evaluator.evaluate(
                self.predictions, self.video_predictions, self.confs, self.kf_pred
            )
            categories = set(x["qtype"] for x in self.results.values())
            metrics = {}
            counter = {}

            for category in categories:  # init metrics
                metrics[category] = {"gt_viou": 0}
                metrics[category].update({"tiou": 0, "viou": 0})
                metrics[category].update({"kf_p": 0, "kf_r": 0})
                for thresh in self.iou_thresholds:
                    metrics[category][f"viou@{thresh}"] = 0
                    metrics[category][f"gt_viou@{thresh}"] = 0
                counter[category] = 0
            result_str = ''
            result_str += '\n' + '=' * 100 + '\n'
            for x in self.results.values():  # sum results
                qtype = x["qtype"]
                metrics[qtype]["tiou"] += x["tiou"]
                metrics[qtype]["viou"] += x["viou"]
                metrics[qtype]["gt_viou"] += x["gt_viou"]
                for thresh in self.iou_thresholds: 
                    metrics[qtype][f"viou@{thresh}"] += x[f"viou@{thresh}"]
                    metrics[qtype][f"gt_viou@{thresh}"] += x[f"gt_viou@{thresh}"]
                metrics[qtype]["kf_p"] += x['kf_pr'][0]
                metrics[qtype]["kf_r"] += x['kf_pr'][1]
                counter[qtype] += 1

            for category in categories:  # average results per category
                for key in metrics[category]:
                    metrics[category][key] = metrics[category][key] / max(counter[category], 1)
                    result_str += f"{category} {key}: {metrics[category][key]:.4f}" + '\n'

            result_str += '=' * 100 + '\n'
            self.logger.info(result_str)
            
            out = {
                f"{qtype}_{name}": metrics[qtype][name]
                for qtype in metrics
                for name in metrics[qtype]
            }
            
            if self.save_pred:
                out["predictions"] = self.predictions
                out["gt"] = self.evaluator.vid2box
                out["att_sequence"] = self.att_predictions
                out["confs"] = self.confs
                out["video_predictions"] = self.video_predictions
                out["vid_metrics"] = self.results
                out['vid2names'] = vid2names
                out['vid2sents'] = vid2sents
                res_path = os.path.join(self.save_dir, 'test_results.json')
                save_json(res_path, out)

            return out

        return None


