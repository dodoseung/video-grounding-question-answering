import torch
import torch.nn
from typing import Dict, List, Any

from vgqa.utils.training_utils import to_device
from vgqa.utils.distributed import synchronize, is_main_process
from tqdm import tqdm


@torch.no_grad()
def linear_interp(bbox_dict: Dict[int, List[List[float]]]):
    """Perform linear interpolation between bounding boxes.

    bbox_dict: {frame_id: [[x1,y1,x2,y2]]}
    """
    frame_ids = sorted(bbox_dict.keys())
    if len(frame_ids) < 2:
        return bbox_dict
    for idx in range(0, len(frame_ids) - 1):
        left_fid = frame_ids[idx]
        right_fid = frame_ids[idx + 1]
        if right_fid - left_fid > 1:
            interval = right_fid - left_fid
            x1_l, y1_l, x2_l, y2_l = bbox_dict[left_fid][0]
            x1_r, y1_r, x2_r, y2_r = bbox_dict[right_fid][0]
            dx1 = (x1_r - x1_l) / interval
            dy1 = (y1_r - y1_l) / interval
            dx2 = (x2_r - x2_l) / interval
            dy2 = (y2_r - y2_l) / interval
            for step in range(1, interval):
                bbox_dict[left_fid + step] = [[x1_l + step * dx1, y1_l + step * dy1, x2_l + step * dx2, y2_l + step * dy2]]

    frame_ids = sorted(bbox_dict.keys())
    assert max(frame_ids) - min(frame_ids) + 1 == len(frame_ids)
    return {fid: bbox_dict[fid] for fid in frame_ids}


@torch.no_grad()
def linear_interp_conf(conf_dict: Dict[int, Any]):
    """Perform interpolation for confidence scores (left/right hold)."""
    frame_ids = sorted(conf_dict.keys())
    if len(frame_ids) < 2:
        return conf_dict
    for idx in range(0, len(frame_ids) - 1):
        left_fid = frame_ids[idx]
        right_fid = frame_ids[idx + 1]
        if right_fid - left_fid > 1:
            interval = right_fid - left_fid
            for step in range(1, interval):
                conf_dict[left_fid + step] = conf_dict[left_fid] if step <= (interval // 2) else conf_dict[right_fid]

    frame_ids = sorted(conf_dict.keys())
    assert max(frame_ids) - min(frame_ids) + 1 == len(frame_ids)
    return {fid: conf_dict[fid] for fid in frame_ids}

@torch.no_grad()
def single_forward(cfg, model, videos, texts, targets, device, postprocessor):
    """Run single forward pass for evaluation"""
    durations = videos.durations
    targets[0]["durations"] = durations
    outputs = model(videos, texts, targets)

    b = len(durations)
    t = max(durations)
    img_sizes = [list(target['ori_size']) for target in targets]
    orig_target_sizes = [s for s in img_sizes for _ in range(t)]
    orig_target_sizes = torch.tensor(orig_target_sizes, device=device)
    assert orig_target_sizes.shape[0] == outputs['pred_boxes'].shape[0]

    frame_ids: List[List[int]] = [target['frame_ids'] for target in targets]
    pred_boxs, pred_att, pred_steds, pred_kf = postprocessor(outputs, orig_target_sizes, frame_ids, durations)
    pred_boxs = pred_boxs.view(b, t, 4)

    vids = [target['item_id'] for target in targets]
    bbox_pred, temp_pred, kf_pred = {}, {}, {}
    att_pred = {}

    for i_b in range(b):
        fids = frame_ids[i_b]
        bbox_pred[vids[i_b]] = {}
        att_pred[vids[i_b]] = {}
        assert durations[i_b] == len(fids)
        for idx in range(durations[i_b]):
            bbox_pred[vids[i_b]][fids[idx]] = [pred_boxs[i_b][idx].detach().cpu().tolist()]
            att_pred[vids[i_b]][fids[idx]] = [pred_att[i_b][idx].detach().cpu().tolist()]

    qtypes = [target['qtype'] for target in targets]
    assert len(pred_steds) == len(qtypes)
    for i_b in range(b):
        temp_pred[vids[i_b]] = {"sted": pred_steds[i_b], "qtype": qtypes[i_b]}
    kf_pred[vids[0]] = pred_kf
    return bbox_pred, att_pred, temp_pred, kf_pred
    

@torch.no_grad()
def do_eval(cfg, mode, logger, model, postprocessor, data_loader, evaluator, device):
    """Evaluate video spatial-temporal grounding model"""
    model.eval()
    logger.info("Start evaluation on the {} split of {} dataset".format(mode, cfg.DATASET.NAME))

    for _, batch_dict in enumerate(tqdm(data_loader)):
        videos = batch_dict['videos'].to(device)
        texts = batch_dict['texts']
        targets = to_device(batch_dict["targets"], device)

        for i in range(len(targets)):
            if 'qtype' not in targets[i]:
                targets[i]['qtype'] = 'none'

        videos1 = videos.subsample(2, start_idx=0)
        targets1 = [{'item_id': t['item_id'], 'ori_size': t['ori_size'], 'vid': t['vid'], 'orignal_frame': t['orignal_frame'][0::2],
                     'qtype': t['qtype'], 'frame_ids': t['frame_ids'][0::2], "boxs": t['boxs'].bbox.clone(), 'actioness': t['actioness'][0::2], "eval": True} for t in targets]

        videos2 = videos.subsample(2, start_idx=1)
        targets2 = [{'item_id': t['item_id'], 'ori_size': t['ori_size'], 'vid': t['vid'], 'orignal_frame': t['orignal_frame'][1::2],
                     'qtype': t['qtype'], 'frame_ids': t['frame_ids'][1::2], "boxs": t['boxs'].bbox.clone(), 'actioness': t['actioness'][1::2], "eval": True} for t in targets]

        if torch.where(targets[0]["actioness"])[0][0] % 2 == 0:
            targets1[0]['boxs'] = targets1[0]['boxs'][0::2]
            targets2[0]['boxs'] = targets2[0]['boxs'][1::2]
        else:
            targets1[0]['boxs'] = targets1[0]['boxs'][1::2]
            targets2[0]['boxs'] = targets2[0]['boxs'][0::2]

        bbox_pred1, att_pred1, temp_pred1, kf_pred1 = single_forward(cfg, model, videos1, texts, targets1, device, postprocessor)
        bbox_pred2, att_pred2, temp_pred2, kf_pred2 = single_forward(cfg, model, videos2, texts, targets2, device, postprocessor)

        bbox_pred, att_pred, temp_pred, kf_pred = {}, {}, {}, {}
        for vid in bbox_pred1:
            bbox_pred1[vid].update(bbox_pred2[vid])
            bbox_pred[vid] = linear_interp(bbox_pred1[vid])
            att_pred1[vid].update(att_pred2[vid])
            att_pred[vid] = linear_interp_conf(att_pred1[vid])
            kf_pred[vid] = [(kf_pred1[vid][0] + kf_pred2[vid][0]) / 2, (kf_pred1[vid][1] + kf_pred2[vid][1]) / 2]
            temp_pred[vid] = {'sted': [min(temp_pred1[vid]['sted'][0], temp_pred2[vid]['sted'][0]),
                                       max(temp_pred1[vid]['sted'][1], temp_pred2[vid]['sted'][1])]}
            if 'qtype' in temp_pred1[vid]:
                temp_pred[vid]['qtype'] = temp_pred1[vid]['qtype']

        evaluator.update(bbox_pred)
        evaluator.update_att(att_pred)
        evaluator.update_kf_pr(kf_pred)
        evaluator.video_update(temp_pred)

    synchronize()
    evaluator.synchronize_between_processes()
    if is_main_process():
        logger.info(f"Complete the inference on {mode} split of {cfg.DATASET.NAME}")

    res = evaluator.summarize()
    return res