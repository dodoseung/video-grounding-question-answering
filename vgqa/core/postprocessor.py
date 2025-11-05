from faulthandler import dump_traceback
from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn

from vgqa.utils.box_utils import box_cxcywh_to_xyxy

    
class PostProcess(nn.Module):
    """Post-process model outputs to final predictions"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes, frames_id, durations):
        """Convert model outputs to final bounding boxes and temporal predictions"""
        # Extract predictions from model outputs
        out_sted, out_bbox, kf_pr = outputs["pred_sted"], outputs["pred_boxes"], outputs["pr"]
        out_att = outputs['att_sequences']
        assert len(out_bbox) == len(target_sizes)

        # Convert bounding boxes to absolute coordinates
        boxes = box_cxcywh_to_xyxy(out_bbox)
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        pred_boxes = boxes * scale_fct

        # Clamp boxes to image boundaries
        pred_boxes = pred_boxes.clamp(min=0)

        # Compute temporal probability map
        b, t, _ = out_sted.shape
        device = out_sted.device
        temp_prob_map = torch.zeros(b,t,t).to(device)
        inf = -1e32

        # Mask invalid temporal regions
        for i_b in range(len(durations)):
            duration = durations[i_b]
            sted_prob = (torch.ones(t, t) * inf).tril(0).to(device)
            sted_prob[duration:,:] = inf
            sted_prob[:,duration:] = inf
            temp_prob_map[i_b,:,:] = sted_prob

        # Add start/end probabilities
        temp_prob_map += F.log_softmax(out_sted[:, :, 0], dim=1).unsqueeze(2) + \
                F.log_softmax(out_sted[:, :, 1], dim=1).unsqueeze(1)

        # Find optimal start/end frames
        pred_steds = []
        for i_b in range(b):
            prob_map = temp_prob_map[i_b]
            frame_id_seq = frames_id[i_b]
            prob_seq = prob_map.flatten(0)
            max_tstamp = prob_seq.max(dim=0)[1].item()
            start_idx = max_tstamp // t
            end_idx = max_tstamp % t
            pred_sted = [frame_id_seq[start_idx], frame_id_seq[end_idx]+1]
            pred_steds.append(pred_sted)
    
        return pred_boxes, out_att, pred_steds, kf_pr

