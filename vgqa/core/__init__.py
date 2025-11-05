"""Core grounding model components for VGQA."""

from .grounding_net import VSTGNet
from .loss import VideoSTGLoss
from .postprocessor import PostProcess


def build_model(cfg):
    """Build the grounding model and loss function.

    Args:
        cfg: Configuration object with model and solver settings

    Returns:
        tuple: (model, loss_model, weight_dict)
    """
    model = VSTGNet(cfg)

    weight_dict = {
        "loss_bbox": cfg.SOLVER.BBOX_COEF,
        "loss_giou": cfg.SOLVER.GIOU_COEF,
        "loss_sted": cfg.SOLVER.TEMP_COEF,
        "logits_f_m": cfg.SOLVER.CONF_COEF,
        "logits_f_a": cfg.SOLVER.CONF2_COEF,
        "logits_r_a": cfg.SOLVER.CONF3_COEF,
        "logits_r_m": cfg.SOLVER.CONF4_COEF,
    }

    if cfg.MODEL.VSTG.USE_ACTION:
        weight_dict["loss_actioness"] = cfg.SOLVER.ACTIONESS_COEF

    if cfg.SOLVER.USE_ATTN:
        weight_dict["loss_guided_attn"] = cfg.SOLVER.ATTN_COEF

    if cfg.SOLVER.USE_AUX_LOSS:
        aux_weight_dict = {}
        for i in range(cfg.MODEL.VSTG.DEC_LAYERS - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ["boxes", "sted", "logits_f_m", "logits_f_a", "logits_r_a", "logits_r_m"]
    if cfg.SOLVER.USE_ATTN:
        losses += ["guided_attn"]
    if cfg.MODEL.VSTG.USE_ACTION:
        losses += ["actioness"]

    loss_model = VideoSTGLoss(cfg, losses)

    return model, loss_model, weight_dict


def build_postprocessors():
    """Build the post-processor for model outputs."""
    return PostProcess()


__all__ = [
    'build_model',
    'VSTGNet',
    'VideoSTGLoss',
    'build_postprocessors',
    'PostProcess',
]
