from typing import Any

from .vidstg_evaluator import VidSTGEvaluator


def build_evaluator(cfg: Any, logger, mode: str) -> VidSTGEvaluator:
    """Factory for VidSTG evaluator with project defaults."""
    return VidSTGEvaluator(
        logger,
        cfg.DATA_DIR,
        mode,
        iou_thresholds=[0.3, 0.5],
        save_pred=(mode == 'test'),
        save_dir=cfg.OUTPUT_DIR,
    )