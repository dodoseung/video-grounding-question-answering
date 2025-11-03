from .vidstg_eval import VidSTGEvaluator

def build_evaluator(cfg, logger, mode):
    return VidSTGEvaluator(
        logger,
        cfg.DATA_DIR,
        mode,
        iou_thresholds=[0.3, 0.5],
        save_pred=(mode=='test'),
        save_dir=cfg.OUTPUT_DIR,
    )