"""Evaluation entrypoint for spatio-temporal video grounding."""

import os
import sys
import argparse

# Ensure PYTHONPATH (os.path-based)
_CURR_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.normpath(os.path.join(_CURR_DIR, os.pardir))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

import torch
import torch.backends.cudnn as cudnn

from vgqa.config import cfg
from vgqa.utils.distributed import synchronize, get_rank, is_main_process
from vgqa.utils.log_setup import setup_logger
from vgqa.utils.training_utils import mkdir, set_seed
from vgqa.utils.checkpoint_manager import VSTGCheckpointer
from vgqa.data import make_data_loader, build_evaluator, build_dataset
from vgqa.core import build_model, build_postprocessors
from vgqa.training import do_eval


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Spatio-Temporal Grounding Evaluation"
    )
    parser.add_argument(
        "--config-file",
        default="experiments/vidstg.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--use-seed",
        dest="use_seed",
        help="use deterministic seed",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options from the command line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args()


def _init_distributed(local_rank: int) -> bool:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_distributed = world_size > 1
    if is_distributed:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()
    return is_distributed


def _load_config(config_file: str, cli_opts):
    if config_file:
        cfg.merge_from_file(config_file)
    cfg.merge_from_list(cli_opts)
    cfg.freeze()


def _prepare_environment(output_dir: str):
    if output_dir:
        mkdir(output_dir)
    # Disable tokenizers parallelism (suppress Hugging Face warnings)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def run():
    args = _parse_args()

    # Initialize distributed environment
    args.distributed = _init_distributed(args.local_rank)

    # Load configuration
    _load_config(args.config_file, args.opts)

    # Deterministic setup
    if args.use_seed:
        cudnn.benchmark = False
        cudnn.deterministic = True
        set_seed(args.seed + get_rank())

    # Prepare output directory and logger
    _prepare_environment(cfg.OUTPUT_DIR)
    logger = setup_logger("Video Grounding", cfg.OUTPUT_DIR, get_rank())
    num_gpus = int(os.environ.get("WORLD_SIZE", "1"))
    logger.info(f"GPUs in use: {num_gpus}")
    logger.info(cfg)

    # Build model and checkpointer
    model, _, _ = build_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    checkpointer = VSTGCheckpointer(cfg, model, logger=logger, is_train=False)
    eval_weight = getattr(cfg.MODEL, "WEIGHT_EVAL", None) or cfg.MODEL.WEIGHT
    _ = checkpointer.load(eval_weight, with_optim=False)

    # Dataset cache (main process only)
    if is_main_process():
        _ = build_dataset(cfg, split="test", transforms=None)
    synchronize()

    # Build data loader
    test_loader = make_data_loader(
        cfg,
        mode="test",
        is_distributed=args.distributed,
    )

    # Run evaluation
    logger.info("Start testing")
    evaluator = build_evaluator(cfg, logger, mode="test")
    postprocessor = build_postprocessors()
    do_eval(
        cfg,
        mode="test",
        logger=logger,
        model=model,
        postprocessor=postprocessor,
        data_loader=test_loader,
        evaluator=evaluator,
        device=device,
    )
    synchronize()


if __name__ == "__main__":
    run()
