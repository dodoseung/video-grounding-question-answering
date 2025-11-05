# Evaluation script for spatio-temporal video grounding model
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import argparse
import os

import torch
import torch.backends.cudnn as cudnn

from vgqa.config import cfg
from vgqa.utils.comm import synchronize, get_rank, is_main_process
from vgqa.utils.logger import setup_logger
from vgqa.utils.misc import mkdir, set_seed
from vgqa.utils.checkpoint import VSTGCheckpointer
from vgqa.data import make_data_loader, build_evaluator, build_dataset
from vgqa.core import build_model, build_postprocessors
from vgqa.training import do_eval


def main():
    parser = argparse.ArgumentParser(description="Spatio-Temporal Grounding Evaluation")
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
        help="If use the random seed",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    # Setup distributed training
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    # Load and merge configuration
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # Set random seed for reproducibility
    if args.use_seed:
        cudnn.benchmark = False
        cudnn.deterministic = True
        set_seed(args.seed + get_rank())

    # Setup output directory and logger
    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("Video Grounding", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Build model and load checkpoint
    model, _, _ = build_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    checkpointer = VSTGCheckpointer(cfg, model, logger=logger, is_train=False)
    # Use WEIGHT_EVAL for evaluation if available, otherwise fallback to WEIGHT
    eval_weight = cfg.MODEL.WEIGHT_EVAL if hasattr(cfg.MODEL, 'WEIGHT_EVAL') and cfg.MODEL.WEIGHT_EVAL else cfg.MODEL.WEIGHT
    _ = checkpointer.load(eval_weight, with_optim=False)

    # Prepare dataset cache on main process
    if args.local_rank == 0:
        _ = build_dataset(cfg, split='test', transforms=None)

    synchronize()

    # Create test data loader
    test_data_loader = make_data_loader(
        cfg,
        mode='test',
        is_distributed=args.distributed,
    )

    # Run evaluation
    logger.info("Start Testing")
    evaluator = build_evaluator(cfg, logger, mode='test')
    postprocessor = build_postprocessors()
    do_eval(
        cfg,
        mode='test',
        logger=logger,
        model=model,
        postprocessor=postprocessor,
        data_loader=test_data_loader,
        evaluator=evaluator,
        device=device
    )
    synchronize()


if __name__ == "__main__":
    main()
