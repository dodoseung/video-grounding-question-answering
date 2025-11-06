"""Training entrypoint for spatio-temporal video grounding."""

import sys
from pathlib import Path

# Add project root to PYTHONPATH (Path-based)
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import argparse
import os
import time
import datetime
from copy import deepcopy
import torch
import torch.backends.cudnn as cudnn

from vgqa.config import cfg
from vgqa.utils.distributed import (
    synchronize,
    get_rank,
    is_main_process,
    reduce_loss_dict,
)
from vgqa.utils.log_setup import setup_logger
from vgqa.utils.training_utils import mkdir, save_config, set_seed, to_device
from vgqa.utils.checkpoint_manager import VSTGCheckpointer
from vgqa.data import make_data_loader, build_evaluator, build_dataset
from vgqa.core import build_model, build_postprocessors
from vgqa.training import make_optimizer, adjust_learning_rate, update_ema, do_eval
from vgqa.utils.metrics_logger import MetricLogger
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class Trainer:
    def __init__(self, cfg, local_rank: int, distributed: bool, logger):
        self.cfg = cfg
        self.local_rank = local_rank
        self.distributed = distributed
        self.logger = logger

        self.device = torch.device(cfg.MODEL.DEVICE)
        self.model = None
        self.criteria = None
        self.weight_dict = None
        self.optimizer = None
        self.model_ema = None
        self.model_without_ddp = None

        self.arguments = {"iteration": 0}
        self.checkpointer = None
        self.verbose_loss_keys = set()

        self.train_loader = None
        self.val_loader = None
        self.writer = None
        self.pbar = None

    def setup(self):
        # Build model, loss criterion, and weight dict
        self.model, self.criteria, self.weight_dict = build_model(self.cfg)
        self.model.to(self.device)
        self.criteria.to(self.device)

        # Optimizer and EMA model
        self.optimizer = make_optimizer(self.cfg, self.model, self.logger)
        self.model_ema = deepcopy(self.model) if self.cfg.MODEL.EMA else None
        self.model_without_ddp = self.model

        # DDP
        if self.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=True,
            )
            self.model_without_ddp = self.model.module

        # Checkpointer
        save_to_disk = self.local_rank == 0
        self.checkpointer = VSTGCheckpointer(
            self.cfg,
            self.model_without_ddp,
            self.model_ema,
            self.optimizer,
            self.cfg.OUTPUT_DIR,
            save_to_disk,
            self.logger,
            is_train=True,
        )
        extra = self.checkpointer.load(self.cfg.MODEL.WEIGHT)
        self.arguments.update(extra)

        # Configure verbose loss keys
        self.verbose_loss_keys = {
            "loss_bbox",
            "loss_giou",
            "loss_sted",
            "logits_f_m",
            "logits_f_a",
            "logits_r_a",
            "logits_r_m",
        }
        if self.cfg.SOLVER.USE_ATTN:
            self.verbose_loss_keys.add("loss_guided_attn")
        if self.cfg.MODEL.VSTG.USE_ACTION:
            self.verbose_loss_keys.add("loss_actioness")

        # Dataset cache (main process only)
        if is_main_process():
            for split in ("train", "test"):
                _ = build_dataset(self.cfg, split=split, transforms=None)
        synchronize()

        # Data loaders
        self.train_loader = make_data_loader(
            self.cfg,
            mode="train",
            is_distributed=self.distributed,
            start_iter=self.arguments["iteration"],
        )
        self.val_loader = make_data_loader(
            self.cfg,
            mode="test",
            is_distributed=self.distributed,
        )

        # TensorBoard
        if self.cfg.TENSORBOARD_DIR and is_main_process():
            mkdir(self.cfg.TENSORBOARD_DIR)
            self.writer = SummaryWriter(self.cfg.TENSORBOARD_DIR)

    def _maybe_pre_validate(self):
        if self.cfg.SOLVER.PRE_VAL:
            self.logger.info("Validating before training")
            self.validate()

    def _open_progress(self, total: int, initial: int):
        if is_main_process():
            import shutil
            term_width = shutil.get_terminal_size().columns
            self.pbar = tqdm(
                total=total,
                initial=initial,
                desc="Training",
                ncols=term_width,
                dynamic_ncols=True,
                bar_format=(
                    "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                    "[{elapsed}<{remaining}, {rate_fmt}]{postfix}"
                ),
            )

    def _close_progress(self):
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None

    def fit(self):
        self.logger.info("Start training")
        metric_logger = MetricLogger(delimiter="  ")
        max_iter = len(self.train_loader)
        start_iter = self.arguments["iteration"]
        start_time = time.time()
        last_time = time.time()

        self._maybe_pre_validate()
        self._open_progress(total=max_iter, initial=start_iter)

        for step, batch in enumerate(self.train_loader, start_iter):
            self.model.train()
            self.criteria.train()

            data_time = time.time() - last_time
            step = step + 1
            self.arguments["iteration"] = step

            # Move batch to device
            videos = batch["videos"].to(self.device)
            texts = batch["texts"]
            durations = batch["durations"]
            targets = to_device(batch["targets"], self.device)
            targets[0]["durations"] = durations

            # Forward pass and loss computation
            outputs = self.model(videos, texts, targets, step / max_iter)
            loss_dict = self.criteria(outputs, targets, durations)
            losses = sum(
                loss_dict[k] * self.weight_dict[k]
                for k in loss_dict.keys()
                if k in self.weight_dict
            )

            # Reduce losses for logging
            loss_reduced = reduce_loss_dict(loss_dict)
            reduced_scaled = {
                k: v * self.weight_dict[k]
                for k, v in loss_reduced.items()
                if k in self.weight_dict and k in self.verbose_loss_keys
            }
            total_scaled = sum(reduced_scaled.values())
            loss_value = total_scaled.item()

            metric_logger.update(loss=loss_value, **reduced_scaled)

            # Backward pass and optimization
            self.optimizer.zero_grad()
            losses.backward()
            if self.cfg.SOLVER.MAX_GRAD_NORM > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.SOLVER.MAX_GRAD_NORM
                )
            self.optimizer.step()

            # Scheduler and EMA
            adjust_learning_rate(self.cfg, self.optimizer, step, max_iter)
            if self.model_ema is not None:
                update_ema(self.model, self.model_ema, self.cfg.MODEL.EMA_DECAY)

            # Timing and metrics
            batch_time = time.time() - last_time
            last_time = time.time()
            metric_logger.update(time=batch_time, data=data_time)

            # ETA calculation
            eta_seconds = metric_logger.time.global_avg * (max_iter - step)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            # TensorBoard logging
            if self.writer is not None and is_main_process() and step % 50 == 0:
                try:
                    for k in reduced_scaled:
                        self.writer.add_scalar(f"{k}", metric_logger.meters[k].avg, step)
                except Exception as e:
                    self.logger.warning(f"Failed to write to TensorBoard: {e}")

            # Progress bar update
            if self.pbar is not None:
                self.pbar.update(1)
                postfix = (
                    f" loss={loss_value:.4f}, lr={self.optimizer.param_groups[0]['lr']:.6f}"
                )
                self.pbar.set_postfix_str(postfix)

            # Periodic logging
            if step % 50 == 0 or step == max_iter:
                log_msg = metric_logger.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter} / {max_iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "lr_vis_encoder: {lr_vis:.6f}",
                        "lr_text_encoder: {lr_text:.6f}",
                        "lr_temp_decoder: {lr_temp:.6f}",
                        "lr_class: {lr_clas:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=step,
                    max_iter=max_iter,
                    meters=str(metric_logger),
                    lr=self.optimizer.param_groups[0]["lr"],
                    lr_vis=self.optimizer.param_groups[1]["lr"],
                    lr_text=self.optimizer.param_groups[2]["lr"],
                    lr_temp=self.optimizer.param_groups[3]["lr"],
                    lr_clas=self.optimizer.param_groups[4]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
                if self.pbar is not None:
                    self.pbar.write(log_msg)
                else:
                    self.logger.info(log_msg)

            # Save checkpoints
            if step % self.cfg.SOLVER.CHECKPOINT_PERIOD == 0:
                self.checkpointer.save(f"model_{step:06d}", **self.arguments)

            if step == max_iter:
                self.checkpointer.save("model_final", **self.arguments)

            # Validation
            if self.cfg.SOLVER.TO_VAL and step % self.cfg.SOLVER.VAL_PERIOD == 0:
                self.validate()

        # Finalization
        self._close_progress()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=total_time))
        self.logger.info(
            "Total training time: {} ({:.4f} s / it)".format(
                total_time_str, total_time / max_iter
            )
        )
        if self.writer is not None:
            try:
                self.writer.close()
            except Exception as e:
                self.logger.warning(f"Failed to close TensorBoard writer: {e}")

    def validate(self):
        self.logger.info("Start validating")
        eval_model = self.model_ema if self.model_ema is not None else self.model
        evaluator = build_evaluator(self.cfg, self.logger, mode="test")
        postprocessor = build_postprocessors()
        torch.cuda.empty_cache()
        do_eval(
            self.cfg,
            mode="test",
            logger=self.logger,
            model=eval_model,
            postprocessor=postprocessor,
            data_loader=self.val_loader,
            evaluator=evaluator,
            device=self.device,
        )
        synchronize()

    def test(self):
        self.logger.info("Start Testing")
        eval_model = self.model_ema if self.model_ema is not None else self.model
        torch.cuda.empty_cache()
        evaluator = build_evaluator(self.cfg, self.logger, mode="test")
        postprocessor = build_postprocessors()
        test_loader = make_data_loader(
            self.cfg, mode="test", is_distributed=self.distributed
        )
        do_eval(
            self.cfg,
            mode="test",
            logger=self.logger,
            model=eval_model,
            postprocessor=postprocessor,
            data_loader=test_loader,
            evaluator=evaluator,
            device=torch.device(self.cfg.MODEL.DEVICE),
        )
        synchronize()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Spatio-Temporal Grounding Training")
    parser.add_argument(
        "--config-file",
        default="experiments/vidstg.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "--use-seed",
        dest="use_seed",
        help="use deterministic seed",
        default=True,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args()


def _init_distributed(local_rank: int) -> bool:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_dist = world_size > 1
    if is_dist:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()
    return is_dist


def main():
    args = _parse_args()
    num_gpus = int(os.environ.get("WORLD_SIZE", "1"))
    args.distributed = _init_distributed(args.local_rank)

    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    if args.use_seed:
        cudnn.benchmark = False
        cudnn.deterministic = True
        set_seed(args.seed + get_rank())

    synchronize()

    if cfg.OUTPUT_DIR:
        mkdir(cfg.OUTPUT_DIR)

    logger = setup_logger("Video Grounding", cfg.OUTPUT_DIR, get_rank())
    logger.info(f"Using {num_gpus} GPUs")
    logger.info(args)
    if args.config_file:
        logger.info(f"Loaded configuration file {args.config_file}")
    logger.info(f"Running with config:\n{cfg}")

    output_config = os.path.join(cfg.OUTPUT_DIR, "config.yml")
    logger.info(f"Saving config into: {output_config}")
    save_config(cfg, output_config)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    trainer = Trainer(cfg, args.local_rank, args.distributed, logger)
    trainer.setup()
    trainer.fit()
    if not args.skip_test:
        trainer.test()


if __name__ == "__main__":
    main()
