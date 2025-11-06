import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.hub import load_state_dict_from_url
from vgqa.utils.distributed import is_main_process

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


class VSTGCheckpointer(object):
    """Checkpoint manager for saving and loading model/optimizer states."""

    def __init__(
        self,
        cfg: Any,
        model: torch.nn.Module,
        model_ema: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        save_dir: str = "",
        save_to_disk: Optional[bool] = None,
        logger: Optional[Any] = None,
        is_train: bool = True,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.model_ema = model_ema
        self.optimizer = optimizer
        self.save_dir = save_dir
        self.save_to_disk = save_to_disk
        self.logger = logger
        self.is_train = is_train

        if self.logger is None:
            # lazy import to avoid cycle
            from .log_setup import setup_logger
            self.logger = setup_logger("checkpointer", save_dir=save_dir, distributed_rank=0)

    def save(self, name: str, **kwargs: Any) -> None:
        """Save model checkpoint to disk."""
        if not self.save_dir or not self.save_to_disk:
            return

        # Prepare checkpoint data
        payload: Dict[str, Any] = {"model": self.model.state_dict()}
        if self.model_ema is not None:
            payload["model_ema"] = self.model_ema.state_dict()
        if self.optimizer is not None:
            payload["optimizer"] = self.optimizer.state_dict()
        payload.update(kwargs)

        # Ensure directory exists
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)

        save_path = Path(self.save_dir) / f"{name}.pth"
        self.logger.info("Saving checkpoint to {}".format(str(save_path)))
        torch.save(payload, str(save_path))

        self.tag_last_checkpoint(str(save_path))

    def load(self, f: Optional[str] = None, with_optim: bool = True, load_mapping: Dict[str, str] = {}):
        """Load model checkpoint or pretrained weight."""
        if self.has_checkpoint() and self.is_train:
            f = self.get_checkpoint_file()
        if not f:
            self.logger.info("No checkpoint found. Initializing model from ImageNet")
            return {}

        self.logger.info("Loading checkpoint from {}".format(f))
        checkpoint = self._load_file(f)
        self._load_model(checkpoint)

        if with_optim and ("optimizer" in checkpoint) and self.optimizer:
            self.logger.info("Loading optimizer from {}".format(f))
            self.optimizer.load_state_dict(checkpoint.pop("optimizer"))

        return checkpoint

    def has_checkpoint(self) -> bool:
        last = Path(self.save_dir) / "last_checkpoint"
        return last.exists()

    def get_checkpoint_file(self) -> str:
        last_file = Path(self.save_dir) / "last_checkpoint"
        try:
            return last_file.read_text().strip()
        except IOError:
            return ""

    def tag_last_checkpoint(self, last_filename: str) -> None:
        last_file = Path(self.save_dir) / "last_checkpoint"
        last_file.write_text(last_filename)
    
    def _load_file(self, f: str) -> Dict[str, Any]:
        # download url files
        if f.startswith("http"):
            self.logger.info("loading checking point from {}".format(f))
            return load_state_dict_from_url(model_urls[self.cfg.MODEL.RESNETS.NAME])
        # load native pytorch checkpoint
        return torch.load(f, map_location=torch.device("cpu"))
        
    def _load_mdetr_weight(self, weight_dict: Dict[str, Any]) -> None:
        load_mapping = {}
        current_keys = sorted(list(self.model.state_dict().keys()))

        for cur_key in current_keys:
            
            if cur_key.startswith('vis_encoder'):
                load_mapping[cur_key] = cur_key.replace('vis_encoder', 'backbone')
            
            if cur_key.startswith('text_encoder'):
                module_names = cur_key.split('.')
                if 'body' in module_names:
                    module_names.remove('body')
                else:
                    module_names.remove('text_encoder')
                    
                module_names.insert(0,'transformer')
                load_mapping[cur_key] = '.'.join(module_names)
                
            if  cur_key.startswith('input_proj'):
                load_mapping[cur_key] = cur_key
            
            if cur_key.startswith('bbox_embed'):
                load_mapping[cur_key] = cur_key
                    
            if cur_key.startswith('ground_encoder'):
                # ground_encoder.encoder.spatial_layers
                module_names = cur_key.split('.')
                if "spatial_layers" in module_names:
                    module_names.remove("ground_encoder")
                    module_names.insert(0,'transformer')
                    module_names.remove("spatial_layers")
                    module_names.insert(2,'layers')
                    load_mapping[cur_key] = '.'.join(module_names)

            if cur_key.startswith('ground_decoder'):
                module_names = cur_key.split('.')
                module_names.remove("ground_decoder")
                module_names.insert(0,'transformer')
                load_mapping[cur_key] = '.'.join(module_names)
                
        loaded_dict: Dict[str, Any] = {}
        for key in load_mapping:
            if load_mapping[key] in weight_dict.keys():
                loaded_dict[key] = weight_dict[load_mapping[key]]

        self.model.load_state_dict(loaded_dict, strict=False)

    def _load_pretrained(self, state_dict: Dict[str, Any]) -> None:
        model_key = 'model'
        if "model_ema" in state_dict:
            model_key = 'model_ema'
        
        if self.is_train:
            # Initialized with the pretrained model weight
            self._load_mdetr_weight(state_dict[model_key])
            if 'args' in state_dict.keys():
                state_dict.pop('args')
            if 'epoch' in state_dict.keys():
                state_dict.pop('epoch')
            if 'optimizer' in state_dict.keys():
                state_dict.pop('optimizer')
        else:
            # Used For Evaluation and Inference, Load trained Checkpoint
            # Filter out keys that don't exist in current model (e.g., position_ids from different transformers versions)
            filtered_state_dict = {k: v for k, v in state_dict[model_key].items() if k in self.model.state_dict()}
            self.model.load_state_dict(filtered_state_dict, strict=False)
        if (self.cfg.MODEL.EMA) and (self.model_ema is not None):
            self.model_ema.load_state_dict(deepcopy(self.model).state_dict()) 
   
    def _load_model(self, checkpoint: Dict[str, Any]) -> None:
        if self.is_train and self.has_checkpoint():   # resume training
            # self.model.load_state_dict(checkpoint["model"])
            state_dict = {k: v for k, v in checkpoint["model"].items() if k in self.model.state_dict()}
            self.model.load_state_dict(state_dict, strict=False)
            if (self.cfg.MODEL.EMA) and (self.model_ema is not None):
                if 'model_ema' not in checkpoint:
                    self.model_ema.load_state_dict(deepcopy(self.model).state_dict())
                else:
                    state_dict = {k: v for k, v in checkpoint["model_ema"].items() if k in self.model_ema.state_dict()}
                    self.model_ema.load_state_dict(state_dict, strict=False)
        else:
            self._load_pretrained(checkpoint)
        if 'model_ema' in checkpoint:
            checkpoint.pop('model_ema')
        checkpoint.pop('model')


