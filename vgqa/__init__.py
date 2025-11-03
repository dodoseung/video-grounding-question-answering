"""VGQA: Video Grounding and Question Answering

A unified framework for spatio-temporal video grounding and video question answering.
"""

__version__ = "1.0.0"
__author__ = "VGQA Team"

from . import core
from . import data
from . import inference
from . import training
from . import utils
from . import config

__all__ = [
    'core',
    'data',
    'inference',
    'training',
    'utils',
    'config',
]
