from .modal_encoder import CrossModalEncoder
from .query_decoder import QueryDecoder
from .classifier import TemporalSampling, SpatialActivation


def build_encoder(cfg):
    """Build cross-modal encoder for video-text fusion"""
    return CrossModalEncoder(cfg)

def build_decoder(cfg):
    """Build query decoder for spatio-temporal prediction"""
    return QueryDecoder(cfg)

def build_TemporalSampling(width):
    """Build temporal frame sampling classifier"""
    return TemporalSampling(width)

def build_SpatialActivation(width, vobsize):
    """Build spatial activation classifier"""
    return SpatialActivation(width, vobsize)