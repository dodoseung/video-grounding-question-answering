from .position_encoding import build_position_encoding
from .backbone import GroupNormBackbone, Backbone, Joiner


def build_vis_encoder(cfg):
    """Build vision encoder with ResNet backbone and position encoding"""
    # Build position encoding module
    position_embedding = build_position_encoding(cfg)

    # Determine if backbone should be trainable
    train_backbone = cfg.SOLVER.VIS_BACKBONE_LR  > 0
    backbone_name = cfg.MODEL.VISION_BACKBONE.NAME

    # Build backbone with GroupNorm or BatchNorm
    if backbone_name in ("resnet50-gn", "resnet101-gn"):
        backbone = GroupNormBackbone(
            backbone_name,
            train_backbone,
            False,
            cfg.MODEL.VISION_BACKBONE.DILATION
        )
    else:
        backbone = Backbone(
            backbone_name,
            train_backbone,
            False,
            cfg.MODEL.VISION_BACKBONE.DILATION
        )

    # Combine backbone with position encoding
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model