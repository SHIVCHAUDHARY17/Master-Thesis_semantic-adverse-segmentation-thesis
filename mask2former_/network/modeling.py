# network/modeling.py
import torch.nn as nn
from .mask2former import Mask2FormerWithWASTAS

#DEFAULT_PRETRAINED_CARD = "facebook/mask2former-swin-base-IN21k-cityscapes-semantic"
DEFAULT_PRETRAINED_CARD = "facebook/mask2former-swin-large-cityscapes-semantic"

def _build_mask2former(
    num_classes: int = 19,  # Default to 19 for Cityscapes
    *,
    pretrained_card: str = DEFAULT_PRETRAINED_CARD,
    was_classes: int = 4,
    tas_classes: int = 2,
    output_hidden_states: bool = True,
    output_auxiliary_logits: bool = True,
    **kwargs,
) -> nn.Module:
    return Mask2FormerWithWASTAS(
        num_classes=num_classes,
        pretrained_card=pretrained_card,
        was_classes=was_classes,
        tas_classes=tas_classes,
        output_hidden_states=output_hidden_states,
        output_auxiliary_logits=output_auxiliary_logits,
        **kwargs,
    )

def _load_model(
    arch_type: str,
    backbone: str = None,
    num_classes: int = 19,  # Default to 19 for Cityscapes
    output_stride: int = 16,  # kept for parity; unused by Mask2Former
    pretrained_backbone: bool = True,
    **kwargs,
) -> nn.Module:
    if arch_type == 'mask2former':
        return _build_mask2former(num_classes=num_classes, **kwargs)
    raise NotImplementedError(f"Architecture {arch_type} not supported.")

def mask2former(
    num_classes: int = 19,  # Default to 19 for Cityscapes
    *,
    pretrained_card: str = DEFAULT_PRETRAINED_CARD,
    was_classes: int = 4,
    tas_classes: int = 2,
    **kwargs,
) -> nn.Module:
    return _load_model(
        'mask2former',
        None,
        num_classes,
        output_stride=kwargs.pop("output_stride", 16),
        pretrained_backbone=True,
        pretrained_card=pretrained_card,
        was_classes=was_classes,
        tas_classes=tas_classes,
        **kwargs,
    )
