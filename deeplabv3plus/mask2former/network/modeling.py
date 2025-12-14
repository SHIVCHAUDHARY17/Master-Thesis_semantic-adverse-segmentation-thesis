# network/modeling.py

from .mask2former import Mask2FormerWithWASTAS

# from .backbone import build_backbone  # Optional: if you modularize backbones later


def _build_mask2former(num_classes=10, **kwargs):
    """
    Build the full Mask2Former model with auxiliary heads (weather/time).
    Args:
        num_classes (int): Number of segmentation classes.
        kwargs: Additional optional arguments for flexibility.
    Returns:
        nn.Module: Mask2FormerWithAuxHead instance
    """
    model = Mask2FormerWithWASTAS(num_classes=num_classes)
    return model


def _load_model(arch_type, backbone, num_classes, output_stride, pretrained_backbone, **kwargs):
    """
    Master loader function for all supported architectures (e.g., DeepLabV3+, Mask2Former).
    Mirrors the DeepLab logic, keeping room for extensibility.

    Args:
        arch_type (str): Architecture name, e.g., 'mask2former'
        backbone (str): Backbone name, if applicable
        num_classes (int): Number of classes for segmentation head
        output_stride (int): For consistency (not used in M2F but kept for compatibility)
        pretrained_backbone (bool): Whether to load a pretrained backbone
        kwargs: Other args as needed

    Returns:
        nn.Module: A model instance (e.g., Mask2FormerWithAuxHead)
    """
    if arch_type == 'mask2former':
        return _build_mask2former(num_classes=num_classes, **kwargs)
    else:
        raise NotImplementedError(f"Architecture {arch_type} not supported.")


def mask2former(num_classes=10):
    """
    Public callable to build a Mask2Former model (like deeplabv3plus_mobilenet).

    Args:
        num_classes (int): Number of classes.

    Returns:
        nn.Module: A ready-to-train Mask2Former model.
    """
    return _load_model('mask2former', None, num_classes, None, None)

