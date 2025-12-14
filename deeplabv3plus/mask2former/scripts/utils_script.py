def freeze_encoder(model):
    """
    Freeze encoder and pixel decoder parameters of Mask2Former
    """
    for name, param in model.named_parameters():
        if "encoder" in name or "pixel_decoder" in name:
            param.requires_grad = False

def unfreeze_encoder(model):
    """
    Unfreeze encoder and pixel decoder parameters of Mask2Former
    """
    for name, param in model.named_parameters():
        if "encoder" in name or "pixel_decoder" in name:
            param.requires_grad = True

