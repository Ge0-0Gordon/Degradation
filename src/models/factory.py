from __future__ import annotations
from typing import Dict
from torchvision.models.segmentation import deeplabv3_resnet50
from src.models.unet import UNet

def build_model(cfg: Dict):
    mcfg = cfg["model"]
    name = str(mcfg["name"]).lower()
    num_classes = int(mcfg.get("num_classes", 21))

    if name == "deeplabv3_resnet50":
        pretrained = bool(mcfg.get("pretrained", True))
        aux_loss = bool(mcfg.get("aux_loss", True))
        try:
            if pretrained:
                return deeplabv3_resnet50(weights="DEFAULT", num_classes=num_classes, aux_loss=aux_loss)
            return deeplabv3_resnet50(weights=None, weights_backbone=None, num_classes=num_classes, aux_loss=aux_loss)
        except TypeError:
            # 兼容旧版 torchvision
            return deeplabv3_resnet50(pretrained=pretrained, num_classes=num_classes, aux_loss=aux_loss)

    if name == "unet":
        return UNet(in_channels=3, num_classes=num_classes, base_ch=int(mcfg.get("base_ch", 64)))

    raise ValueError(f"Unsupported model: {name}")
