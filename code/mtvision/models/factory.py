
from .resnet18 import make_resnet18_last2

from .vit_small import ViTSmall_last4

def create_model(model_name: str, num_classes: int, in_channels: int, dropout_p: float = 0.25):
    model_name = model_name.lower()

    if model_name in ("cnn", "simplecnn"):
        from .cnn import SimpleCNN
        return SimpleCNN(in_ch=in_channels, num_classes=num_classes, p=dropout_p)

    elif model_name == "resnet18_last2":
        return make_resnet18_last2(
            num_classes=num_classes,
            in_channels=in_channels,
            dropout_p=dropout_p,
            pretrained=True,
        )
    elif model_name == "resnet18_whole_backbone":
        return make_resnet18_last2(
            num_classes=num_classes,
            in_channels=in_channels,
            dropout_p=dropout_p,
            pretrained=False,
        )
    elif model_name == "vitsmall_last4":
        return ViTSmall_last4(
            num_classes=num_classes,
            in_channels=in_channels,
            dropout_p=dropout_p,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
