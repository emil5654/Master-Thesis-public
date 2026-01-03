
"""
ResNet-18 Linear Probe
----------------------
ImageNet-pretrained backbone is frozen; only the final classification head
(and an optional input adapter for non-RGB inputs) is trainable.

Provides:
- ResNet18LinearProbe (nn.Module)
- make_resnet18(num_classes, in_channels, dropout_p=0.25, pretrained=True)
"""

import torch
import torch.nn as nn
import torchvision.models as tvm

class ResNet18FinetuneLast2(nn.Module):
    """
    ResNet-18 with conditional finetuning:

    - If pretrained=True:
        Freeze entire backbone, then unfreeze last residual *stage*
        (implemented as 4 residual blocks):
          - layer4[1]
          - layer4[0]

    - If pretrained=False:
        Train the entire model end-to-end (no freezing at all).

    In both cases the classification head and optional adapter are trainable.
    """

    def __init__(
        self,
        num_classes: int = 3,
        in_channels: int = 3,
        dropout_p: float = 0.0,
        pretrained: bool = True,
    ):
        super().__init__()

        # --- Load pretrained ResNet-18 backbone ---
        weights = tvm.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = tvm.resnet18(weights=weights)

        # --- Optional adapter for non-RGB inputs ---
        self.adapter = None
        if in_channels != 3:
            self.adapter = nn.Conv2d(in_channels, 3, kernel_size=1, bias=False)

        # --- Replace classification head ---
        in_features = self.backbone.fc.in_features
        head_layers = []
        if dropout_p and dropout_p > 0:
            head_layers.append(nn.Dropout(dropout_p))
        head_layers.append(nn.Linear(in_features, num_classes))
        self.backbone.fc = nn.Sequential(*head_layers)

        # ======================================================
        # FREEZE / UNFREEZE LOGIC
        # ======================================================

        if pretrained:
            # ------------------------------------------
            # Case 1: Pretrained → freeze everything first
            # ------------------------------------------
            for p in self.backbone.parameters():
                p.requires_grad = False

            # Unfreeze last 2 stages (4 blocks)
            last2_blocks = [
                self.backbone.layer4[1],
                self.backbone.layer4[0],
            ]

            for block in last2_blocks:
                for p in block.parameters():
                    p.requires_grad = True

            print("[INFO] Using pretrained backbone → finetuning last stage only.")

        else:
            # ------------------------------------------
            # Case 2: Not pretrained → train EVERYTHING
            # ------------------------------------------
            for p in self.backbone.parameters():
                p.requires_grad = True

            print("[INFO] Pretrained=False → training full backbone end-to-end.")

        # In both cases → head is always trainable
        for p in self.backbone.fc.parameters():
            p.requires_grad = True

        # Adapter, if present, should always be trainable
        if self.adapter is not None:
            for p in self.adapter.parameters():
                p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.adapter is not None:
            x = self.adapter(x)
        return self.backbone(x)


def make_resnet18_last2(
    num_classes: int,
    in_channels: int,
    dropout_p: float = 0.25,
    pretrained: bool = True,
) -> nn.Module:
    return ResNet18FinetuneLast2(
        num_classes=num_classes,
        in_channels=in_channels,
        dropout_p=dropout_p,
        pretrained=pretrained,
    )



def make_resnet18_last2(
    num_classes: int,
    in_channels: int,
    dropout_p: float = 0.25,
    pretrained: bool = True,
) -> nn.Module:
    """
    Factory function compatible with your create_model(...):

        model = make_resnet18_last2(
            num_classes=num_classes,
            in_channels=in_channels,
            dropout_p=dropout_p,
            pretrained=True,
        )

    Args:
        num_classes: number of output classes
        in_channels: input channels (3 for RGB; >3 uses a 1x1 adapter)
        dropout_p: dropout on the classification head (default 0.25)
        pretrained: load ImageNet weights for the backbone

    Returns:
        nn.Module (ResNet18FinetuneLast2)
    """
    return ResNet18FinetuneLast2(
        num_classes=num_classes,
        in_channels=in_channels,
        dropout_p=dropout_p,
        pretrained=pretrained,
    )
