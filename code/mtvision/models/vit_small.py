import torch
import torch.nn as nn
from transformers import ViTConfig, ViTModel


class ViTSmall_last4(nn.Module):
    """
    ViT-S/16 fine-tuning the last 4 transformer blocks:
      - pretrained backbone (WinKawaks/vit-small-patch16-224)
      - all backbone frozen except last 4 encoder blocks + final LayerNorm
      - classifier head always trainable
    """

    def __init__(self, num_classes: int, in_channels: int, dropout_p: float = 0.0):
        super().__init__()

        if in_channels != 3:
            raise ValueError(
                f"ViTSmall supports in_channels=3 only, got {in_channels}"
            )

        # Load config
        config = ViTConfig.from_pretrained("WinKawaks/vit-small-patch16-224")
        config.num_channels = 3

        # Load backbone
        self.vit = ViTModel.from_pretrained(
            "WinKawaks/vit-small-patch16-224",
            config=config,
            add_pooling_layer=False,
        )

        embed_dim = config.hidden_size  # 384 for ViT-S
        total_layers = len(self.vit.encoder.layer)
        n_unfrozen_layers = 4  # fine-tune the last 4 transformer blocks

        # 1) Freeze everything
        for p in self.vit.parameters():
            p.requires_grad = False

        # 2) Unfreeze last 4 transformer blocks
        if n_unfrozen_layers > total_layers:
            raise ValueError(
                f"Requested {n_unfrozen_layers} layers, but ViT-S has {total_layers}"
            )

        for block in self.vit.encoder.layer[-n_unfrozen_layers:]:
            for p in block.parameters():
                p.requires_grad = True

        # Also unfreeze final LayerNorm
        for p in self.vit.layernorm.parameters():
            p.requires_grad = True

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.vit(x).last_hidden_state
        cls = out[:, 0]
        return self.classifier(cls)
