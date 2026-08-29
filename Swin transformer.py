!pip install timm

import torch
import torch.nn as nn
import timm

swin = timm.create_model(
    'swin_tiny_patch4_window7_224',
    pretrained=True
)

print(swin)


class MalVEITSwinEncoder(nn.Module):

    def __init__(self, input_dim=128):

        super().__init__()

        # Load pretrained Swin-Tiny
        self.swin = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=True
        )

        # Your PatchEmbed dimension = 128
        # Swin-Tiny dimension = 96
        self.input_projection = nn.Linear(
            input_dim,
            96
        )

        # Remove Swin classification head
        self.swin.head = nn.Identity()

    def forward(self, x):

        """
        Input:
            x : (B, N, 128)

        Output:
            features : Swin feature representation
        """

        B, N, D = x.shape

        # ------------------------------------------------
        # Project MalVEIT embeddings: 128 → 96
        # ------------------------------------------------

        x = self.input_projection(x)

        # ------------------------------------------------
        # Convert token sequence to spatial feature map
        # ------------------------------------------------
        # N = 3136 = 56 × 56

        x = x.reshape(
            B,
            56,
            56,
            96
        )

        # ------------------------------------------------
        # Swin Transformer stages
        # ------------------------------------------------

        x = self.swin.layers[0](x)

        x = self.swin.layers[1](x)

        x = self.swin.layers[2](x)

        x = self.swin.layers[3](x)

        # ------------------------------------------------
        # Final normalization
        # ------------------------------------------------

        x = self.swin.norm(x)

        return x
