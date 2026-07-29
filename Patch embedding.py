#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
=========================================================
patch_embed.py
Patch Partition and Patch Embedding for MalVEIT
=========================================================

Author : Your Name
Model  : MalVEIT
"""

import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """
    Converts an RGB malware image into non-overlapping patch embeddings.

    Input:
        x : (B, C, H, W)

    Output:
        patches : (B, N, embed_dim)

    where
        B = Batch Size
        N = Number of patches
    """

    def __init__(
        self,
        img_size=224,
        patch_size=4,
        in_channels=3,
        embed_dim=128,
        norm_layer=nn.LayerNorm,
    ):

        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size

        self.grid_size = (
            img_size // patch_size,
            img_size // patch_size,
        )

        self.num_patches = (
            self.grid_size[0] *
            self.grid_size[1]
        )

        # Patch Partition + Linear Projection
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.norm = norm_layer(embed_dim)

    def forward(self, x):

        """
        x
        ----
        (B,C,H,W)

        returns

        (B,N,D)
        """

        B, C, H, W = x.shape

        if H != self.img_size or W != self.img_size:

            raise ValueError(
                f"Input image size ({H}x{W}) "
                f"does not match model "
                f"({self.img_size}x{self.img_size})"
            )

        # ----------------------------------------
        # Patch Partition
        # (B,D,H/P,W/P)
        # ----------------------------------------

        x = self.proj(x)

        # ----------------------------------------
        # Flatten
        # ----------------------------------------

        x = x.flatten(2)

        # ----------------------------------------
        # (B,N,D)
        # ----------------------------------------

        x = x.transpose(1, 2)

        # ----------------------------------------
        # Layer Normalization
        # ----------------------------------------

        x = self.norm(x)

        return x


if __name__ == "__main__":

    model = PatchEmbed(
        img_size=224,
        patch_size=4,
        in_channels=3,
        embed_dim=128,
    )

    x = torch.randn(2, 3, 224, 224)

    y = model(x)

    print("Input Shape :", x.shape)
    print("Output Shape:", y.shape)
    print("No. of patches:", model.num_patches)

