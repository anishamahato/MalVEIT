import torch
import torch.nn as nn


class MaskTokenInsertion(nn.Module):
    """
    Inserts a learnable mask token at masked patch positions
    and restores the original patch ordering.

    Input:
        x_masked   : (B, N_visible, D)
        ids_restore: (B, N)

    Output:
        x_full     : (B, N, D)
    """

    def __init__(self, embed_dim=128):

        super().__init__()

        # Learnable mask token
        self.mask_token = nn.Parameter(
            torch.zeros(1, 1, embed_dim)
        )

        # Initialize mask token
        nn.init.normal_(
            self.mask_token,
            mean=0.0,
            std=0.02
        )

    def forward(self, x_masked, ids_restore):

        B, N_visible, D = x_masked.shape

        # Total number of patches
        N = ids_restore.shape[1]

        # Number of masked patches
        N_masked = N - N_visible

        # Create mask tokens
        mask_tokens = self.mask_token.repeat(
            B,
            N_masked,
            1
        )

        # Combine visible patches and mask tokens
        x_ = torch.cat(
            [x_masked, mask_tokens],
            dim=1
        )

        # Restore original patch ordering
        x_full = torch.gather(
            x_,
            dim=1,
            index=ids_restore.unsqueeze(-1).repeat(
                1, 1, D
            )
        )

        return x_full
