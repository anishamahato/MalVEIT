#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn


# In[2]:


class RandomMasking(nn.Module):
    """
    Random Patch Masking Module for MalVEIT

    Input:
        x : (B, N, D)

    Output:
        x_masked    : Visible patches
        mask        : Binary mask
        ids_restore : Restore indices
    """

    def __init__(self, mask_ratio=0.60):

        super().__init__()

        self.mask_ratio = mask_ratio

    def forward(self, x):

        B, N, D = x.shape

        # Number of visible patches
        len_keep = int(N * (1 - self.mask_ratio))

        # Generate random noise
        noise = torch.rand(B, N, device=x.device)

        # Shuffle patch indices
        ids_shuffle = torch.argsort(noise, dim=1)

        # Restore indices
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Keep first visible patches
        ids_keep = ids_shuffle[:, :len_keep]

        # Gather visible patches
        x_masked = torch.gather(
            x,
            dim=1,
            index=ids_keep.unsqueeze(-1).repeat(1,1,D)
        )

        # Binary mask
        mask = torch.ones(B, N, device=x.device)

        mask[:, :len_keep] = 0

        mask = torch.gather(mask, 1, ids_restore)

        return x_masked, mask, ids_restore


# In[ ]:


# Patch Embedding

patch_embed = PatchEmbed(
    img_size=224,
    patch_size=4,
    embed_dim=128
)

image = torch.randn(2,3,224,224)

patches = patch_embed(image)

print("Patch Embeddings:", patches.shape)


# In[ ]:


mask_layer = RandomMasking(mask_ratio=0.60)

visible_patches, mask, ids_restore = mask_layer(patches)

print("Original Patches :", patches.shape)
print("Visible Patches  :", visible_patches.shape)
print("Mask Shape       :", mask.shape)
print("Restore Shape    :", ids_restore.shape)


# In[ ]:


import matplotlib.pyplot as plt

plt.figure(figsize=(12,2))

plt.imshow(mask[0].reshape(56,56), cmap="gray")

plt.title("Random Mask (White = Masked)")
plt.axis("off")

plt.show()


# In[ ]:





# In[ ]:




