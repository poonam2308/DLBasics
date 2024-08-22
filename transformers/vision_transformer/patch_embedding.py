from unittest.mock import patch

import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, embed_dim, in_channels=3):
        super(PatchEmbedding, self).__init__()
        self.patch_size =patch_size
        self.embed_dim = embed_dim
        self.in_channels = in_channels

        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.projection(x)  # (N, E, H', W')
        x = x.flatten(2)  # (N, E, H'*W')
        x = x.transpose(1,2) # (N, H'*W', E)
        return x
        