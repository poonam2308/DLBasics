import torch
import torch.nn as nn

from transformers.vision_transformer.patch_embedding import PatchEmbedding
from transformers.vision_transformer.transformer_block import TransformerBlock


class VisionTransformer(nn.Module):
    def __init__(self, patch_size, embed_dim, num_heads, num_layers, ff_dim, num_classes):
        super(VisionTransformer, self).__init__()
        self.patch_embedding = PatchEmbedding(patch_size, embed_dim)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)
        ])
        self.cls_token = nn.Parameter(torch.randn(1,1,embed_dim))
        self.positional_embedding = nn.Parameter(torch.randn((32//patch_size) ** 2 + 1, embed_dim))
        self.mlp_head = nn.Sequential(
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):

        N = x.shape[0]
        x = self.patch_embedding(x)
        cls_tokens = self.cls_token.expand(N, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.positional_embedding

        for block in self.transformer_blocks:
            x = block(x)

        cls_out = x[:,0]
        out = self.mlp_head(cls_out)
        return out
    

