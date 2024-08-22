import torch.nn as nn

from transformers.vision_transformer.multihead_selfattention import MultiHeadSelfAttention


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_out = self.attention(x)
        x = self.layernorm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.layernorm2(x + ffn_out)
        return x