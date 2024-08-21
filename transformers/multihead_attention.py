import torch
import torch.nn as nn

from transformers.layer_norm import LayerNorm
from transformers.self_attention import SelfAttention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.attention = SelfAttention(d_model, n_heads)
        self.layer_norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask=None):
        attn_output = self.attention(q=x, k=x, v=x, mask=mask)
        return self.layer_norm(x + self.dropout(attn_output))
