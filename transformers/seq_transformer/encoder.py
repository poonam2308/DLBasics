import torch.nn as nn

from transformers.seq_transformer.feed_forward import FeedForward
from transformers.seq_transformer.layer_norm import LayerNorm
from transformers.seq_transformer.multihead_attention import MultiHeadAttention


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.layer_norm1 = LayerNorm(d_model)
        self.layer_norm2 = LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = self.attention(x, mask=mask)
        x = self.layer_norm1(x + self.feed_forward(x))
        return x
