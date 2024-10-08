import torch.nn as nn

from transformers.seq_transformer.feed_forward import FeedForward
from transformers.seq_transformer.layer_norm import LayerNorm
from transformers.seq_transformer.multihead_attention import MultiHeadAttention



class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super(DecoderLayer, self).__init__()
        self.attention1 = MultiHeadAttention(d_model, n_heads)
        self.attention2 = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.layer_norm1 = LayerNorm(d_model)
        self.layer_norm2 = LayerNorm(d_model)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        x = self.attention1(x, mask=tgt_mask)
        x = self.layer_norm1(x + self.attention2(x, mask=src_mask))
        x = self.layer_norm2(x + self.feed_forward(x))
        return x
