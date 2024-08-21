import torch.nn as nn

from transformers.seq_transformer.decoder import DecoderLayer
from transformers.seq_transformer.encoder import EncoderLayer
from transformers.seq_transformer.positional_encoding import PositionalEncoding


class Transformer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, n_layers, vocab_size, max_len, dropout=0.1):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src = self.dropout(self.positional_encoding(self.embedding(src)))
        tgt = self.dropout(self.positional_encoding(self.embedding(tgt)))

        for layer in self.encoder_layers:
            src = layer(src)

        for layer in self.decoder_layers:
            tgt = layer(tgt, src, src_mask, tgt_mask)

        output = self.fc_out(tgt)
        return output
