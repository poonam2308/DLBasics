import torch
import torch.nn as nn
import numpy as np

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "Embedding dimension must be divisible by number of heads"

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        N, L, E = x.shape

        qkv = self.qkv(x)  # (N, L, 3*E)
        qkv = qkv.reshape(N, L, 3, self.num_heads, self.head_dim) #(N, L, 3, num_heds, head_dim)
        qkv = qkv.permute(2,0,3,1,4) # (3, N, num_heads, L, head_dim)

        q, k , v = qkv[0], qkv[1], qkv[2]
        attn_weights = torch.matmul(q, k.transpose(-2,-1)) # (N, num_heads, L, L)
        attn_weights = attn_weights / np.sqrt(self.head_dim)
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)

        out = torch.matmul(attn_weights, v) # (N, num_heads, L, head_dim)
        out = out.transpose(1,2).reshape(N,L,E) # (N, L, E)
        out = self.out(out)

        return out



