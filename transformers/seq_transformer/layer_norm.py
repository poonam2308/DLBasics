import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.ln = nn.LayerNorm(d_model, eps=eps)

    def forward(self, x):
        return self.ln(x)
