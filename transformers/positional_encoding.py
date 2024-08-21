import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        positions = torch.arange (0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model,2).float() * - (math.log(10000.0)/ d_model))
        self.encoding[:,0::2] = torch.sin(positions * div_term)
        self.encoding[:,1::2] = torch.cos(positions * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        device = x.device
        encoding = self.encoding.to(device)
        return x + encoding[:, :x.size(1)]
