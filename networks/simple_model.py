import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 2)  # 10 input features, 2 output classes (binary classification)

    def forward(self, x):
        return self.fc(x)