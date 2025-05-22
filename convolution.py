import torch
import torch.nn as nn


class DoubleConv2D(nn.Module):
    def __init__(self, d, hidden_channels=16):
        super().__init__()
        self.conv1 = nn.Conv2d(6 * d, hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, threeD_embedding):
        x = threeD_embedding.permute(0, 3, 1, 2)
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        x = x.permute(0, 2, 3, 1)
        return x
