import torch
import torch.nn.functional as F
from torch import nn
import random


def init_weights(m, seed=42):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        random.seed(seed)
        torch.manual_seed(seed)
        torch.nn.init.xavier_uniform_(m.weight)

class DoubleConv(nn.Module):
    """ 
    Apply two consecuitive conv layers
    """

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        # init mid channels if needed
        if mid_channels is None:
            mid_channels = out_channels
        # sequential layer
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.double_conv.apply(init_weights)

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """
    Downscale: maxpool -> DubleConv
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_doubleconv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_doubleconv(x)


class Up(nn.Module):
    """
    Upscale: transposeconv -> torch.cat -> doubleconv
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        torch.nn.init.xavier_uniform_(self.up.weight) # init weights
        self.doubleconv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # apply transpose conv
        x1 = self.up(x1)
        # pad
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.doubleconv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        torch.nn.init.xavier_uniform_(self.conv.weight) # init weights

    def forward(self, x):
        return self.conv(x)
