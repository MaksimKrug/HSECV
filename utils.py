import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def clip_gradient(optimizer, grad_clip=0.5):
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def structure_loss(pred, mask):

    weit = 1 + 5 * torch.abs(
        F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask
    )
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce="none")
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


def metrics(pred, target):
    pred = torch.sigmoid(pred)
    pred = pred >= 0.5
    pred_np = pred.reshape(1, -1).data.to("cpu").numpy()[0].astype(int)
    target_np = target.reshape(1, -1).data.to("cpu").numpy()[0].astype(int)
    # calculate statistics
    tp = len(np.where((pred_np == target_np) & (pred_np == 1))[0])
    fp = len(np.where((pred_np == 1) & (pred_np != target_np))[0])
    tn = len(np.where((pred_np == target_np) & (pred_np == 0))[0])
    fn = len(np.where((pred_np == 0) & (pred_np != target_np))[0])
    # calculate scores
    acc = (tp + tn) / (tp + fp + tn + fn)
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    mIoU = tp / (tp + fp + fn)
    mDice = 2 * tp / (2 * tp + fp + fn)

    return acc, recall, precision, mIoU, mDice


class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        return torch.mean(
            torch.stack(self.losses[np.maximum(len(self.losses) - self.num, 0) :])
        )


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
        torch.nn.init.xavier_uniform_(self.up.weight)  # init weights
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
        torch.nn.init.xavier_uniform_(self.conv.weight)  # init weights

    def forward(self, x):
        return self.conv(x)
