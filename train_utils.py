import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch import optim
import numpy as np


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
    acc = accuracy_score(pred_np, target_np)
    f1 = f1_score(pred_np, target_np)

    return acc, f1


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


"""
def dice_loss(pred, target, smooth=1e-5):
    intersection = (pred * target).sum(dim=(2,3))
    union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3))
    dice = 2.0 * (intersection + smooth) / (union + smooth)
    loss = 1 - dice
    
    return loss.sum(), dice.sum()


"""
