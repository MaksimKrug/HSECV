import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score


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
    tp = len(np.where((pred_np == target_np) & (pred_np==1))[0])
    fp = len(np.where((pred_np == 1) & (pred_np!=target_np))[0])
    tn = len(np.where((pred_np == target_np) & (pred_np==0))[0])
    fn = len(np.where((pred_np == 0) & (pred_np!=target_np))[0])
    # calculate scores
    acc = (tp + tn) / (tp + fp + tn + fn)
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    mIoU = tp / (tp + fp + fn)
    mDice = 2*tp / (2*tp + fp + fn)

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
