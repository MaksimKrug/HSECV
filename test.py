from warnings import filterwarnings

import torch
from tqdm import tqdm

from model import *
from utils import *

filterwarnings("ignore")


def get_test_score(
    dataloader_test: torch.utils.data.DataLoader,
    model_path: str = "baseline_model.pth",
    batchsize: int = 1,
    model_type: str = 'UNet',
):
    # check device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # init model
    if model_type == 'HarDMSEG':
        model = HarDMSEG()
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    # init loss and metrics lists
    criterion = structure_loss
    (
        running_loss_test,
        running_precision_test,
        running_recall_test,
        running_dice_test,
    ) = (
        AvgMeter(),
        AvgMeter(),
        AvgMeter(),
        AvgMeter(),
    )
    # eval
    with torch.no_grad():
        for batch_num, (x, y) in tqdm(enumerate(dataloader_test)):
            # get pred
            x = x.to(device)
            y = y.to(device)
            yhat = model(x)
            # loss and metrics
            loss = criterion(yhat, y)
            acc, recall, precision, mIoU, mDice = metrics(yhat, y)
            # update metrics
            running_loss_test.update(loss.data, batchsize)
            running_precision_test.update(torch.Tensor([precision]), batchsize)
            running_recall_test.update(torch.Tensor([recall]), batchsize)
            running_dice_test.update(torch.Tensor([mDice]), batchsize)

    print(
        f"TEST: Loss: {round(running_loss_test.show().item(), 3)}, Precision: {round(running_precision_test.show().item(), 3)}, Recall: {round(running_recall_test.show().item(), 3)}, mDice: {round(running_dice_test.show().item(), 3)}"
    )

