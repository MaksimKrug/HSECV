import torch
from tqdm import tqdm

from model import *
from train_utils import *


def get_test_score(
    dataloader_test: torch.utils.data.DataLoader, model_path: str = "baseline_model.pth"
):
    # check device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # init model
    model = UNet()
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    # init loss and metrics lists
    criterion = structure_loss
    running_loss_test, running_accuracy_test, running_f1_test = (
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
            acc, f1 = metrics(yhat, y)
            # update metrics
            running_loss_test.update(loss.data, 8)
            running_accuracy_test.update(torch.Tensor([acc]), 8)
            running_f1_test.update(torch.Tensor([f1]), 8)

    res_loss = round(running_loss_test.show().item(), 3)
    res_accuracy = round(running_accuracy_test.show().item(), 3)
    res_f1 = round(running_f1_test.show().item(), 3)

    print(f"TEST: Loss: {res_loss}, Accuracy: {res_accuracy}, F1: {res_f1}")
