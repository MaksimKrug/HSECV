import argparse
from warnings import filterwarnings

import torch
from torch import optim
from tqdm import tqdm

from data import *
from model import *
from utils import *

filterwarnings("ignore")

# check device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


def train(model, dataloader_train, dataloader_val, lr, batchsize=8, epochs_num=10):
    # init criterion, optimizer and scheduler
    criterion = structure_loss
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=3, factor=0.5
    )
    best_val_score = 0
    # init running
    train_loss, train_precision, train_recall, train_dice = [], [], [], []
    val_loss, val_precision, val_recall, val_dice = [], [], [], []

    for epoch in tqdm(range(epochs_num)):
        """
        TRAIN PART
        """
        model.train()

        (
            running_loss_train,
            running_precision_train,
            running_recall_train,
            running_dice_train,
        ) = (
            AvgMeter(),
            AvgMeter(),
            AvgMeter(),
            AvgMeter(),
        )
        (
            running_loss_val,
            running_precision_val,
            running_recall_val,
            running_dice_val,
        ) = (
            AvgMeter(),
            AvgMeter(),
            AvgMeter(),
            AvgMeter(),
        )

        for _, (x, y) in enumerate(dataloader_train):
            # get pred
            x = x.to(device)
            y = y.to(device)
            yhat = model(x)
            # loss and metrics
            loss = criterion(yhat, y)
            acc, recall, precision, mIoU, mDice = metrics(yhat, y)
            # step
            optimizer.zero_grad()
            loss.backward()
            clip_gradient(optimizer)
            optimizer.step()
            # update metrics
            running_loss_train.update(loss.data, batchsize)
            running_precision_train.update(torch.Tensor([precision]), batchsize)
            running_recall_train.update(torch.Tensor([recall]), batchsize)
            running_dice_train.update(torch.Tensor([mDice]), batchsize)

        train_loss.append(running_loss_train.show().item())
        train_precision.append(running_precision_train.show().item())
        train_recall.append(running_recall_train.show().item())
        train_dice.append(running_dice_train.show().item())

        """
        VALIDATION PART
        """

        model.eval()
        with torch.no_grad():
            for _, (x, y) in enumerate(dataloader_val):
                # get pred
                x = x.to(device)
                y = y.to(device)
                yhat = model(x)
                # loss and metrics
                loss = criterion(yhat, y)
                acc, recall, precision, mIoU, mDice = metrics(yhat, y)
                # update metrics
                running_loss_val.update(loss.data, batchsize)
                running_precision_val.update(torch.Tensor([precision]), batchsize)
                running_recall_val.update(torch.Tensor([recall]), batchsize)
                running_dice_val.update(torch.Tensor([mDice]), batchsize)

        val_loss.append(running_loss_val.show().item())
        val_precision.append(running_precision_val.show().item())
        val_recall.append(running_recall_val.show().item())
        val_dice.append(running_dice_val.show().item())

        scheduler.step(running_dice_val.show().item())  # scheduler step
        # save model if needed
        if running_dice_val.show().item() > best_val_score:
            torch.save(model.state_dict(), "HardNetMSEG.pth")
            best_val_score = running_dice_val.show().item()

        if epoch % 20 == 0:
            print("-" * 15)
            print(f"Epoch: {epoch}")
            print(
                f"TRAIN: Loss: {round(running_loss_train.show().item(), 3)}, Precision: {round(running_precision_train.show().item(), 3)}, Recall: {round(running_recall_train.show().item(), 3)}, mDice: {round(running_dice_train.show().item(), 3)}"
            )
            print(
                f"VAL: Loss: {round(running_loss_val.show().item(), 3)}, Precision: {round(running_precision_val.show().item(), 3)}, Recall: {round(running_recall_val.show().item(), 3)}, mDice: {round(running_dice_val.show().item(), 3)}"
            )
            print("-" * 15)

    return (
        train_loss,
        train_precision,
        train_recall,
        train_dice,
        val_loss,
        val_precision,
        val_recall,
        val_dice,
    )


def get_args():
    parser = argparse.ArgumentParser(
        description="Train the UNet model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--imgs-dir",
        dest="imgs_dir",
        type=str,
        default="Kvasir-SEG/images/*.jpg",
        help="Images dir",
    )
    parser.add_argument(
        "--masks-dir",
        dest="masks_dir",
        type=str,
        default="Kvasir-SEG/masks/*.jpg",
        help="Masks dir",
    )
    parser.add_argument(
        "--epochs-num", type=int, default=1, help="Number of epochs", dest="epochsnum",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size", dest="batchsize",
    )
    parser.add_argument(
        "--resize", dest="resize", type=int, default=100, help="Image resize size???",
    )
    parser.add_argument(
        "--learning-rate",
        metavar="LR",
        type=float,
        default=0.0001,
        help="Learning rate",
        dest="lr",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # # get dataloaders
    # dataloader_train, dataloader_test, dataloader_val = get_data(
    #     args.imgs_dir, args.masks_dir, resize=args.resize, batch_size=args.batchsize
    # )

    # # init model
    # model = HarDMSEG()
    # model = model.to(device)

    # # train
    # train_loss, train_accuracy, train_f1, val_loss, val_accuracy, val_f1 = train(
    #     model,
    #     dataloader_train=dataloader_train,
    #     dataloader_val=dataloader_val,
    #     batchsize=args.batchsize,
    #     epochs_num=args.epochs_num,
    # )

