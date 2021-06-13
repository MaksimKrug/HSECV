import argparse
from warnings import filterwarnings

import torch
from torch import optim
from tqdm import tqdm

from data import *
from model import *
from train_utils import *

filterwarnings("ignore")

# check device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


def train(model, dataloader_train, dataloader_val, epochs_num=10):
    # init criterion, optimizer and scheduler
    criterion = structure_loss
    optimizer = optim.Adam(model.parameters(), lr=1e-7, weight_decay=1e-8)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=3, factor=0.5
    )
    best_val_score = 0
    # init running
    train_loss, train_accuracy, train_f1 = [], [], []
    val_loss, val_accuracy, val_f1 = [], [], []

    for epoch in tqdm(range(epochs_num)):
        """
        TRAIN PART
        """
        model.train()

        running_loss_train, running_accuracy_train, running_f1_train = (
            AvgMeter(),
            AvgMeter(),
            AvgMeter(),
        )
        running_loss_val, running_accuracy_val, running_f1_val = (
            AvgMeter(),
            AvgMeter(),
            AvgMeter(),
        )

        for batch_num, (x, y) in enumerate(dataloader_train)):
            # get pred
            x = x.to(device)
            y = y.to(device)
            yhat = model(x)
            # loss and metrics
            loss = criterion(yhat, y)
            acc, f1 = metrics(yhat, y)
            # step
            optimizer.zero_grad()
            loss.backward()
            # clip_gradient(optimizer)
            optimizer.step()
            # update metrics
            running_loss_train.update(loss.data, 8)
            running_accuracy_train.update(torch.Tensor([acc]), 8)
            running_f1_train.update(torch.Tensor([f1]), 8)

        train_loss.append(running_loss_train.show().item())
        train_accuracy.append(running_accuracy_train.show().item())
        train_f1.append(running_f1_train.show().item())

        """
        VALIDATION PART
        """

        model.eval()
        with torch.no_grad():
            for batch_num, (x, y) in enumerate(dataloader_val):
                # get pred
                x = x.to(device)
                y = y.to(device)
                yhat = model(x)
                # loss and metrics
                loss = criterion(yhat, y)
                acc, f1 = metrics(yhat, y)
                # update metrics
                running_loss_val.update(loss.data, 8)
                running_accuracy_val.update(torch.Tensor([acc]), 8)
                running_f1_val.update(torch.Tensor([f1]), 8)

        val_loss.append(running_loss_val.show().item())
        val_accuracy.append(running_accuracy_val.show().item())
        val_f1.append(running_f1_val.show().item())

        scheduler.step(running_f1_val.show().item())  # scheduler step
        # save model if needed
        if running_f1_val.show().item() > best_val_score:
            torch.save(model.state_dict(), "UNet.pth")
            best_val_score = running_f1_val.show().item()

        if epoch % 20 == 0 and epoch != 0:
            print("-" * 15)
            print(f"Epoch: {epoch}")
            print(
                f"TRAIN: Loss: {round(running_loss_train.show().item(), 3)}, Accuracy: {round(running_accuracy_train.show().item(), 3)}, F1: {round(running_f1_train.show().item(), 3)}"
            )
            print(
                f"VAL: Loss: {round(running_loss_val.show().item(), 3)}, Accuracy: {round(running_accuracy_val.show().item(), 3)}, F1: {round(running_f1_val.show().item(), 3)}"
            )
            print("-" * 15)

    return train_loss, train_accuracy, train_f1, val_loss, val_accuracy, val_f1


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
        "--epochs-num", type=int, default=1, help="Number of epochs", dest="epochs_num",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size", dest="batch_size",
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

    # get dataloaders
    dataloader_train, dataloader_test, dataloader_val = get_data(
        args.imgs_dir, args.masks_dir, resize=args.resize, batch_size=args.batch_size
    )

    # init model
    model = UNet()
    model = model.to(device)

    # train
    train_loss, train_accuracy, train_f1, val_loss, val_accuracy, val_f1 = train(
        model,
        dataloader_train=dataloader_train,
        dataloader_val=dataloader_val,
        epochs_num=args.epochs_num,
    )

    # train viz
    train_visualization(
        train_loss, train_accuracy, train_f1, val_loss, val_accuracy, val_f1
    )
