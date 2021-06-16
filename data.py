import glob
import os
import random
from typing import List

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class PolypDataset(Dataset):
    def __init__(
        self,
        imgs_dir,
        masks_dir,
        resize: int = 252,
        apply_transform: bool = True,
        model_type: str = "UNet",
    ):
        self.imgs = sorted(imgs_dir)
        self.masks = sorted(masks_dir)
        self.size = len(self.imgs)
        self.resize = resize
        self.apply_transform = apply_transform
        self.model_type = model_type
        self.transform()  # init transform

    def __getitem__(self, index):
        assert os.path.basename(self.imgs[index]) == os.path.basename(self.masks[index])
        # load image and mask
        image = self.load_image(self.imgs[index])
        mask = self.load_mask(self.masks[index])
        # apply transform
        seed = np.random.randint(1e6)
        # to image
        random.seed(seed)
        torch.manual_seed(seed)
        image = self.img_transform(image)
        # to mask
        random.seed(seed)
        torch.manual_seed(seed)
        mask = self.mask_transform(mask)

        return image, mask

    def __len__(self):
        return self.size

    def load_image(self, path):
        img = Image.open(path)
        return img.convert("RGB")

    def load_mask(self, path):
        mask = Image.open(path)
        return mask.convert("1")

    def transform(self):
        if self.model_type == "HardNetMSEG":
            if self.apply_transform:
                self.img_transform = transforms.Compose(
                    [
                        transforms.RandomRotation(90),
                        transforms.RandomVerticalFlip(p=0.5),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.Resize((self.resize, self.resize)),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                        ),
                    ]
                )

                self.mask_transform = transforms.Compose(
                    [
                        transforms.RandomRotation(90),
                        transforms.RandomVerticalFlip(p=0.5),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.Resize((self.resize, self.resize)),
                        transforms.ToTensor(),
                    ]
                )
            else:
                self.img_transform = transforms.Compose(
                    [
                        transforms.Resize((self.resize, self.resize)),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                        ),
                    ]
                )
                self.mask_transform = transforms.Compose(
                    [
                        transforms.Resize((self.resize, self.resize)),
                        transforms.ToTensor(),
                    ]
                )
        elif self.model_type == "UNet":
            if self.apply_transform:
                self.img_transform = transforms.Compose(
                    [
                        transforms.RandomRotation(90),
                        transforms.RandomVerticalFlip(p=0.5),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.Resize((self.resize, self.resize)),
                        transforms.ToTensor(),
                    ]
                )

                self.mask_transform = transforms.Compose(
                    [
                        transforms.RandomRotation(90),
                        transforms.RandomVerticalFlip(p=0.5),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.Resize((self.resize, self.resize)),
                        transforms.ToTensor(),
                    ]
                )
            else:
                self.img_transform = transforms.Compose(
                    [
                        transforms.Resize((self.resize, self.resize)),
                        transforms.ToTensor(),
                    ]
                )
                self.mask_transform = transforms.Compose(
                    [
                        transforms.Resize((self.resize, self.resize)),
                        transforms.ToTensor(),
                    ]
                )


def get_data(
    imgs_dir: List[str],
    masks_dir: List[str],
    resize: int = 252,
    batch_size: int = 4,
    model_type: str = "UNet",
):
    # split data
    imgs_path = glob.glob(imgs_dir)
    masks_path = glob.glob(masks_dir)
    np.random.seed(42)
    np.random.shuffle(imgs_path)
    np.random.seed(42)
    np.random.shuffle(masks_path)
    assert [i.split("\\")[-1] for i in imgs_path] == [
        i.split("\\")[-1] for i in masks_path
    ]
    train_imgs, test_imgs, val_imgs = (
        imgs_path[:800],
        imgs_path[-200:-100],
        imgs_path[-100:],
    )
    train_masks, test_masks, val_masks = (
        masks_path[:800],
        masks_path[-200:-100],
        masks_path[-100:],
    )
    # init datasets
    dataset_train = PolypDataset(
        train_imgs, train_masks, resize, apply_transform=True, model_type=model_type
    )
    dataset_test = PolypDataset(
        test_imgs, test_masks, resize, apply_transform=False, model_type=model_type
    )
    dataset_val = PolypDataset(
        val_imgs, val_masks, resize, apply_transform=False, model_type=model_type
    )
    # init dataloaders
    dataloader_train = DataLoader(
        dataset=dataset_train, batch_size=batch_size, shuffle=True, pin_memory=True
    )
    dataloader_test = DataLoader(
        dataset=dataset_test, batch_size=1, shuffle=False, pin_memory=True
    )
    dataloader_val = DataLoader(
        dataset=dataset_val, batch_size=batch_size, shuffle=False, pin_memory=True
    )

    return dataloader_train, dataloader_test, dataloader_val
