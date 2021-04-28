import glob
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from model import *


def pred_and_viz(img_path, mask_path, model_path: str = "baseline_model.pth", resize:int = 212):
    # check device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # read image and mask
    pil_img = Image.open(img_path)
    pil_mask = Image.open(mask_path)
    # to tensor
    transform = transforms.Compose(
        [transforms.Resize((resize, resize)), transforms.ToTensor()]
    )
    img = transform(pil_img).unsqueeze(0)
    img = img.to(device)
    # load model
    model = UNet()
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    # get pred
    start_time = time.time()
    pred = model(img)
    pred_time = time.time() - start_time
    pred_img = torch.sigmoid(pred)[0]
    mask_pred = pred_img[0] > 0.5
    # get viz data
    img_viz = np.array(pil_img)
    mask_viz = np.array(pil_mask)
    pred_viz = mask_pred.data.cpu().numpy()
    # viz
    fig, ax = plt.subplots(1, 3, figsize=(15, 6))
    ax[0].imshow(img_viz)
    ax[0].title.set_text("Image")
    ax[1].imshow(mask_viz, cmap="gray")
    ax[1].title.set_text("GT Mask")
    ax[2].imshow(pred_viz, cmap="gray")
    ax[2].title.set_text("Inference")
    plt.show()

    return pred_time
