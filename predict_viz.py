import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from model import *
from model_baseline import *


def pred_and_viz(img_path, mask_path, model_type, model_path: str, resize: int, title: str):
    # check device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # read image and mask
    pil_img = Image.open(img_path)
    pil_mask = Image.open(mask_path)
    # load model
    if model_type == "UNet":
        model = UNet().to(device)
        transform = transforms.Compose(
            [transforms.Resize((resize, resize)), transforms.ToTensor(),]
        )
    elif model_type == "HardNetMSEG":
        model = HarDMSEG().to(device)
        transform = transforms.Compose(
            [
                transforms.Resize((resize, resize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    model.load_state_dict(torch.load(model_path))
    model.eval()
    # to tensor
    transform_input = transforms.Compose(
        [transforms.Resize((resize, resize)), transforms.ToTensor(),]
    )
    img = transform(pil_img).unsqueeze(0)
    pil_img = transform_input(pil_img)
    pil_mask = transform_input(pil_mask)
    img = img.to(device)
    # get pred
    pred = model(img)
    pred_img = torch.sigmoid(pred)[0]
    mask_pred = pred_img[0] > 0.3
    # get viz data
    img_viz = np.moveaxis(np.array(pil_img), 0, -1)
    mask_viz = np.moveaxis(np.array(pil_mask), 0, -1)
    pred_viz = mask_pred.data.cpu().numpy()
    # viz
    fig, ax = plt.subplots(1, 3, figsize=(15, 6))
    ax[0].imshow(img_viz)
    ax[0].title.set_text("Image")
    ax[1].imshow(mask_viz, cmap="gray")
    ax[1].title.set_text("GT Mask")
    ax[2].imshow(pred_viz, cmap="gray")
    ax[2].title.set_text("Inference")
    fig.suptitle(title)
    plt.show()

    return img_viz, mask_viz, pred_viz
