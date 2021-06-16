import argparse

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from model import *


def forecast(img_path: str, resize: int):
    # check device
    device = "cpu"
    # read image and mask
    pil_img = Image.open(img_path)
    # load model
    model = HarDMSEG().to(device)
    transform = transforms.Compose(
        [
            transforms.Resize((resize, resize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    model.load_state_dict(torch.load('HardNetMSEG.pth', map_location=torch.device('cpu')))
    model.eval()
    # to tensor
    transform_input = transforms.Compose(
        [transforms.Resize((resize, resize)), transforms.ToTensor(),]
    )
    img = transform(pil_img).unsqueeze(0)
    pil_img = transform_input(pil_img)
    img = img.to(device)
    # get pred
    pred = model(img)
    pred_img = torch.sigmoid(pred)[0]
    mask_pred = pred_img[0] > 0.5
    # get viz data
    img_save = np.moveaxis(np.array(mask_pred.cpu()), 0, -1)
    img_save = Image.fromarray(img_save)
    img_save.save('Forecast.png')

def get_args():
    parser = argparse.ArgumentParser(
        description="Make forecast",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--img-path",
        dest="img_path",
        type=str,
        help="Image path",
    )
    parser.add_argument(
        "--resize", dest="resize", type=int, default=352, help="Image resize size???",
    )

    return parser.parse_args()


if __name__ == "__main__":
    # get args
    args = get_args()
    img_path = args.img_path
    resize = args.resize
    forecast(img_path, resize)
    print('Done') 
