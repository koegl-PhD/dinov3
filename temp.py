import pickle
import os
import urllib

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torchvision.transforms.functional as TF
from sklearn.decomposition import PCA
from scipy import signal

import vis

PATCH_SIZE = 16
IMAGE_SIZE = 768

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def resize_transform(
    mask_image: Image,
    image_size: int = IMAGE_SIZE,
    patch_size: int = PATCH_SIZE,
) -> torch.Tensor:
    w, h = mask_image.size
    h_patches = int(image_size / patch_size)
    w_patches = int((w * image_size) / (h * patch_size))
    return TF.to_tensor(TF.resize(mask_image, (h_patches * patch_size, w_patches * patch_size)))


if __name__ == "__main__":
    image_1 = Image.open(
        r"/home/fryderyk/Pictures/Screenshots/im1.png").convert("RGB")
    image_1_resized = resize_transform(image_1)

    image_2 = Image.open(
        r"/home/fryderyk/Pictures/Screenshots/im2.png").convert("RGB")
    image_2_resized = resize_transform(image_2)

    image_1_feat = torch.load(r"/home/fryderyk/Downloads/im1_feat.pt")
    image_2_feat = torch.load(r"/home/fryderyk/Downloads/im2_feat.pt")

    app = vis.build_app(image_1_feat, image_2_feat)
    app.run(debug=True, port=8050)
    x = 0
