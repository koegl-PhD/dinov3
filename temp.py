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
    # cat_image = Image.open(r"/home/fryderyk/Downloads/cat.jpg").convert("RGB")
    # cat_image_resized = resize_transform(cat_image)
    # cat_image_resized_norm = TF.normalize(
    #     cat_image_resized, IMAGENET_MEAN, IMAGENET_STD)
    cat1_tensor = torch.load(r"/home/fryderyk/Downloads/tensor.pt")
    cat2_tensor = torch.load(r"/home/fryderyk/Downloads/cat2_feat.pt")

    app = vis.build_app(cat1_tensor, cat2_tensor)
    app.run(debug=True, port=8050)
    x = 0
