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

import utils


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

model = utils.load_model()
image_path = "/home/koeglf/Downloads/cat_top.jpg"


image = Image.open(image_path)
image_resized = utils.resize_transform(image)
image_resized_norm = TF.normalize(
    image_resized, mean=IMAGENET_MEAN, std=IMAGENET_STD)

n_layers = 12

with torch.inference_mode():
    with torch.autocast(device_type='cuda', dtype=torch.float32):
        feats = model.get_intermediate_layers(image_resized_norm.unsqueeze(
            0).cuda(), n=range(n_layers), reshape=True, norm=True)
        x = feats[-1].squeeze().detach().cpu()
        dim = x.shape[0]
        x = x.view(dim, -1).permute(1, 0)

x = 0
