import pickle
import os
import urllib

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import nibabel as nib

import torch
import torchvision.transforms.functional as TF
from sklearn.decomposition import PCA
from scipy import signal

import utils
from dinov3.layers import patch_embed_3d

device = "cuda"
patch_embed = patch_embed_3d.PatchEmbed3D(
    vol_size=224,
    patch_size=16,
    embed_dim=384,
    flatten_embedding=False,
).to(device)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

model = utils.load_model()
model.patch_embed = patch_embed
image_path = "/home/koeglf/Downloads/cat_top.jpg"

volume = torch.from_numpy(np.asarray(nib.load(
    "/home/koeglf/data/LungCT/LungCT_preprocessed_div16/imagesTr/LungCT_0020_0000.nii.gz").dataobj)).to(device)

image = Image.open(image_path)
image_resized = utils.resize_transform(image)
image_resized_norm = TF.normalize(
    image_resized, mean=IMAGENET_MEAN, std=IMAGENET_STD)

n_layers = 12

with torch.inference_mode():
    with torch.autocast(device_type='cuda', dtype=torch.float32):
        feats = model.get_intermediate_layers(
            volume.unsqueeze(0).cuda(), n=range(n_layers), reshape=True, norm=True)
        x = feats[-1].squeeze().detach().cpu()
        dim = x.shape[0]
        x = x.view(dim, -1).permute(1, 0)

x = 0
