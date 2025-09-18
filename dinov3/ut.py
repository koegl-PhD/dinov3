from torch import nn
from typing import List, Tuple
from typing import Tuple, Dict, Iterable, List
import torch
from torch import nn, Tensor
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
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
from torchvision.transforms import InterpolationMode

import vis
import copy
import torch
from torch import nn


IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

PATCH_SIZE = 16
IMAGE_SIZE = 768


def swap_backbones_with_hub(model: nn.Module, repo_dir: str, model_name: str, ckpt_path: str) -> None:
    """Build the exact Hub backbone and install it as student/teacher/model_ema.backbone."""
    hub_bb = torch.hub.load(
        repo_or_dir=repo_dir, model=model_name, source="local",
        pretrained=False, skip_validation=True, trust_repo=True
    )
    state = torch.load(ckpt_path, map_location="cpu")
    payload = state.get("model", state)
    hub_bb.load_state_dict(payload, strict=True)
    hub_bb.eval()

    # install *identical* modules (teacher/ema start from same weights)
    model.student["backbone"] = copy.deepcopy(hub_bb)
    model.teacher["backbone"] = copy.deepcopy(hub_bb)
    model.model_ema["backbone"] = copy.deepcopy(hub_bb)


def set_ln_eps(module: nn.Module, eps: float) -> None:
    """Set LayerNorm.eps for all LayerNorms in module."""
    for m in module.modules():
        if isinstance(m, nn.LayerNorm):
            m.eps = eps


def align_ln_eps_all_backbones(model: nn.Module, eps: float = 1e-5) -> None:
    """Set LN eps on student/teacher/model_ema backbones to given value."""
    for name in ("student", "teacher", "model_ema"):
        bb = getattr(model, name)["backbone"]
        set_ln_eps(bb, eps)


def load_model() -> torch.nn.Module:
    DINOV3_REPO_LOCATION = r"/home/koeglf/Documents/code/dinov3"
    DINOV3_MODEL_LOCATION = r"/home/koeglf/Downloads/dinov3_models/dinov3_vits16.pth"
    MODEL_NAME = "dinov3_vits16"

    model = torch.hub.load(
        repo_or_dir=DINOV3_REPO_LOCATION,   # local repo folder with hubconf.py
        model=MODEL_NAME,              # e.g. "dinov3_vits16"
        source="local",                 # never GitHub
        pretrained=False,               # don't fetch weights
        force_reload=False,             # don't refresh from remote
        skip_validation=True,           # avoid network checks
        trust_repo=True,                # suppress extra validation that can hit network
    )

    state = torch.load(DINOV3_MODEL_LOCATION, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.cuda()

    return model


def resize_transform(img: Image.Image, size: int = 224) -> Tensor:
    """Resize->center-crop to 224 and to_tensor (no normalization)."""
    img = TF.resize(img, size, interpolation=InterpolationMode.BICUBIC)
    img = TF.center_crop(img, size)
    return TF.to_tensor(img)


def infer_single_image(model: nn.Module, image_path: str, save_path: str, x0: int, y0: int) -> Tensor:
    """Get last layer feature map [C,H,W] from the student backbone deterministically."""
    # locate backbone and its dtype/device
    backbone: nn.Module = model.student["backbone"]
    was_training = backbone.training
    backbone.eval()

    param = next(p for p in backbone.parameters()
                 if p.requires_grad or p.numel() > 0)
    device = param.device
    dtype = param.dtype  # likely torch.bfloat16 in your run

    img = Image.open(image_path).convert("RGB")
    x = resize_transform(img, 224)
    x = TF.normalize(x, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    x = x.unsqueeze(0).to(device, non_blocking=True)

    with torch.no_grad():
        # autocast to the backbone’s dtype for consistent numerics
        use_amp = dtype in (torch.bfloat16, torch.float16)
        ctx = torch.autocast(
            device_type=device.type, dtype=dtype) if use_amp else torch.cuda.amp.autocast(enabled=False)
        with ctx:
            feats = backbone.get_intermediate_layers(
                x, n=12, reshape=True, norm=True)
            last_feat: Tensor = feats[-1].squeeze().detach().cpu().float()
            vis.simple_plot(last_feat, x0, y0, save_path)

    if was_training:
        backbone.train()

    return last_feat


def old_run(image_path: str, save_path: str, x0: int, y0: int):

    model = load_model()

    image = Image.open(image_path)
    image_resized = resize_transform(image)
    image_resized_norm = TF.normalize(
        image_resized, mean=IMAGENET_MEAN, std=IMAGENET_STD)

    n_layers = 12

    with torch.inference_mode():
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            feats = model.get_intermediate_layers(image_resized_norm.unsqueeze(
                0).cuda(), n=range(n_layers), reshape=True, norm=True)
            x = feats[-1].squeeze().detach().cpu().to(torch.float32)

            vis.simple_plot(x, x0, y0, save_path)

            return x

            # app = vis.build_app(x, x)
            # app.run(debug=True, port=8051)

            x = 0
            y = 0


def load_backbone_into_ssl(model: nn.Module, ckpt_path: str,
                           prefixes: Iterable[str] = ("student.backbone",
                                                      "teacher.backbone",
                                                      "model_ema.backbone")) -> Tuple[List[str], List[str]]:
    """Map a backbone-only checkpoint (keys like 'blocks.*') into SSLMetaArch backbones."""
    msd: Dict[str, Tensor] = model.state_dict()
    raw = torch.load(ckpt_path, map_location="cpu")
    payload: Dict[str, Tensor] = raw.get(
        "model", raw)  # accept both {model:...} or flat

    remapped: Dict[str, Tensor] = {}
    for k, v in payload.items():
        for pref in prefixes:
            dst = f"{pref}.{k}"
            if dst in msd and msd[dst].shape == v.shape:
                remapped[dst] = v

    missing, unexpected = model.load_state_dict(remapped, strict=False)
    return list(missing), list(unexpected)


def scan_params(module: nn.Module) -> List[Tuple[str, str]]:
    """Return (name, issue) for any param that is NaN/Inf or meta."""
    issues: List[Tuple[str, str]] = []
    for name, p in module.named_parameters():
        if getattr(p, "is_meta", False):
            issues.append((name, "META"))
            continue
        if p.is_floating_point():
            if torch.isnan(p).any():
                issues.append((name, "NaN"))
            elif torch.isinf(p).any():
                issues.append((name, "Inf"))
    return issues
