# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

from torch import Tensor, nn
import torch.nn.functional as F
import torch
from typing import Dict, List, Optional, Sequence, Tuple, Union
import logging

import numpy as np
from torch import nn
from torchvision import transforms

from dinov3.data.transforms import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, GaussianBlur, make_normalize_transform
import torch.nn.functional as F
from typing import Tuple

logger = logging.getLogger("dinov3")


def _to_3tuple(v: Union[int, Sequence[int]]) -> Tuple[int, int, int]:
    if isinstance(v, int):
        return (v, v, v)
    t = tuple(v)
    if len(t) != 3:
        raise ValueError("expected int or length-3 sequence")
    return t  # type: ignore[return-value]


def _rand_resized_crop_3d(x: Tensor, out_size: Tuple[int, int, int], scale: Tuple[float, float]) -> Tensor:
    _, H, W, D = x.shape
    area = H * W * D
    for _ in range(10):
        target = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
        ratio_d = torch.empty(1).uniform_(0.67, 1.5).item()
        ratio_h = torch.empty(1).uniform_(0.67, 1.5).item()
        ratio_w = torch.empty(1).uniform_(0.67, 1.5).item()
        d = int(round((target * ratio_d ** (1/3)) ** (1/3)))
        h = int(round((target * ratio_h ** (1/3)) ** (1/3)))
        w = int(round((target * ratio_w ** (1/3)) ** (1/3)))
        if 1 <= d <= D and 1 <= h <= H and 1 <= w <= W:
            z0 = torch.randint(0, D - d + 1, ()).item()
            y0 = torch.randint(0, H - h + 1, ()).item()
            x0 = torch.randint(0, W - w + 1, ()).item()
            crop = x[:, z0:z0 + d, y0:y0 + h, x0:x0 + w]
            return F.interpolate(crop.unsqueeze(0), size=out_size, mode="trilinear", align_corners=False).squeeze(0)
    dd = min(D, int(D * scale[0]))
    hh = min(H, int(H * scale[0]))
    ww = min(W, int(W * scale[0]))
    z0 = (D - dd) // 2
    y0 = (H - hh) // 2
    x0 = (W - ww) // 2
    crop = x[:, z0:z0 + dd, y0:y0 + hh, x0:x0 + ww]
    return F.interpolate(crop.unsqueeze(0), size=out_size, mode="trilinear", align_corners=False).squeeze(0)


def _hflip_w(x: Tensor, p: float) -> Tensor:
    return x.flip(-1) if torch.rand(()) < p else x


def _brightness_contrast(x: Tensor, b: float, c: float) -> Tensor:
    mean = x.mean()
    return torch.clamp((x - mean) * c + mean + b, min=x.min(), max=x.max())


def _color_jitter_like(x: Tensor, p: float = 0.8) -> Tensor:
    if torch.rand(()) >= p:
        return x
    b = torch.empty(1).uniform_(-0.2, 0.2).item()
    c = torch.empty(1).uniform_(0.8, 1.2).item()
    return _brightness_contrast(x, b, c)


def _gaussian_kernel1d(sigma: float, radius: int) -> Tensor:
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    k = k / k.sum()
    return k


def _gaussian_blur_3d(x: Tensor, p: float, sigma_range: Tuple[float, float] = (0.5, 1.5)) -> Tensor:
    if torch.rand(()) >= p:
        return x
    sigma = torch.empty(1).uniform_(sigma_range[0], sigma_range[1]).item()
    radius = max(1, int(3.0 * sigma))
    k = _gaussian_kernel1d(sigma, radius).to(x.device, x.dtype)
    kx = k.view(1, 1, 1, 1, -1)
    ky = k.view(1, 1, 1, -1, 1)
    kz = k.view(1, 1, -1, 1, 1)
    x5 = x.unsqueeze(0)
    x5 = F.conv3d(F.pad(x5, (radius, radius, radius, radius, radius, radius),
                  mode="replicate"), kx.expand(x.shape[0], 1, 1, 1, -1), groups=x.shape[0])
    x5 = F.conv3d(F.pad(x5, (0, 0, radius, radius, radius, radius),
                  mode="replicate"), ky.expand(x.shape[0], 1, 1, -1, 1), groups=x.shape[0])
    x5 = F.conv3d(F.pad(x5, (0, 0, 0, 0, radius, radius), mode="replicate"),
                  kz.expand(x.shape[0], 1, -1, 1, 1), groups=x.shape[0])
    return x5.squeeze(0)


def _resize_to(x: Tensor, size3: Tuple[int, int, int]) -> Tensor:
    """Resize [C,D,H,W] to exactly size3 via trilinear."""
    return F.interpolate(x.unsqueeze(0), size=size3, mode="trilinear", align_corners=False).squeeze(0)


def _normalize_single_channel(x: Tensor, mean: List[float], std: List[float]) -> Tensor:
    m = sum(mean)/len(mean)
    s = sum(std)/len(std)
    return (x - m) / s


class DataAugmentationDINO(object):
    def __init__(
        self,
        global_crops_scale: Tuple[float, float],
        local_crops_scale: Tuple[float, float],
        local_crops_number: int,
        global_crops_size: Union[int, Sequence[int]] = 224,
        local_crops_size: Union[int, Sequence[int]] = 96,
        gram_teacher_crops_size: Optional[Union[int, Sequence[int]]] = None,
        gram_teacher_no_distortions: bool = False,
        teacher_no_color_jitter: bool = False,
        local_crops_subset_of_global_crops: bool = False,
        patch_size: Union[int, Sequence[int]] = 16,
        share_color_jitter: bool = False,
        horizontal_flips: bool = True,
        mean: Union[float, Sequence[float]] = 0.5,
        std: Union[float, Sequence[float]] = 0.5,
    ) -> None:
        """DINO-style augmentations for 3D volumes without channel; expects [D,H,W]."""
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size3 = _to_3tuple(global_crops_size)
        self.local_crops_size3 = _to_3tuple(local_crops_size)
        self.gram_teacher_crops_size3 = _to_3tuple(
            gram_teacher_crops_size) if gram_teacher_crops_size is not None else None
        self.gram_teacher_no_distortions = gram_teacher_no_distortions
        self.teacher_no_color_jitter = teacher_no_color_jitter
        self.local_crops_subset_of_global_crops = local_crops_subset_of_global_crops
        self.patch_size3 = _to_3tuple(patch_size)
        self.share_color_jitter = share_color_jitter
        self.horizontal_flips = horizontal_flips
        self.mean = mean
        self.std = std

        max_g = tuple(
            max(a, b) for a, b in zip(self.global_crops_size3, self.gram_teacher_crops_size3 or (0, 0, 0))
        )
        self._global_crop_max_size3 = max_g

    def _normalize(self, x: Tensor) -> Tensor:
        return _normalize_single_channel(x, self.mean, self.std)

    def _geo_global(self, x: Tensor) -> Tensor:
        x = _rand_resized_crop_3d(
            x, self._global_crop_max_size3, self.global_crops_scale)
        if self.horizontal_flips:
            x = _hflip_w(x, 0.5)
        return x

    def _geo_local(self, x: Tensor) -> Tensor:
        x = _rand_resized_crop_3d(
            x, self.local_crops_size3, self.local_crops_scale)
        if self.horizontal_flips:
            x = _hflip_w(x, 0.5)
        return x

    def _transfo1(self, x: Tensor) -> Tensor:
        x = _gaussian_blur_3d(x, p=1.0)
        return self._normalize(x)

    def _transfo2(self, x: Tensor) -> Tensor:
        x = _gaussian_blur_3d(x, p=0.1)
        return self._normalize(x)

    def _local_transfo(self, x: Tensor) -> Tensor:
        x = _gaussian_blur_3d(x, p=0.5)
        return self._normalize(x)

    def __call__(self, volume: Tensor) -> Dict[str, Union[bool, List[Tensor], Tuple]]:
        """Apply DINO-style multi-crop pipeline to a [D,H,W] tensor and return [D,H,W] crops."""
        if volume.dim() != 3:
            raise ValueError(f"Expected [D,H,W], got {tuple(volume.shape)}")
        v = volume.unsqueeze(0)  # -> [1,D,H,W] for ops

        if self.share_color_jitter:
            v = _color_jitter_like(v)

        im1_base = self._geo_global(v)
        g1 = self._transfo1(im1_base)
        g1_post = F.interpolate(g1.unsqueeze(
            0), size=self.global_crops_size3, mode="trilinear", align_corners=False).squeeze(0)

        im2_base = self._geo_global(v)
        g2 = self._transfo2(im2_base)
        g2_post = F.interpolate(g2.unsqueeze(
            0), size=self.global_crops_size3, mode="trilinear", align_corners=False).squeeze(0)

        out: Dict[str, Union[bool, List[Tensor], Tuple]] = {"weak_flag": True}
        out["global_crops"] = [g1_post.squeeze(0), g2_post.squeeze(0)]

        if self.teacher_no_color_jitter:
            out["global_crops_teacher"] = [self._normalize(
                im1_base).squeeze(0), self._normalize(im2_base).squeeze(0)]
        else:
            out["global_crops_teacher"] = [
                g1_post.squeeze(0), g2_post.squeeze(0)]

        if self.gram_teacher_crops_size3 is not None:
            if self.gram_teacher_no_distortions:
                gram1 = self._normalize(
                    F.interpolate(im1_base.unsqueeze(
                        0), size=self.gram_teacher_crops_size3, mode="trilinear", align_corners=False).squeeze(0)
                )
                gram2 = self._normalize(
                    F.interpolate(im2_base.unsqueeze(
                        0), size=self.gram_teacher_crops_size3, mode="trilinear", align_corners=False).squeeze(0)
                )
            else:
                gram1 = F.interpolate(g1.unsqueeze(
                    0), size=self.gram_teacher_crops_size3, mode="trilinear", align_corners=False).squeeze(0)
                gram2 = F.interpolate(g2.unsqueeze(
                    0), size=self.gram_teacher_crops_size3, mode="trilinear", align_corners=False).squeeze(0)
            out["gram_teacher_crops"] = [gram1.squeeze(0), gram2.squeeze(0)]

        if self.local_crops_subset_of_global_crops:
            locals_final: List[Tensor] = []
            offsets: List[Tuple[int, int, int]] = []
            gs = self.global_crops_size3
            ls = self.local_crops_size3
            ps = self.patch_size3
            for img in _locals:
                rz = int(np.random.randint(
                    0, max(1, (gs[0] - ls[0]) // ps[0] + 1))) * ps[0]
                ry = int(np.random.randint(
                    0, max(1, (gs[1] - ls[1]) // ps[1] + 1))) * ps[1]
                rx = int(np.random.randint(
                    0, max(1, (gs[2] - ls[2]) // ps[2] + 1))) * ps[2]
                patch = img[:, rz: rz + ls[0], ry: ry + ls[1], rx: rx + ls[2]]
                patch = _resize_to(patch, ls)  # <- enforce exact [ls]
                locals_final.append(patch.squeeze(0))
                offsets.append((rz, ry, rx))
            out["local_crops"] = locals_final
            out["offsets"] = offsets
        else:
            out["local_crops"] = [
                _resize_to(self._local_transfo(self._geo_local(v)),
                           self.local_crops_size3).squeeze(0)
                for _ in range(self.local_crops_number)
            ]
            out["offsets"] = ()

        return out


class DataAugmentationDINOo(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
        gram_teacher_crops_size=None,
        gram_teacher_no_distortions=False,
        teacher_no_color_jitter=False,
        local_crops_subset_of_global_crops=False,
        patch_size=16,
        share_color_jitter=False,
        horizontal_flips=True,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size
        self.gram_teacher_crops_size = gram_teacher_crops_size
        self.gram_teacher_no_distortions = gram_teacher_no_distortions
        self.teacher_no_color_jitter = teacher_no_color_jitter
        self.local_crops_subset_of_global_crops = local_crops_subset_of_global_crops
        self.patch_size = patch_size
        self.share_color_jitter = share_color_jitter
        self.mean = mean
        self.std = std

        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info(f"gram_crops_size: {gram_teacher_crops_size}")
        logger.info(
            f"gram_teacher_no_distortions: {gram_teacher_no_distortions}")
        logger.info(f"teacher_no_color_jitter: {teacher_no_color_jitter}")
        logger.info(
            f"local_crops_subset_of_global_crops: {local_crops_subset_of_global_crops}")
        logger.info(
            f"patch_size if local_crops_subset_of_global_crops: {patch_size}")
        logger.info(f"share_color_jitter: {share_color_jitter}")
        logger.info(f"horizontal flips: {horizontal_flips}")
        logger.info("###################################")

        # Global crops and gram teacher crops can have different sizes. We first take a crop of the maximum size
        # and then resize it to the desired size for global and gram teacher crops.
        global_crop_max_size = max(
            global_crops_size, gram_teacher_crops_size if gram_teacher_crops_size else 0)

        # random resized crop and flip
        self.geometric_augmentation_global = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    global_crop_max_size,
                    scale=global_crops_scale,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(
                    p=0.5 if horizontal_flips else 0.0),
            ]
        )

        # Resize transform applied to global crops after random crop
        resize_global = nn.Identity()
        self.resize_global_post_transf = (
            nn.Identity()
        )  # Resize transform applied to global crops after all other transforms
        # Resize transform applied to crops for gram teacher
        self.resize_gram_teacher = None
        if gram_teacher_crops_size is not None:
            # All resize transforms will do nothing if the crop size is already the desired size.
            if gram_teacher_no_distortions:
                # When there a no distortions for the gram teacher crop, we can resize before the distortions.
                # This is the preferred order, because it keeps the image size for the augmentations consistent,
                # which matters e.g. for GaussianBlur.
                resize_global = transforms.Resize(
                    global_crops_size,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                )
            else:
                # When there a no distortions for the gram teacher crop, we need to resize after the distortions,
                # because the distortions are shared between global and gram teacher crops.
                self.resize_global_post_transf = transforms.Resize(
                    global_crops_size,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                )

            self.resize_gram_teacher = transforms.Resize(
                gram_teacher_crops_size,
                interpolation=transforms.InterpolationMode.BICUBIC,
            )

        self.geometric_augmentation_local = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    local_crops_size,
                    scale=local_crops_scale,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(
                    p=0.5 if horizontal_flips else 0.0),
            ]
        )

        # color distortions / blurring
        color_jittering = transforms.Compose(
            [
                transforms.RandomApply(
                    [transforms.ColorJitter(
                        brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )

        global_transfo1_extra = GaussianBlur(p=1.0)

        global_transfo2_extra = transforms.Compose(
            [
                GaussianBlur(p=0.1),
                transforms.RandomSolarize(threshold=128, p=0.2),
            ]
        )

        local_transfo_extra = GaussianBlur(p=0.5)

        # normalization
        self.normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                make_normalize_transform(mean=mean, std=std),
            ]
        )

        if self.share_color_jitter:
            self.color_jittering = color_jittering
            self.global_transfo1 = transforms.Compose(
                [resize_global, global_transfo1_extra, self.normalize])
            self.global_transfo2 = transforms.Compose(
                [resize_global, global_transfo2_extra, self.normalize])
            self.local_transfo = transforms.Compose(
                [local_transfo_extra, self.normalize])
        else:
            self.global_transfo1 = transforms.Compose(
                [resize_global, color_jittering,
                    global_transfo1_extra, self.normalize]
            )
            self.global_transfo2 = transforms.Compose(
                [resize_global, color_jittering,
                    global_transfo2_extra, self.normalize]
            )
            self.local_transfo = transforms.Compose(
                [color_jittering, local_transfo_extra, self.normalize])

    def __call__(self, image):
        output = {}
        output["weak_flag"] = True  # some residual from mugs

        if self.share_color_jitter:
            image = self.color_jittering(image)

        # global crops:
        im1_base = self.geometric_augmentation_global(image)
        global_crop_1_transf = self.global_transfo1(im1_base)
        global_crop_1 = self.resize_global_post_transf(global_crop_1_transf)

        im2_base = self.geometric_augmentation_global(image)
        global_crop_2_transf = self.global_transfo2(im2_base)
        global_crop_2 = self.resize_global_post_transf(global_crop_2_transf)

        output["global_crops"] = [global_crop_1, global_crop_2]

        # global crops for teacher:
        if self.teacher_no_color_jitter:
            output["global_crops_teacher"] = [
                self.normalize(im1_base),
                self.normalize(im2_base),
            ]
        else:
            output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        if self.gram_teacher_crops_size is not None:
            # crops for gram teacher:
            if self.gram_teacher_no_distortions:
                gram_crop_1 = self.normalize(
                    self.resize_gram_teacher(im1_base))
                gram_crop_2 = self.normalize(
                    self.resize_gram_teacher(im2_base))
            else:
                gram_crop_1 = self.resize_gram_teacher(global_crop_1_transf)
                gram_crop_2 = self.resize_gram_teacher(global_crop_2_transf)
            output["gram_teacher_crops"] = [gram_crop_1, gram_crop_2]

        # local crops:
        if self.local_crops_subset_of_global_crops:
            _local_crops = [self.local_transfo(im1_base) for _ in range(self.local_crops_number // 2)] + [
                self.local_transfo(im2_base) for _ in range(self.local_crops_number // 2)
            ]

            local_crops = []
            offsets = []
            gs = self.global_crops_size
            ls = self.local_crops_size
            for img in _local_crops:
                rx, ry = np.random.randint(
                    0, (gs - ls) // self.patch_size, 2) * self.patch_size
                local_crops.append(img[:, rx: rx + ls, ry: ry + ls])
                offsets.append((rx, ry))

            output["local_crops"] = local_crops
            output["offsets"] = offsets
        else:
            local_crops = [
                self.local_transfo(self.geometric_augmentation_local(image)) for _ in range(self.local_crops_number)
            ]
            output["local_crops"] = local_crops
            output["offsets"] = ()

        return output
