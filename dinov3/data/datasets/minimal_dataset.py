import os
from enum import Enum

from typing import Union, List, Optional, Sequence

import nibabel as nib
import torch

from .extended import ExtendedVisionDataset
from .decoders import ImageDataDecoder, TargetDecoder


class _Split(Enum):
    TRAIN = "train"
    VAL = "val"


class MinimalDataset(ExtendedVisionDataset):
    """Very small dataset that reads images from a directory tree.
    Args:
    root: Directory containing images (recursively searched).
    transforms: Optional joint (image, target) transform.
    transform: Optional image-only transform.
    target_transform: Optional target-only transform.
    extensions: Allowed file extensions.
    recursive: If True, search subfolders.
    sorted_order: If True, sort file list for determinism.
    Returns image bytes -> decoded by ImageDataDecoder; target is 0.
    """
    Target = int
    Split = Union[_Split]

    def __init__(
        self,
        *,
        split: "MinimalDataset.Split",
        root: str,
        transforms=None,
        transform=None,
        target_transform=None,
        extension: str = ".nii.gz",
        sorted_order: bool = True,
    ) -> None:
        super().__init__(
            root=root,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
            image_decoder=ImageDataDecoder,
            target_decoder=TargetDecoder,
        )
        self._split = split
        files: List[str] = []
        for fn in os.listdir(root):
            fp = os.path.join(root, fn)
            if os.path.isfile(fp) and fn.endswith(extension):
                files.append(fp)
        if sorted_order:
            files.sort()
        if not files:
            raise RuntimeError(f"No images found under '{root}'.")
        self._files = files

    @property
    def split(self) -> "MinimalDataset.Split":
        return self._split

    def get_image_data(self, index: int) -> torch.Tensor:

        volume = nib.load(self._files[index])

        volume_data = torch.from_numpy(volume.get_fdata())

        return volume_data

    def get_target(self, index: int) -> Target:
        return 0

    def __len__(self) -> int:
        return len(self._files)


class MinimalDatasets(ExtendedVisionDataset):
    """Very small dataset that reads images from a directory tree.
    Args:
    root: Directory containing images (recursively searched).
    transforms: Optional joint (image, target) transform.
    transform: Optional image-only transform.
    target_transform: Optional target-only transform.
    extensions: Allowed file extensions.
    recursive: If True, search subfolders.
    sorted_order: If True, sort file list for determinism.
    Returns image bytes -> decoded by ImageDataDecoder; target is 0.
    """
    Target = int
    Split = Union[_Split]

    def __init__(
        self,
        *,
        split: "MinimalDataset.Split",
        root: str,
        transforms=None,
        transform=None,
        target_transform=None,
        extensions: Optional[Sequence[str]] = (".jpg", ".jpeg", ".png", ".bmp",
                                               ".webp", ".tif", ".tiff"),
        sorted_order: bool = True,
    ) -> None:
        super().__init__(
            root=root,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
            image_decoder=ImageDataDecoder,
            target_decoder=TargetDecoder,
        )
        self._split = split
        exts = tuple(e.lower() for e in (extensions or ()))
        files: List[str] = []
        for fn in os.listdir(root):
            fp = os.path.join(root, fn)
            if os.path.isfile(fp) and (not exts or os.path.splitext(fn)[1].lower() in exts):
                files.append(fp)
        if sorted_order:
            files.sort()
        if not files:
            raise RuntimeError(f"No images found under '{root}'.")
        self._files = files

    @property
    def split(self) -> "MinimalDataset.Split":
        return self._split

    def get_image_data(self, index: int) -> bytes:
        with open(self._files[index], "rb") as f:
            return f.read()

    def get_target(self, index: int) -> Target:
        return 0

    def __len__(self) -> int:
        return len(self._files)
