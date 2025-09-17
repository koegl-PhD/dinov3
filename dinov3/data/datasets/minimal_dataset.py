# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import os
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple

from .decoders import ImageDataDecoder, TargetDecoder
from .extended import ExtendedVisionDataset


def _normalize_extensions(exts: Optional[Iterable[str]]) -> Tuple[str, ...]:
    if exts is None:
        return (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")
    return tuple({e.lower() if e.startswith(".") else f".{e.lower()}" for e in exts})


class MinimalDataset(ExtendedVisionDataset):
    """
    Minimal directory dataset:
      - Recursively reads all image files under root (configurable).
      - Returns (image, target) where target is the relative file path.
      - Uses ImageDataDecoder to decode raw bytes into a PIL image.
      - Applies transforms in the same way as other datasets in this repo.

    This integrates with loaders.make_dataset via dataset string:
      MinimalDataset:root=/path/to/images
    """

    def __init__(
        self,
        *,
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        extensions: Optional[Sequence[str]] = None,
        recursive: bool = True,
        follow_symlinks: bool = False,
        sort_files: bool = True,
        # Accept and ignore common extra kwargs from parser (e.g., split/extra)
        **_: Any,
    ) -> None:
        super().__init__(
            root=root,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
            image_decoder=ImageDataDecoder,
            target_decoder=TargetDecoder,
        )

        self._extensions = _normalize_extensions(extensions)
        self._recursive = recursive
        self._follow_symlinks = follow_symlinks

        # Collect file list
        self._files: List[str] = self._collect_files(self.root)
        if sort_files:
            self._files.sort()

        if len(self._files) == 0:
            raise RuntimeError(f"No image files found under: {self.root}")

        # ExtendedVisionDataset only uses self.transforms. If only transform/target_transform
        # were provided, wrap them as a combined callable so they are applied.
        if self.transforms is None and (self.transform is not None or self.target_transform is not None):
            def _combined_transforms(image, target):
                if self.transform is not None:
                    image = self.transform(image)
                if self.target_transform is not None:
                    target = self.target_transform(target)
                return image, target

            self.transforms = _combined_transforms

    def _collect_files(self, root: str) -> List[str]:
        files: List[str] = []
        if self._recursive:
            for dirpath, dirnames, filenames in os.walk(root, followlinks=self._follow_symlinks):
                for fname in filenames:
                    if self._is_supported_file(fname):
                        files.append(os.path.join(dirpath, fname))
        else:
            try:
                for fname in os.listdir(root):
                    full = os.path.join(root, fname)
                    if os.path.isfile(full) and self._is_supported_file(fname):
                        files.append(full)
            except OSError as e:
                raise RuntimeError(f'Cannot read directory "{root}"') from e
        return files

    def _is_supported_file(self, filename: str) -> bool:
        return os.path.splitext(filename)[1].lower() in self._extensions

    def get_image_data(self, index: int) -> bytes:
        path = self._files[index]
        try:
            with open(path, mode="rb") as f:
                return f.read()
        except OSError as e:
            raise RuntimeError(
                f'Cannot read image file "{path}" (index {index})') from e

    def get_target(self, index: int) -> Any:
        # Use relative path from root as a lightweight target
        # (train.py usually passes target_transform=lambda _: () so it gets discarded)
        abs_path = self._files[index]
        rel = os.path.relpath(abs_path, self.root)
        return rel

    def __len__(self) -> int:
        return len(self._files)
