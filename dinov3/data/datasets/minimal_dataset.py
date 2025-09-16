# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import os
from typing import Optional, Callable, Tuple, Any
from PIL import Image
from torch.utils.data import Dataset


class MinimalDataset(Dataset[Tuple[Any, Any]]):
    def __init__(self, root: str, transform: Optional[Callable] = None):
        self.root = root
        self.transform = transform
        self.samples = []

        # Find all image files in the root directory
        valid_extensions = ('.jpg', '.jpeg', '.png')
        for root_dir, dirs, files in os.walk(self.root):
            for file in files:
                if file.lower().endswith(valid_extensions):
                    full_path = os.path.join(root_dir, file)
                    self.samples.append(full_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        # Get the path of the image at the given index
        image_path = self.samples[index]

        # Load the image and convert to RGB
        image = Image.open(image_path).convert('RGB')

        # Apply transform if provided
        if self.transform is not None:
            image = self.transform(image)

        # Return image and dummy target (empty tuple for self-supervised learning)
        return image, ()
