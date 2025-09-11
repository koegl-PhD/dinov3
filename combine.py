from typing import Literal

import torch


shape = (10, 20, 30)

image = torch.randn(shape)


def get_slice(volume: torch.Tensor, orientation: Literal['axial', 'sagittal', 'coronal'], idx: int) -> torch.Tensor:

    if orientation == 'axial':
        dim = 0
    elif orientation == 'sagittal':
        dim = 1
    elif orientation == 'coronal':
        dim = 2
    else:
        raise ValueError(
            "Wrong orientation, only 'axial', 'sagittal', 'coronal' are accepted")

    return torch.index_select(volume, dim, torch.tensor([idx], device=volume.device)).squeeze(dim)


def dino_simulator(image: torch.Tensor, value: float) -> torch.Tensor:

    shape = image.shape
    if shape[0] != 3 and len(shape) != 3:
        raise ValueError("image must have shape (3,H,W)")

    new_shape = (7, shape[1], shape[2])

    features = torch.ones(new_shape) * value

    return features


def encode_slice(slice: torch.Tensor, value: int) -> torch.Tensor:

    if len(slice.shape) != 2:
        raise ValueError("slice has to be 2D")

    slice_rgb = slice.unsqueeze(0).expand(3, -1, -1)

    features = dino_simulator(slice_rgb, value)

    return features


def encode_volume(volume: torch.Tensor) -> torch.Tensor:

    orientations = ['axial', 'sagittal', 'coronal']

    shape = volume.shape

    result = None

    i = 0
    for sh, ori in zip(shape, orientations):
        i += 1

        temp_features = []

        for idx in range(sh):

            sl = get_slice(volume, ori, idx)

            sl_features = encode_slice(sl, i)

            temp_features.append(sl_features)

        # combine features into one volume
        temp_volume = torch.stack(temp_features, dim=i)

        if result is None:
            result = temp_volume
        else:
            # append along channel dimension
            result = torch.cat((result, temp_volume), dim=0)

    return result


if __name__ == "__main__":

    features = encode_volume(image)

    print("image shape:", image.shape)
    print("features shape:", features.shape)
