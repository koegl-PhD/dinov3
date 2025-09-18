# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import math
from typing import Literal

import numpy as np
import torch
from torch import Tensor, nn


class RopePositionEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int,
        base: float | None = 100.0,
        min_period: float | None = None,
        max_period: float | None = None,
        normalize_coords: Literal["min", "max", "separate"] = "separate",
        shift_coords: float | None = None,
        jitter_coords: float | None = None,
        rescale_coords: float | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        """3D RoPE for (D,H,W). Pads periods to nearest multiple of 6, crops back at output."""
        super().__init__()
        self.D_head = embed_dim // num_heads
        self.base = base
        self.min_period = min_period
        self.max_period = max_period
        self.normalize_coords = normalize_coords
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords
        self.dtype = dtype

        # nearest multiple of 6
        self.D_head_padded = math.ceil(self.D_head / 6) * 6

        self.register_buffer(
            "periods",
            torch.empty(self.D_head_padded // 6, device=device, dtype=dtype),
            persistent=True,
        )
        self._init_weights()

    def forward(self, *, D: int, H: int, W: int) -> tuple[Tensor, Tensor]:
        device = self.periods.device
        dtype = self.dtype
        dd = {"device": device, "dtype": dtype}

        if self.normalize_coords == "max":
            m = max(H, W, D)
            coords_d = torch.arange(0.5, D, **dd) / m
            coords_h = torch.arange(0.5, H, **dd) / m
            coords_w = torch.arange(0.5, W, **dd) / m
        elif self.normalize_coords == "min":
            m = min(H, W, D)
            coords_d = torch.arange(0.5, D, **dd) / m
            coords_h = torch.arange(0.5, H, **dd) / m
            coords_w = torch.arange(0.5, W, **dd) / m
        elif self.normalize_coords == "separate":
            coords_d = torch.arange(0.5, D, **dd) / D
            coords_h = torch.arange(0.5, H, **dd) / H
            coords_w = torch.arange(0.5, W, **dd) / W
        else:
            raise ValueError(
                f"Unknown normalize_coords: {self.normalize_coords}")

        coords = torch.stack(
            torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij"), dim=-1
        ).flatten(0, 2)  # [DHW, 3]
        coords = 2.0 * coords - 1.0

        if self.training and self.shift_coords is not None:
            shift = torch.empty(
                3, **dd).uniform_(-self.shift_coords, self.shift_coords)
            coords += shift[None, :]

        if self.training and self.jitter_coords is not None:
            jmax = np.log(self.jitter_coords)
            jitter = torch.empty(3, **dd).uniform_(-jmax, jmax).exp()
            coords *= jitter[None, :]

        if self.training and self.rescale_coords is not None:
            rmax = np.log(self.rescale_coords)
            scale = torch.empty(1, **dd).uniform_(-rmax, rmax).exp()
            coords *= scale

        # [DHW, 3, D_head_padded//6]
        angles = 2 * math.pi * coords[:, :, None] / self.periods[None, None, :]
        angles = angles.flatten(1, 2)       # [DHW, D_head_padded//2]
        angles = angles.tile(2)             # [DHW, D_head_padded]
        cos = torch.cos(angles)[..., :self.D_head]
        sin = torch.sin(angles)[..., :self.D_head]
        return sin, cos

    def _init_weights(self) -> None:
        device = self.periods.device
        dtype = self.dtype
        if self.base is not None:
            periods = self.base ** (
                2 * torch.arange(self.D_head_padded // 6,
                                 device=device, dtype=dtype)
                / (self.D_head_padded // 3)
            )
        else:
            base = self.max_period / self.min_period
            exponents = torch.linspace(
                0, 1, self.D_head_padded // 6, device=device, dtype=dtype)
            periods = (base**exponents) / base
            periods = periods * self.max_period
        self.periods.data = periods


# RoPE positional embedding with no mixing of coordinates (axial) and no learnable weights
# Supports two parametrizations of the rope parameters: either using `base` or `min_period` and `max_period`.
class RopePositionEmbeddings(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int,
        base: float | None = 100.0,
        min_period: float | None = None,
        max_period: float | None = None,
        normalize_coords: Literal["min", "max", "separate"] = "separate",
        shift_coords: float | None = None,
        jitter_coords: float | None = None,
        rescale_coords: float | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        assert embed_dim % (4 * num_heads) == 0
        both_periods = min_period is not None and max_period is not None
        if (base is None and not both_periods) or (base is not None and both_periods):
            raise ValueError(
                "Either `base` or `min_period`+`max_period` must be provided.")

        D_head = embed_dim // num_heads
        self.base = base
        self.min_period = min_period
        self.max_period = max_period
        self.D_head = D_head
        self.normalize_coords = normalize_coords
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords

        # Needs persistent=True because we do teacher.load_state_dict(student.state_dict()) to initialize the teacher
        self.dtype = dtype  # Don't rely on self.periods.dtype
        self.register_buffer(
            "periods",
            torch.empty(D_head // 4, device=device, dtype=dtype),
            persistent=True,
        )
        self._init_weights()

    def forward(self, *, H: int, W: int) -> tuple[Tensor, Tensor]:
        device = self.periods.device
        dtype = self.dtype
        dd = {"device": device, "dtype": dtype}

        # Prepare coords in range [-1, +1]
        if self.normalize_coords == "max":
            max_HW = max(H, W)
            coords_h = torch.arange(0.5, H, **dd) / max_HW  # [H]
            coords_w = torch.arange(0.5, W, **dd) / max_HW  # [W]
        elif self.normalize_coords == "min":
            min_HW = min(H, W)
            coords_h = torch.arange(0.5, H, **dd) / min_HW  # [H]
            coords_w = torch.arange(0.5, W, **dd) / min_HW  # [W]
        elif self.normalize_coords == "separate":
            coords_h = torch.arange(0.5, H, **dd) / H  # [H]
            coords_w = torch.arange(0.5, W, **dd) / W  # [W]
        else:
            raise ValueError(
                f"Unknown normalize_coords: {self.normalize_coords}")
        coords = torch.stack(torch.meshgrid(
            coords_h, coords_w, indexing="ij"), dim=-1)  # [H, W, 2]
        coords = coords.flatten(0, 1)  # [HW, 2]
        coords = 2.0 * coords - 1.0  # Shift range [0, 1] to [-1, +1]

        # Shift coords by adding a uniform value in [-shift, shift]
        if self.training and self.shift_coords is not None:
            shift_hw = torch.empty(
                2, **dd).uniform_(-self.shift_coords, self.shift_coords)
            coords += shift_hw[None, :]

        # Jitter coords by multiplying the range [-1, 1] by a log-uniform value in [1/jitter, jitter]
        if self.training and self.jitter_coords is not None:
            jitter_max = np.log(self.jitter_coords)
            jitter_min = -jitter_max
            jitter_hw = torch.empty(
                2, **dd).uniform_(jitter_min, jitter_max).exp()
            coords *= jitter_hw[None, :]

        # Rescale coords by multiplying the range [-1, 1] by a log-uniform value in [1/rescale, rescale]
        if self.training and self.rescale_coords is not None:
            rescale_max = np.log(self.rescale_coords)
            rescale_min = -rescale_max
            rescale_hw = torch.empty(
                1, **dd).uniform_(rescale_min, rescale_max).exp()
            coords *= rescale_hw

        # Prepare angles and sin/cos
        angles = 2 * math.pi * coords[:, :, None] / \
            self.periods[None, None, :]  # [HW, 2, D//4]
        angles = angles.flatten(1, 2)  # [HW, D//2]
        angles = angles.tile(2)  # [HW, D]
        cos = torch.cos(angles)  # [HW, D]
        sin = torch.sin(angles)  # [HW, D]

        return (sin, cos)  # 2 * [HW, D]

    def _init_weights(self):
        device = self.periods.device
        dtype = self.dtype
        if self.base is not None:
            periods = self.base ** (
                2 * torch.arange(self.D_head // 4, device=device,
                                 dtype=dtype) / (self.D_head // 2)
            )  # [D//4]
        else:
            base = self.max_period / self.min_period
            exponents = torch.linspace(
                # [D//4] range [0, 1]
                0, 1, self.D_head // 4, device=device, dtype=dtype)
            periods = base**exponents  # range [1, max_period / min_period]
            periods = periods / base  # range [min_period / max_period, 1]
            # range [min_period, max_period]
            periods = periods * self.max_period
        self.periods.data = periods
