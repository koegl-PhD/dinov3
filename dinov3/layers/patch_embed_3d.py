import math
from typing import Callable

from torch import Tensor, nn


def make_3tuple(x: int | tuple[int, int, int]) -> tuple[int, int, int]:
    """Return (x,x,x) if int, else verify 3-tuple."""
    if isinstance(x, tuple):
        assert len(x) == 3
        return x
    assert isinstance(x, int)
    return (x, x, x)


class PatchEmbed3D(nn.Module):
    """
    3D volume to patch embedding: (B,H,W,D) -> (B,N,embed_dim) or (B,H',W',D',embed_dim).
    """

    def __init__(
        self,
        vol_size: int | tuple[int, int, int] = 224,
        patch_size: int | tuple[int, int, int] = 16,
        embed_dim: int = 768,
        norm_layer: Callable | None = None,
        flatten_embedding: bool = True,
    ) -> None:
        super().__init__()
        VHW = make_3tuple(vol_size)
        P = make_3tuple(patch_size)
        grid = (VHW[0] // P[0], VHW[1] // P[1], VHW[2] // P[2])

        self.vol_size = VHW
        self.patch_size = P
        self.patches_resolution = grid
        self.num_patches = grid[0] * grid[1] * grid[2]

        self.embed_dim = embed_dim
        self.flatten_embedding = flatten_embedding

        self.proj = nn.Conv3d(1, embed_dim, kernel_size=P, stride=P)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """x: (B,H,W,D) -> tokens."""
        x = x.unsqueeze(1)
        x = self.proj(x)
        H, W, D = x.size(2), x.size(3), x.size(4)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, H, W, D, self.embed_dim)
        return x

    def flops(self) -> float:
        """Approximate FLOPs."""
        Ho, Wo, Do = self.patches_resolution
        k = self.patch_size[0] * self.patch_size[1] * self.patch_size[2]
        flops = Ho * Wo * Do * self.embed_dim * 1 * k
        flops += Ho * Wo * Do * self.embed_dim
        return flops

    def reset_parameters(self) -> None:
        """Uniform init like 2D variant."""
        k = 1 / (1 * (self.patch_size[0] *
                 self.patch_size[1] * self.patch_size[2]))
        nn.init.uniform_(self.proj.weight, -math.sqrt(k), math.sqrt(k))
        if self.proj.bias is not None:
            nn.init.uniform_(self.proj.bias, -math.sqrt(k), math.sqrt(k))
