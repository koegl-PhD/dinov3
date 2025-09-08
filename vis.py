from typing import Tuple
import numpy as np
import torch
import torch.nn.functional as F
from dash import Dash, dcc, html, Input, Output, no_update
import plotly.graph_objects as go
from typing import List, Tuple
from dash import Dash, dcc, html, Input, Output, no_update, ctx


def normalize_features_torch(feat: torch.Tensor) -> torch.Tensor:
    """L2-normalize feature tensor (C,H,W) per pixel along C."""
    if feat.ndim != 3:
        raise ValueError("Expected shape (C,H,W)")
    return F.normalize(feat, p=2, dim=0)


def cosine_map_np(f_norm: np.ndarray, y: int, x: int) -> np.ndarray:
    """Compute HxW cosine-similarity in [0,1] to pixel (y,x) from normalized features (C,H,W)."""
    C, H, W = f_norm.shape
    if not (0 <= y < H and 0 <= x < W):
        raise ValueError("y,x out of bounds")
    v = f_norm[:, y, x]                        # (C,)
    sim = np.tensordot(v, f_norm, axes=(0, 0))  # (H,W) in [-1,1]
    return (np.clip(sim, -1.0, 1.0) + 1.0) * 0.5


def map_coords(y: int, x: int, src_hw: Tuple[int, int], dst_hw: Tuple[int, int]) -> Tuple[int, int]:
    """Map (y,x) from src H×W to dst H×W via normalized coordinates."""
    Hs, Ws = src_hw
    Hd, Wd = dst_hw
    u = x / max(Ws - 1, 1)
    v = y / max(Hs - 1, 1)
    xd = int(round(u * (Wd - 1)))
    yd = int(round(v * (Hd - 1)))
    return yd, xd


def make_figure(sim: np.ndarray, x: int, y: int, total_min: float, total_max: float) -> go.Figure:
    """Create similarity heatmap with the selected pixel highlighted."""
    H, W = sim.shape
    heat = go.Heatmap(
        z=sim, zmin=total_min, zmax=total_max,
        colorscale="magma",  # magma-like as in your screenshot
        colorbar=dict(title="cosine sim"),
        hovertemplate="x:%{x} y:%{y}<br>sim:%{z:.3f}<extra></extra>",
    )

    fig = go.Figure(data=[heat])
    fig.update_layout(
        xaxis=dict(range=[-0.5, W - 0.5], constrain="domain",
                   scaleanchor="y", scaleratio=1, autorange=False),
        yaxis=dict(range=[H - 0.5, -0.5], autorange=False),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    return fig


def build_app(f1: torch.Tensor, f2: torch.Tensor) -> Dash:
    """Interactive app for two feature maps; hover on one picks reference vector,
    both images are recolored by similarity to that vector.
    """
    f1n = normalize_features_torch(f1).detach().cpu().numpy()
    f2n = normalize_features_torch(f2).detach().cpu().numpy()
    H1, W1 = f1n.shape[1:]
    H2, W2 = f2n.shape[1:]

    # initial similarity (center of image 1)
    y0, x0 = H1 // 2, W1 // 2
    ref_vec = f1n[:, y0, x0]
    sim1 = cosine_map_np(f1n, y0, x0)
    sim2 = (np.tensordot(ref_vec, f2n, axes=(0, 0)) + 1) * 0.5

    total_min = np.minimum(sim1.min(), sim2.min())
    total_max = np.maximum(sim1.max(), sim2.max())

    app = Dash(__name__)
    app.layout = html.Div(
        [
            html.Div(
                [dcc.Graph(id="sim-a", figure=make_figure(sim1,
                           x0, y0, total_min, total_max)), html.Div(id="readout-a")],
                style={"display": "inline-block", "width": "49%"},
            ),
            html.Div(
                [dcc.Graph(id="sim-b", figure=make_figure(sim2, W2 //
                           2, H2 // 2, total_min, total_max)), html.Div(id="readout-b")],
                style={"display": "inline-block", "width": "49%"},
            ),
        ]
    )

    @app.callback(
        Output("sim-a", "figure"),
        Output("readout-a", "children"),
        Output("sim-b", "figure"),
        Output("readout-b", "children"),
        Input("sim-a", "hoverData"),
        Input("sim-b", "hoverData"),
        prevent_initial_call=False,
    )
    def on_hover(hoverA, hoverB):
        nonlocal f1n, f2n
        H1, W1 = f1n.shape[1:]
        H2, W2 = f2n.shape[1:]

        if hoverA:  # pick ref from image 1
            pt = hoverA["points"][0]
            x, y = int(pt["x"]), int(pt["y"])
            ref_vec = f1n[:, y, x]
            sim1 = cosine_map_np(f1n, y, x)
            sim2 = (np.tensordot(ref_vec, f2n, axes=(0, 0)) + 1) * 0.5

            total_min = np.minimum(sim1.min(), sim2.min())
            total_max = np.maximum(sim1.max(), sim2.max())

            return (make_figure(sim1, x, y, total_min, total_max),
                    f"ref from img1 (x={x},y={y})",
                    make_figure(sim2, W2 // 2, H2 // 2, total_min, total_max),
                    "updated by img1")

        if hoverB:  # pick ref from image 2
            pt = hoverB["points"][0]
            x, y = int(pt["x"]), int(pt["y"])
            ref_vec = f2n[:, y, x]
            sim2 = cosine_map_np(f2n, y, x)
            sim1 = (np.tensordot(ref_vec, f1n, axes=(0, 0)) + 1) * 0.5

            total_min = np.minimum(sim1.min(), sim2.min())
            total_max = np.maximum(sim1.max(), sim2.max())
            return (make_figure(sim1, W1 // 2, H1 // 2, total_min, total_max), "updated by img2",
                    make_figure(sim2, x, y, total_min, total_max), f"ref from img2 (x={x},y={y})")

        return no_update, no_update, no_update, no_update

    return app
