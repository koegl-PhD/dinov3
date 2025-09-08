from typing import Tuple
import numpy as np
import torch
import torch.nn.functional as F
from dash import Dash, dcc, html, Input, Output, no_update
import plotly.graph_objects as go


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


def make_figure(sim: np.ndarray, x: int, y: int) -> go.Figure:
    """Create similarity heatmap with the selected pixel highlighted."""
    H, W = sim.shape
    heat = go.Heatmap(
        z=sim,
        zmin=sim.min(), zmax=sim.max(),
        colorscale="magma",
        colorbar=dict(title="cosine sim"),
        hovertemplate="x:%{x} y:%{y}<br>sim:%{z:.3f}<extra></extra>",
    )
    marker = go.Scatter(
        x=[x], y=[y],
        mode="markers",
        marker=dict(size=10, symbol="x", line=dict(width=2), color="black"),
        hoverinfo="skip",
    )
    fig = go.Figure(data=[heat, marker])
    fig.update_layout(
        xaxis=dict(title="x", range=[-0.5, W - 0.5], constrain="domain",
                   scaleanchor="y", scaleratio=1, autorange=False),
        yaxis=dict(title="y", range=[H - 0.5, -0.5], autorange=False),
        margin=dict(l=10, r=10, t=30, b=10),
        title="Feature cosine similarity",
    )
    return fig


def build_app(feat: torch.Tensor) -> Dash:
    """Build a Dash app for interactive cosine-similarity exploration."""
    f_norm_t = normalize_features_torch(feat)
    f_norm = f_norm_t.detach().cpu().numpy()   # (C,H,W)
    _, H, W = f_norm.shape

    y0, x0 = H // 2, W // 2
    sim0 = cosine_map_np(f_norm, y0, x0)

    app = Dash(__name__)
    app.layout = html.Div(
        [
            dcc.Graph(id="sim-graph", figure=make_figure(sim0,
                      x0, y0), clear_on_unhover=False),
            html.Div(id="xy-readout"),
        ],
        style={"width": "800px"},
    )

    @app.callback(
        Output("sim-graph", "figure"),
        Output("xy-readout", "children"),
        Input("sim-graph", "hoverData"),
        prevent_initial_call=False,
    )
    def on_hover(hoverData) -> Tuple[go.Figure, str]:
        """Update similarity map on hover."""
        if not hoverData:
            return make_figure(sim0, x0, y0), f"(x={x0}, y={y0})"
        pt = hoverData.get("points", [{}])[0]
        x = int(pt.get("x", x0))
        y = int(pt.get("y", y0))
        if not (0 <= x < W and 0 <= y < H):
            return no_update, no_update
        sim = cosine_map_np(f_norm, y, x)
        return make_figure(sim, x, y), f"(x={x}, y={y})"

    return app
