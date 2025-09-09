from typing import Tuple
import numpy as np
import torch
import torch.nn.functional as F
from dash import Dash, dcc, html, Input, Output, no_update
import plotly.graph_objects as go
from typing import List, Tuple
from dash import Dash, dcc, html, Input, Output, no_update, ctx
from typing import Tuple
import numpy as np
from dash import Input, Output, no_update, ctx
import plotly.graph_objects as go

from dash import Dash, dcc, html, Input, Output, no_update, ctx, Patch
import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
from typing import Tuple


def normalize_features_torch(feat: torch.Tensor) -> torch.Tensor:
    """L2-normalize feature tensor (C,H,W) per pixel along C."""
    if feat.ndim != 3:
        raise ValueError("Expected shape (C,H,W)")
    return F.normalize(feat, p=2, dim=0)


def cosine_map_fast(f_norm: np.ndarray, ref_vec: np.ndarray) -> np.ndarray:
    """Cosine sim in [0,1]; f_norm (C,H,W) and ref_vec (C,) must be L2-normalized."""
    # ((ref·f)+1)/2 with float32 to cut payload & math cost
    sim = (ref_vec.astype(np.float32)[
           :, None, None] * f_norm.astype(np.float32)).sum(0)
    return (np.clip(sim, -1.0, 1.0) + 1.0) * 0.5


def init_figure(sim: np.ndarray, title: str) -> go.Figure:
    """Static figure scaffold: heatmap + an (initially hidden) marker trace."""
    H, W = sim.shape
    fig = go.Figure([
        go.Heatmap(
            z=sim, zmin=float(sim.min()), zmax=float(sim.max()), colorscale="magma",
            colorbar=dict(title="cosine sim"),
            hovertemplate="x:%{x} y:%{y}<br>sim:%{z:.3f}<extra></extra>",
        ),
        go.Scatter(x=[None], y=[None], mode="markers",
                   marker=dict(size=10, symbol="x", line=dict(
                       width=2), color="black"),
                   hoverinfo="skip")
    ])
    fig.update_layout(
        title=title,
        xaxis=dict(range=[-0.5, W - 0.5], constrain="domain", scaleanchor="y",
                   scaleratio=1, autorange=False),
        yaxis=dict(range=[H - 0.5, -0.5], autorange=False),
        margin=dict(l=10, r=10, t=30, b=10),
        uirevision="lock",  # prevents relayout on updates
    )
    return fig


def map_coords(y: int, x: int, src_hw: Tuple[int, int], dst_hw: Tuple[int, int]) -> Tuple[int, int]:
    """Map (y,x) from src H×W to dst H×W via normalized coordinates."""
    Hs, Ws = src_hw
    Hd, Wd = dst_hw
    u = x / max(Ws - 1, 1)
    v = y / max(Hs - 1, 1)
    xd = int(round(u * (Wd - 1)))
    yd = int(round(v * (Hd - 1)))
    return yd, xd


def build_app(f1: torch.Tensor, f2: torch.Tensor) -> Dash:
    """Interactive app for two feature maps; hover on one picks reference vector,
    both images are recolored by similarity to that vector.
    """
    f1n = F.normalize(f1, p=2, dim=0).detach().cpu().numpy().astype(np.float32)
    f2n = F.normalize(f2, p=2, dim=0).detach().cpu().numpy().astype(np.float32)
    H1, W1 = f1n.shape[1:]
    H2, W2 = f2n.shape[1:]

    y0, x0 = H1 // 2, W1 // 2
    ref0 = f1n[:, y0, x0]
    sim1 = cosine_map_fast(f1n, ref0)
    sim2 = cosine_map_fast(f2n, ref0)

    app = Dash(__name__)
    app.layout = html.Div([
        html.Div([dcc.Graph(id="sim-a", figure=init_figure(sim1, "Image 1")),
                  html.Div(id="readout-a")],
                 style={"display": "inline-block", "width": "49%"}),
        html.Div([dcc.Graph(id="sim-b", figure=init_figure(sim2, "Image 2")),
                  html.Div(id="readout-b")],
                 style={"display": "inline-block", "width": "49%"}),
    ])

    @app.callback(
        Output("sim-a", "figure"),
        Output("readout-a", "children"),
        Output("sim-b", "figure"),
        Output("readout-b", "children"),
        Input("sim-a", "hoverData"),
        Input("sim-b", "hoverData"),
        prevent_initial_call=False,
    )
    def on_hover_both(hoverA, hoverB) -> Tuple[go.Figure, str, go.Figure, str]:
        """Patch-update z arrays + marker; fixed zmin/zmax/uirevision keeps it snappy."""
        trig = ctx.triggered_id

        if trig == "sim-a" and hoverA:
            x, y = int(hoverA["points"][0]["x"]), int(hoverA["points"][0]["y"])
            ref_vec = f1n[:, y, x]
            simA = cosine_map_fast(f1n, ref_vec)
            simB = cosine_map_fast(f2n, ref_vec)

            pA, pB = Patch(), Patch()
            pA["data"][0]["z"] = simA
            pA["data"][0]["zmin"] = float(simA.min())
            pA["data"][0]["zmax"] = float(simA.max())
            pA["data"][1]["x"] = [x]
            pA["data"][1]["y"] = [y]
            pB["data"][0]["z"] = simB
            pB["data"][0]["zmin"] = float(simB.min())
            pB["data"][0]["zmax"] = float(simB.max())
            pB["data"][1]["x"] = [None]
            pB["data"][1]["y"] = [None]
            return pA, f"ref: img1 (x={x}, y={y})", pB, "updated by img1"

        if trig == "sim-b" and hoverB:
            x, y = int(hoverB["points"][0]["x"]), int(hoverB["points"][0]["y"])
            ref_vec = f2n[:, y, x]
            simB = cosine_map_fast(f2n, ref_vec)
            simA = cosine_map_fast(f1n, ref_vec)

            pA, pB = Patch(), Patch()
            pA["data"][0]["z"] = simA
            pA["data"][0]["zmin"] = float(simA.min())
            pA["data"][0]["zmax"] = float(simA.max())
            pA["data"][1]["x"] = [None]
            pA["data"][1]["y"] = [None]
            pB["data"][0]["z"] = simB
            pB["data"][0]["zmin"] = float(simB.min())
            pB["data"][0]["zmax"] = float(simB.max())
            pB["data"][1]["x"] = [x]
            pB["data"][1]["y"] = [y]
            return pA, "", pB, "ref: img2"

        return no_update, no_update, no_update, no_update

    return app
