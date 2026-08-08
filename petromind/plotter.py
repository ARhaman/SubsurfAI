"""
PetroMind Plotter
Interactive Plotly log tracks with uncertainty ribbons.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .knowledge_base import LITHOLOGY_COLORS


TRACK_CONFIG = [
    {"col": "GR",      "title": "GR (API)",      "color": "#378ADD", "xmin": 0,    "xmax": 150},
    {"col": "NPHI",    "title": "NPHI (v/v)",    "color": "#1D9E75", "xmin": 0.45, "xmax": -0.05},
    {"col": "RHOB",    "title": "RHOB (g/cc)",   "color": "#D85A30", "xmin": 1.95, "xmax": 2.95},
    {"col": "RT",      "title": "RT (Ω·m)",      "color": "#7F77DD", "xmin": 0.2,  "xmax": 2000, "log": True},
    {"col": "PHI_EFF", "title": "φ_eff (v/v)",   "color": "#0F6E56", "xmin": 0.45, "xmax": 0},
    {"col": "SW",      "title": "Sw (v/v)",       "color": "#185FA5", "xmin": 1,    "xmax": 0},
]


def build_log_figure(df: pd.DataFrame, depth_range=None) -> go.Figure:
    """
    Build multi-track Plotly log figure.
    Always shows: GR | NPHI+RHOB | RT | PHI_EFF+Unc | SW | Lithology
    """
    if depth_range:
        df = df[(df["DEPTH"] >= depth_range[0]) & (df["DEPTH"] <= depth_range[1])]

    avail = [t for t in TRACK_CONFIG if t["col"] in df.columns]
    n_tracks = len(avail) + 2  # +litho +fluid

    fig = make_subplots(
        rows=1, cols=n_tracks,
        shared_yaxes=True,
        column_widths=[1]*n_tracks,
        horizontal_spacing=0.01,
    )

    depth = df["DEPTH"].values

    for i, trk in enumerate(avail, start=1):
        col = trk["col"]
        vals = df[col].values
        xtype = "log" if trk.get("log") else "linear"

        fig.add_trace(go.Scatter(
            x=vals, y=depth,
            mode="lines",
            line=dict(color=trk["color"], width=1),
            name=trk["title"],
            showlegend=False,
        ), row=1, col=i)

        # Uncertainty ribbon for porosity
        if col == "PHI_EFF" and "UNC_PHI" in df:
            unc = df["UNC_PHI"].values
            fig.add_trace(go.Scatter(
                x=np.concatenate([vals+unc, (vals-unc)[::-1]]),
                y=np.concatenate([depth, depth[::-1]]),
                fill="toself", fillcolor="rgba(15,110,86,0.15)",
                line=dict(width=0), showlegend=False, name="φ uncertainty",
            ), row=1, col=i)

        if col == "SW" and "UNC_SW" in df:
            unc = df["UNC_SW"].values
            fig.add_trace(go.Scatter(
                x=np.concatenate([vals+unc, (vals-unc)[::-1]]),
                y=np.concatenate([depth, depth[::-1]]),
                fill="toself", fillcolor="rgba(24,95,165,0.15)",
                line=dict(width=0), showlegend=False, name="Sw uncertainty",
            ), row=1, col=i)

        fig.update_xaxes(
            title_text=trk["title"], row=1, col=i,
            range=[trk["xmin"], trk["xmax"]] if not trk.get("log") else None,
            type=xtype, side="top", showgrid=True, gridcolor="#E8E8E8",
        )

    # Lithology track
    lith_col = len(avail) + 1
    if "LITHOLOGY" in df.columns:
        prev_lith, prev_d = df["LITHOLOGY"].iloc[0], depth[0]
        for lith, d in zip(df["LITHOLOGY"].values[1:], depth[1:]):
            if lith != prev_lith:
                fig.add_shape(
                    type="rect", xref=f"x{lith_col}", yref="y",
                    x0=0, x1=1, y0=prev_d, y1=d,
                    fillcolor=LITHOLOGY_COLORS.get(prev_lith, "#DDD"),
                    line=dict(width=0), layer="below",
                    row=1, col=lith_col,
                )
                prev_lith, prev_d = lith, d
        fig.add_shape(
            type="rect", xref=f"x{lith_col}", yref="y",
            x0=0, x1=1, y0=prev_d, y1=depth[-1],
            fillcolor=LITHOLOGY_COLORS.get(prev_lith, "#DDD"),
            line=dict(width=0), layer="below",
            row=1, col=lith_col,
        )
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
            marker=dict(size=0), showlegend=False), row=1, col=lith_col)
        fig.update_xaxes(title_text="Lithology", row=1, col=lith_col,
            range=[0,1], showticklabels=False, side="top")

    # Fluid flag track
    fluid_col = len(avail) + 2
    FLUID_COLORS = {
        "hydrocarbon": "#22C55E",
        "transition":  "#F59E0B",
        "water":       "#3B82F6",
        "tight":       "#94A3B8",
        "shale":       "#9CA3AF",
        "undetermined":"#D1D5DB",
    }
    if "FLUID_FLAG" in df.columns:
        prev_f, prev_d = df["FLUID_FLAG"].iloc[0], depth[0]
        for f, d in zip(df["FLUID_FLAG"].values[1:], depth[1:]):
            if f != prev_f:
                fig.add_shape(
                    type="rect", xref=f"x{fluid_col}", yref="y",
                    x0=0, x1=1, y0=prev_d, y1=d,
                    fillcolor=FLUID_COLORS.get(prev_f, "#DDD"),
                    line=dict(width=0), layer="below",
                    row=1, col=fluid_col,
                )
                prev_f, prev_d = f, d
        fig.add_shape(
            type="rect", xref=f"x{fluid_col}", yref="y",
            x0=0, x1=1, y0=prev_d, y1=depth[-1],
            fillcolor=FLUID_COLORS.get(prev_f, "#DDD"),
            line=dict(width=0), layer="below",
            row=1, col=fluid_col,
        )
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
            marker=dict(size=0), showlegend=False), row=1, col=fluid_col)
        fig.update_xaxes(title_text="Fluid", row=1, col=fluid_col,
            range=[0,1], showticklabels=False, side="top")

    fig.update_yaxes(
        autorange="reversed", title_text="Depth (m)",
        showgrid=True, gridcolor="#E8E8E8",
        row=1, col=1,
    )

    fig.update_layout(
        height=700,
        margin=dict(l=60, r=20, t=60, b=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=11),
        hovermode="y unified",
    )

    return fig


def build_crossplot(df: pd.DataFrame, x_col: str, y_col: str, color_col: str = "LITHOLOGY") -> go.Figure:
    """Scatter crossplot coloured by lithology.
    Auto-switches to density contour when > 2000 points to fix overplotting.
    """
    sub = df[[x_col, y_col, color_col]].dropna()
    liths = sub[color_col].unique()
    n_pts = len(sub)

    fig = go.Figure()

    if n_pts > 2000:
        # ── Density mode: one contour layer per lithology ────────────────────
        for lith in liths:
            mask = sub[color_col] == lith
            lith_lower = str(lith).lower()
            col = LITHOLOGY_COLORS.get(lith_lower, "#9CA3AF")
            s = sub.loc[mask]
            if len(s) < 10:
                continue
            fig.add_trace(go.Histogram2dContour(
                x=s[x_col], y=s[y_col],
                name=lith.capitalize(),
                colorscale=[[0, "rgba(0,0,0,0)"], [1, col]],
                showscale=False,
                ncontours=8,
                contours=dict(coloring="fill", showlines=True),
                opacity=0.75,
                hovertemplate=(
                    f"<b>{lith}</b><br>"
                    f"{x_col}: %{{x:.3f}}<br>"
                    f"{y_col}: %{{y:.3f}}<extra></extra>"
                ),
            ))
            # Sparse scatter overlay so legend markers show colour
            sample = s.sample(min(120, len(s)), random_state=0)
            fig.add_trace(go.Scatter(
                x=sample[x_col], y=sample[y_col],
                mode="markers",
                name=lith.capitalize(),
                showlegend=False,
                marker=dict(color=col, size=3, opacity=0.4),
            ))
        fig.add_annotation(
            text=f"⚠ Density view — {n_pts:,} pts (scatter auto-switched at >2 000)",
            xref="paper", yref="paper", x=0.01, y=0.99,
            showarrow=False, font=dict(size=10, color="#94A3B8"),
            align="left",
        )
    else:
        # ── Scatter mode (≤ 2000 pts) ─────────────────────────────────────────
        for lith in liths:
            mask = sub[color_col] == lith
            lith_lower = str(lith).lower()
            fig.add_trace(go.Scatter(
                x=sub.loc[mask, x_col],
                y=sub.loc[mask, y_col],
                mode="markers",
                name=lith.capitalize(),
                marker=dict(color=LITHOLOGY_COLORS.get(lith_lower, "#9CA3AF"), size=5, opacity=0.75),
            ))

    fig.update_layout(
        xaxis_title=x_col, yaxis_title=y_col,
        height=400, margin=dict(l=50, r=20, t=30, b=40),
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(itemsizing="constant"),
    )
    if y_col in ["NPHI", "PHI_EFF", "SW"]:
        fig.update_yaxes(autorange="reversed")

    return fig
