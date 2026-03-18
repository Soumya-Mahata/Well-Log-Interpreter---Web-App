"""
plots.py  —  Interactive Plotly Visualization Library  (v3.1)
==============================================================
Every function returns a go.Figure ready for st.plotly_chart().

Color / visibility fixes applied throughout:
  • All text (titles, axis labels, tick labels, legends, colorbars,
    annotations) uses dark colors (#212121 or named dark hex).
  • Colorbar: explicit dark tickfont + title font, white bg strip.
  • Legend: dark font, white semi-opaque background, visible border.
  • Subplot titles: patched to dark after make_subplots (Plotly default is grey).
  • Mineral/annotation labels: opaque white bg box + dark border so they
    are readable over any scatter density.
  • Colorscale changed from RdYlGn_r (mid-green invisible on white bg)
    to Viridis_r which has no invisible midpoint.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from utils import MINERALS

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLE TOKENS
# ─────────────────────────────────────────────────────────────────────────────

_TRACK_COLORS = [
    "#1565C0",  # strong blue
    "#C62828",  # strong red
    "#2E7D32",  # dark green
    "#E65100",  # burnt orange
    "#6A1B9A",  # deep purple
    "#00695C",  # dark teal
    "#4E342E",  # dark brown
    "#37474F",  # dark blue-grey
    "#AD1457",  # deep pink
    "#F57F17",  # amber
]

_MIN_COLORS = {
    "Sandstone": "#B8860B",
    "Limestone": "#1565C0",
    "Dolomite":  "#1B5E20",
    "Anhydrite": "#4A148C",
    "Halite":    "#B71C1C",
    "Gypsum":    "#4E342E",
    "Salt":      "#B71C1C",
    "Illite":    "#37474F",
}

_GRID = dict(
    showgrid=True, gridcolor="#d4d4d4", gridwidth=1,
    zeroline=True,  zerolinecolor="#bbbbbb", zerolinewidth=1,
)
_DEPTH_AX = dict(
    autorange="reversed",
    title_text="Depth",
    title_font=dict(size=12, color="#212121"),
    tickfont=dict(color="#212121", size=10),
    **_GRID,
)
_MARG = dict(l=60, r=30, t=65, b=55)

_CBAR = dict(
    tickfont=dict(color="#212121", size=11),
    title_font=dict(color="#212121", size=12),
    bgcolor="white",
    bordercolor="#aaaaaa",
    borderwidth=1,
    outlinecolor="#aaaaaa",
    outlinewidth=1,
    thickness=14,
    len=0.75,
)

_LEGEND = dict(
    orientation="h",
    yanchor="bottom", y=-0.25,
    xanchor="center", x=0.5,
    bgcolor="rgba(255,255,255,0.95)",
    bordercolor="#888888",
    borderwidth=1,
    font=dict(color="#212121", size=11),
)


def _base_layout(title="", height=700):
    return dict(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=15, color="#0D2B5E"),
            x=0, xanchor="left",
        ),
        height=height,
        margin=_MARG,
        legend=_LEGEND,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="#212121"),
    )


def _xax(title, log=False, **kw):
    return dict(
        title_text=title,
        title_font=dict(size=11, color="#212121"),
        tickfont=dict(color="#212121", size=10),
        type="log" if log else "linear",
        **_GRID,
        **kw,
    )


def _label_ann(text, x, y, color="#212121"):
    """Annotation label with opaque white background box."""
    return dict(
        x=x, y=y, text=f"<b>{text}</b>",
        showarrow=False,
        font=dict(color=color, size=10),
        bgcolor="rgba(255,255,255,0.90)",
        bordercolor=color,
        borderwidth=1,
        borderpad=3,
    )


def _fix_cb(fig, label=""):
    """Apply dark-text colorbar style to first coloraxis."""
    cb = dict(**_CBAR)
    if label:
        cb["title"] = dict(text=label, font=dict(color="#212121", size=12))
    fig.update_coloraxes(colorbar=cb)
    return fig


def _fix_subtitles(fig):
    """Force all make_subplots annotation titles to dark text."""
    for ann in fig.layout.annotations:
        ann.font = dict(color="#212121", size=11, family="sans-serif")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-TRACK RAW LOG
# ─────────────────────────────────────────────────────────────────────────────

def plot_raw_logs(df, curves, depth_col="DEPTH", log_scale_curves=None):
    log_scale_curves = log_scale_curves or []
    n = len(curves)
    if n == 0:
        return go.Figure()

    fig = make_subplots(
        rows=1, cols=n, shared_yaxes=True, subplot_titles=curves,
        horizontal_spacing=max(0.01, min(0.04, 0.25 / n)),
    )
    _fix_subtitles(fig)
    depth = df[depth_col]
    for i, curve in enumerate(curves, 1):
        if curve not in df.columns:
            continue
        fig.add_trace(
            go.Scatter(x=df[curve], y=depth, mode="lines", name=curve,
                       line=dict(color=_TRACK_COLORS[(i-1) % len(_TRACK_COLORS)], width=1.3)),
            row=1, col=i,
        )
        fig.update_xaxes(**_xax(curve, log=(curve in log_scale_curves)), row=1, col=i)

    fig.update_yaxes(**_DEPTH_AX, row=1, col=1)
    fig.update_layout(**_base_layout(f"Well Log Tracks  ({n} curves)"), showlegend=False)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# QC PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_before_after(df_raw, df_qc, curve, depth_col="DEPTH", log_scale=False):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_raw[curve], y=df_raw[depth_col], mode="lines",
        name="Before QC", line=dict(color="#C62828", width=1.3, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=df_qc[curve], y=df_qc[depth_col], mode="lines",
        name="After QC", line=dict(color="#1565C0", width=2.0),
    ))
    fig.update_xaxes(**_xax(curve, log=log_scale))
    fig.update_yaxes(**_DEPTH_AX)
    fig.update_layout(**_base_layout(f"QC Before / After — {curve}"))
    return fig


def plot_outlier_flags(df, curve, mask, depth_col="DEPTH", log_scale=False):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df[curve], y=df[depth_col], mode="lines",
        name=curve, line=dict(color="#1565C0", width=1.3),
    ))
    out = df[mask]
    if not out.empty:
        fig.add_trace(go.Scatter(
            x=out[curve], y=out[depth_col], mode="markers", name="Outlier",
            marker=dict(color="#C62828", size=8, symbol="x",
                        line=dict(width=2.5, color="#C62828")),
        ))
    fig.update_xaxes(**_xax(curve, log=log_scale))
    fig.update_yaxes(**_DEPTH_AX)
    fig.update_layout(**_base_layout(f"Outlier Detection — {curve}"))
    return fig


def plot_hole_quality(df, cali_col, bit_size, hq, depth_col="DEPTH"):
    STATUS_COLOR = {
        "Washout":  "#C62828",
        "In-Gauge": "#1B5E20",
        "Mudcake":  "#1565C0",
        "No CALI":  "#757575",
    }
    fig    = go.Figure()
    depth  = df[depth_col].values
    cali   = df[cali_col].values
    hq_arr = hq.values

    for status, color in STATUS_COLOR.items():
        sel = hq_arr == status
        if not sel.any():
            continue
        fig.add_trace(go.Scatter(
            x=cali[sel], y=depth[sel], mode="markers", name=status,
            marker=dict(color=color, size=5, opacity=0.80, symbol="square"),
        ))

    fig.add_trace(go.Scatter(
        x=cali, y=depth, mode="lines", name="Caliper",
        line=dict(color="#212121", width=2.0),
    ))

    x_lo = min(float(np.nanmin(cali)), bit_size) * 0.90
    x_hi = max(float(np.nanmax(cali)), bit_size * 1.15) * 1.05

    fig.add_vline(
        x=bit_size,
        line=dict(color="#E65100", width=2, dash="dash"),
        annotation_text=f'<b>Bit {bit_size}"</b>',
        annotation_font=dict(color="#E65100", size=11),
        annotation_bgcolor="rgba(255,255,255,0.88)",
        annotation_bordercolor="#E65100",
        annotation_position="top left",
    )
    fig.add_vline(
        x=bit_size * 1.10,
        line=dict(color="#C62828", width=1.5, dash="dot"),
        annotation_text="<b>Washout limit</b>",
        annotation_font=dict(color="#C62828", size=10),
        annotation_bgcolor="rgba(255,255,255,0.88)",
        annotation_bordercolor="#C62828",
        annotation_position="top right",
    )
    fig.update_xaxes(**_xax(f"Caliper ({cali_col})"), range=[x_lo, x_hi])
    fig.update_yaxes(**_DEPTH_AX)
    fig.update_layout(**_base_layout("Hole Quality — Caliper vs Bit Size"))
    fig.update_layout(plot_bgcolor="#FAFAFA")   # slight off-white bg for caliper track
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CROSSPLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_nphi_rhob(df, nphi_col, rhob_col, color_col=None, show_lines=True):
    nphi = df[nphi_col].copy()
    if nphi.dropna().median() > 1.0:
        nphi = nphi / 100.0
    pdf = df[[rhob_col]].copy()
    pdf["NPHI_frac"] = nphi.values

    if color_col and color_col in df.columns:
        pdf[color_col] = df[color_col].values
        fig = px.scatter(pdf, x="NPHI_frac", y=rhob_col, color=color_col,
                         color_continuous_scale="Viridis_r", opacity=0.60,
                         labels={color_col: color_col,
                                 "NPHI_frac": "NPHI (v/v)", rhob_col: "RHOB (g/cc)"})
        _fix_cb(fig, color_col)
    else:
        fig = px.scatter(pdf, x="NPHI_frac", y=rhob_col, opacity=0.50,
                         color_discrete_sequence=["#1565C0"])

    if show_lines:
        ends = {"Sandstone": [(0.00,2.65),(0.40,1.90)],
                "Limestone": [(0.00,2.71),(0.40,1.88)],
                "Dolomite":  [(0.02,2.87),(0.40,1.85)]}
        ann_y = {"Sandstone": 1.90, "Limestone": 1.88, "Dolomite": 1.85}
        for name, pts in ends.items():
            clr = _MIN_COLORS.get(name, "#212121")
            fig.add_trace(go.Scatter(x=[p[0] for p in pts], y=[p[1] for p in pts],
                                     mode="lines", name=name,
                                     line=dict(color=clr, width=2.2, dash="dash")))
            fig.add_annotation(**_label_ann(name, x=0.41, y=ann_y[name], color=clr))
        fig.add_shape(type="rect", x0=0.30,x1=0.40,y0=2.35,y1=2.50,
                      line=dict(color="#5D4037",dash="dot",width=1.5),
                      fillcolor="rgba(93,64,55,0.07)")
        fig.add_annotation(**_label_ann("Shale Zone", x=0.35, y=2.425, color="#5D4037"))

    fig.update_xaxes(**_xax("NPHI (v/v)", range=[-0.05,0.60]))
    fig.update_yaxes(**_xax("RHOB (g/cc)"), autorange=False, range=[3.05,1.80])
    fig.update_layout(**_base_layout("Neutron–Density Crossplot", height=560))
    return fig


def plot_nphi_dt(df, nphi_col, dt_col, color_col=None, show_lines=True):
    nphi = df[nphi_col].copy()
    if nphi.dropna().median() > 1.0:
        nphi = nphi / 100.0
    pdf = df[[dt_col]].copy()
    pdf["NPHI_frac"] = nphi.values

    if color_col and color_col in df.columns:
        pdf[color_col] = df[color_col].values
        fig = px.scatter(pdf, x="NPHI_frac", y=dt_col, color=color_col,
                         color_continuous_scale="Viridis_r", opacity=0.60,
                         labels={color_col: color_col,
                                 "NPHI_frac": "NPHI (v/v)", dt_col: "DT (µs/ft)"})
        _fix_cb(fig, color_col)
    else:
        fig = px.scatter(pdf, x="NPHI_frac", y=dt_col, opacity=0.50,
                         color_discrete_sequence=["#2E7D32"])

    if show_lines:
        ends = {"Sandstone": [(0.00,55.5),(0.30,88.5)],
                "Limestone": [(0.00,47.5),(0.30,89.5)],
                "Dolomite":  [(0.02,43.5),(0.30,86.0)]}
        ann_y = {"Sandstone": 91, "Limestone": 93, "Dolomite": 88}
        for name, pts in ends.items():
            clr = _MIN_COLORS.get(name, "#212121")
            fig.add_trace(go.Scatter(x=[p[0] for p in pts], y=[p[1] for p in pts],
                                     mode="lines", name=name,
                                     line=dict(color=clr, width=2.2, dash="dash")))
            fig.add_annotation(**_label_ann(name, x=0.31, y=ann_y[name], color=clr))
        fig.add_shape(type="rect",x0=0.30,x1=0.40,y0=70,y1=100,
                      line=dict(color="#5D4037",dash="dot",width=1.5),
                      fillcolor="rgba(93,64,55,0.07)")
        fig.add_annotation(**_label_ann("Shale Zone", x=0.35, y=85, color="#5D4037"))

    fig.update_xaxes(**_xax("NPHI (v/v)", range=[-0.02,0.46]))
    fig.update_yaxes(**_xax("DT (µs/ft)",  range=[40,115]))
    fig.update_layout(**_base_layout("Neutron–Sonic Crossplot", height=560))
    return fig


def plot_rhob_dt(df, rhob_col, dt_col, color_col=None, show_lines=True):
    if color_col and color_col in df.columns:
        pdf = df[[dt_col, rhob_col, color_col]].copy()
        fig = px.scatter(pdf, x=dt_col, y=rhob_col, color=color_col,
                         color_continuous_scale="Viridis_r", opacity=0.60,
                         labels={color_col: color_col,
                                 dt_col: "DT (µs/ft)", rhob_col: "RHOB (g/cc)"})
        _fix_cb(fig, color_col)
    else:
        fig = px.scatter(df[[dt_col, rhob_col]], x=dt_col, y=rhob_col, opacity=0.50,
                         color_discrete_sequence=["#6A1B9A"])

    if show_lines:
        rho_f, dt_f, phi = 1.0, 189.0, 0.35
        for name, (rho_ma, dt_ma) in [("Sandstone",(2.65,55.5)),
                                       ("Limestone", (2.71,47.5)),
                                       ("Dolomite",  (2.87,43.5))]:
            clr = _MIN_COLORS.get(name, "#212121")
            fig.add_trace(go.Scatter(
                x=[dt_ma, dt_ma+phi*(dt_f-dt_ma)],
                y=[rho_ma, rho_ma+phi*(rho_f-rho_ma)],
                mode="lines", name=name,
                line=dict(color=clr, width=2.2, dash="dash")))
            fig.add_annotation(**_label_ann(name, x=dt_ma-2, y=rho_ma, color=clr))

    fig.update_xaxes(**_xax("DT (µs/ft)",  range=[40,125]))
    fig.update_yaxes(**_xax("RHOB (g/cc)"), autorange=False, range=[3.05,1.80])
    fig.update_layout(**_base_layout("Density–Sonic Crossplot", height=560))
    return fig


def plot_mn(df, M_col, N_col, color_col=None):
    pdf = df[[N_col, M_col]].dropna()
    if color_col and color_col in df.columns:
        pdf[color_col] = df.loc[pdf.index, color_col]
        fig = px.scatter(pdf, x=N_col, y=M_col, color=color_col,
                         color_continuous_scale="Viridis_r", opacity=0.60,
                         labels={color_col: color_col})
        _fix_cb(fig, color_col)
    else:
        fig = px.scatter(pdf, x=N_col, y=M_col, opacity=0.50,
                         color_discrete_sequence=["#37474F"])

    mn_refs = {"Sandstone":(0.64,0.84),"Limestone":(0.60,0.85),"Dolomite":(0.53,0.80),
               "Anhydrite":(0.52,0.72),"Halite":(1.02,1.24),"Gypsum":(0.31,1.05)}
    for name,(n_v,m_v) in mn_refs.items():
        clr = _MIN_COLORS.get(name, "#212121")
        fig.add_trace(go.Scatter(x=[n_v],y=[m_v],mode="markers",name=name,
                                 marker=dict(symbol="diamond",size=14,color=clr,
                                             line=dict(color="#212121",width=1.5))))
        fig.add_annotation(**_label_ann(name, x=n_v+0.02, y=m_v, color=clr))

    fig.add_shape(type="rect",x0=0.30,x1=0.56,y0=0.48,y1=0.72,
                  line=dict(color="#5D4037",dash="dot",width=1.5),
                  fillcolor="rgba(93,64,55,0.06)")
    fig.add_annotation(**_label_ann("Approx. Shale", x=0.43, y=0.60, color="#5D4037"))
    fig.update_xaxes(**_xax("N", range=[0.25,1.15]))
    fig.update_yaxes(**_xax("M", range=[0.42,1.30]))
    fig.update_layout(**_base_layout("M-N Plot — Lithology Indicator", height=560))
    return fig


def plot_mid_dt_rho(df, dt_maa_col, rho_maa_col, color_col=None):
    pdf = df[[dt_maa_col, rho_maa_col]].dropna()
    if color_col and color_col in df.columns:
        pdf[color_col] = df.loc[pdf.index, color_col]
        fig = px.scatter(pdf, x=dt_maa_col, y=rho_maa_col, color=color_col,
                         color_continuous_scale="Viridis_r", opacity=0.60,
                         labels={color_col: color_col})
        _fix_cb(fig, color_col)
    else:
        fig = px.scatter(pdf, x=dt_maa_col, y=rho_maa_col, opacity=0.50,
                         color_discrete_sequence=["#37474F"])

    mineral_pts = {"Sandstone":(55.5,2.65),"Limestone":(47.5,2.71),"Dolomite":(43.5,2.87),
                   "Anhydrite":(50.0,2.98),"Gypsum":(52.0,2.35),"Salt":(67.0,2.03)}
    for name,(dt_v,rho_v) in mineral_pts.items():
        clr = _MIN_COLORS.get(name, "#212121")
        fig.add_trace(go.Scatter(x=[dt_v],y=[rho_v],mode="markers",name=name,
                                 marker=dict(symbol="diamond",size=14,color=clr,
                                             line=dict(color="#212121",width=1.5))))
        fig.add_annotation(**_label_ann(name, x=dt_v+0.5, y=rho_v, color=clr))

    fig.update_xaxes(**_xax("Δtmaa (µs/ft)", range=[30,75]))
    fig.update_yaxes(**_xax("ρmaa (g/cm³)"), autorange=False, range=[3.15,1.90])
    fig.update_layout(**_base_layout("MID Plot — Δtmaa vs ρmaa", height=560))
    return fig


def plot_mid_u_rho(df, u_maa_col, rho_maa_col, color_col=None):
    pdf = df[[u_maa_col, rho_maa_col]].dropna()
    if color_col and color_col in df.columns:
        pdf[color_col] = df.loc[pdf.index, color_col]
        fig = px.scatter(pdf, x=u_maa_col, y=rho_maa_col, color=color_col,
                         color_continuous_scale="Viridis_r", opacity=0.60,
                         labels={color_col: color_col})
        _fix_cb(fig, color_col)
    else:
        fig = px.scatter(pdf, x=u_maa_col, y=rho_maa_col, opacity=0.50,
                         color_discrete_sequence=["#37474F"])

    u_refs = {"Sandstone":(4.79,2.65),"Limestone":(13.78,2.71),"Dolomite":(9.11,2.87),
              "Anhydrite":(15.06,2.98),"Gypsum":(18.76,2.35),"Halite":(9.49,2.17),
              "Illite":(10.97,2.52)}
    for name,(u_v,rho_v) in u_refs.items():
        clr = _MIN_COLORS.get(name,"#37474F")
        fig.add_trace(go.Scatter(x=[u_v],y=[rho_v],mode="markers",name=name,
                                 marker=dict(symbol="diamond",size=14,color=clr,
                                             line=dict(color="#212121",width=1.5))))
        fig.add_annotation(**_label_ann(name, x=u_v+0.2, y=rho_v, color=clr))

    fig.update_xaxes(**_xax("Umaa (barns/cm³)", range=[2,22]))
    fig.update_yaxes(**_xax("ρmaa (g/cm³)"),    autorange=False, range=[3.15,1.90])
    fig.update_layout(**_base_layout("MID Plot — Umaa vs ρmaa", height=560))
    return fig


def plot_crossplot(df, x_col, y_col, color_col=None, cluster_col=None, title="Crossplot"):
    pdf = df[[x_col, y_col]].copy()
    if cluster_col and cluster_col in df.columns:
        pdf["Cluster"] = df[cluster_col].astype(str)
        fig = px.scatter(pdf, x=x_col, y=y_col, color="Cluster", opacity=0.60,
                         title=title,
                         color_discrete_sequence=_TRACK_COLORS)
    elif color_col and color_col in df.columns:
        pdf[color_col] = df[color_col].values
        fig = px.scatter(pdf, x=x_col, y=y_col, color=color_col,
                         color_continuous_scale="Viridis_r", opacity=0.60, title=title,
                         labels={color_col: color_col})
        _fix_cb(fig, color_col)
    else:
        fig = px.scatter(pdf, x=x_col, y=y_col, opacity=0.50, title=title,
                         color_discrete_sequence=["#1565C0"])
    fig.update_xaxes(**_xax(x_col))
    fig.update_yaxes(**_xax(y_col))
    fig.update_layout(**_base_layout(title, height=540))
    return fig


def plot_cluster_strip(df, cluster_col, depth_col="DEPTH"):
    labels  = df[cluster_col].astype(str)
    unique  = sorted(labels.unique())
    c_map   = {lbl: _TRACK_COLORS[i % len(_TRACK_COLORS)] for i,lbl in enumerate(unique)}
    fig = go.Figure()
    for lbl in unique:
        sel = labels == lbl
        fig.add_trace(go.Scatter(
            x=[0.5]*int(sel.sum()), y=df.loc[sel, depth_col],
            mode="markers", name=f"Cluster {lbl}",
            marker=dict(color=c_map[lbl], size=9, symbol="square",
                        line=dict(color="white", width=0.5)),
        ))
    fig.update_yaxes(**_DEPTH_AX)
    fig.update_xaxes(visible=False, range=[0,1])
    fig.update_layout(**_base_layout("Clusters vs Depth", height=650))
    return fig


def plot_porosity(df, depth_col="DEPTH", phid_col=None, phin_col=None,
                  phis_col=None, phit_col=None, phie_col=None, shale_mask=None):
    tracks = [(c,lbl,clr) for c,lbl,clr in
              [(phid_col,"PHID","#1565C0"),(phin_col,"PHIN","#C62828"),
               (phis_col,"PHIS","#2E7D32")] if c and c in df.columns]
    has_combo = any(c and c in df.columns for c in [phit_col, phie_col])
    n_cols = len(tracks) + (1 if has_combo else 0)
    if n_cols == 0:
        return go.Figure()

    sub_titles = [t[1] for t in tracks] + (["PHIT vs PHIE"] if has_combo else [])
    fig = make_subplots(rows=1, cols=n_cols, shared_yaxes=True,
                        subplot_titles=sub_titles, horizontal_spacing=0.03)
    _fix_subtitles(fig)
    depth = df[depth_col]

    for i,(col,lbl,clr) in enumerate(tracks,1):
        fig.add_trace(go.Scatter(x=df[col],y=depth,mode="lines",name=lbl,
                                 line=dict(color=clr,width=1.5)), row=1,col=i)
        if shale_mask is not None and shale_mask.any():
            fig.add_trace(go.Scatter(x=df.loc[shale_mask,col],y=depth[shale_mask],
                                     mode="markers",name=f"{lbl} Shale",
                                     marker=dict(color="#5D4037",size=4),
                                     showlegend=(i==1)), row=1,col=i)
        fig.update_xaxes(**_xax(lbl), range=[0,0.50], row=1,col=i)

    if has_combo:
        last = n_cols
        if phit_col and phit_col in df.columns:
            fig.add_trace(go.Scatter(x=df[phit_col],y=depth,mode="lines",name="PHIT",
                                     line=dict(color="#0D47A1",width=2.0)),row=1,col=last)
        if phie_col and phie_col in df.columns:
            fig.add_trace(go.Scatter(x=df[phie_col],y=depth,mode="lines",name="PHIE",
                                     line=dict(color="#B71C1C",width=2.0,dash="dot")),row=1,col=last)
        fig.update_xaxes(**_xax("Porosity (v/v)"), range=[0,0.50], row=1,col=last)

    fig.update_yaxes(**_DEPTH_AX, row=1, col=1)
    fig.update_layout(**_base_layout("Porosity Logs", height=750), showlegend=True)
    return fig


def plot_pickett(df, rt_col, phit_col, rw=0.1, a=1.0, m=2.0, n=2.0, color_col=None):
    pdf = df[[rt_col, phit_col]].dropna()
    pdf = pdf[(pdf[rt_col]>0) & (pdf[phit_col]>0.005)]
    if color_col and color_col in df.columns:
        pdf[color_col] = df.loc[pdf.index, color_col]
        fig = px.scatter(pdf, x=phit_col, y=rt_col, color=color_col,
                         color_continuous_scale="Viridis_r", opacity=0.60,
                         log_x=True, log_y=True,
                         labels={color_col: color_col})
        _fix_cb(fig, color_col)
    else:
        fig = px.scatter(pdf, x=phit_col, y=rt_col, opacity=0.50,
                         log_x=True, log_y=True,
                         color_discrete_sequence=["#37474F"])

    phi_range = np.logspace(np.log10(0.01), np.log10(0.60), 200)
    for sw,(clr,dash,lbl) in {1.00:("#1565C0","solid","Sw=1.0  (water)"),
                               0.50:("#E65100","dash","Sw=0.50"),
                               0.25:("#C62828","dot","Sw=0.25")}.items():
        rt_line = (a*rw) / (sw**n * phi_range**m)
        fig.add_trace(go.Scatter(x=phi_range,y=rt_line,mode="lines",name=lbl,
                                 line=dict(color=clr,width=2.0,dash=dash)))

    fig.update_xaxes(**_xax("Porosity ϕ (v/v)"))
    fig.update_yaxes(**_xax("Rt (ohm·m)"))
    fig.update_layout(**_base_layout("Pickett Plot — Rw Estimation", height=560))
    return fig


def plot_sw(df, sw_col, phie_col=None, rt_col=None, sw_cut=0.60, depth_col="DEPTH"):
    tracks = [("sw",sw_col,"Sw","#1565C0")]
    if rt_col and rt_col in df.columns:   tracks.append(("rt", rt_col,   "RT (ohm·m)", "#C62828"))
    if phie_col and phie_col in df.columns: tracks.append(("phi",phie_col,"PHIE (v/v)","#2E7D32"))
    n = len(tracks)
    fig = make_subplots(rows=1,cols=n,shared_yaxes=True,
                        subplot_titles=[t[2] for t in tracks], horizontal_spacing=0.03)
    _fix_subtitles(fig)
    depth = df[depth_col]
    for i,(kind,col,title,clr) in enumerate(tracks,1):
        fig.add_trace(go.Scatter(x=df[col],y=depth,mode="lines",name=title,
                                 line=dict(color=clr,width=1.5),
                                 fill=("tozerox" if kind=="sw" else None),
                                 fillcolor="rgba(21,101,192,0.10)"),row=1,col=i)
        if kind=="rt":
            fig.update_xaxes(**_xax(title,log=True),row=1,col=i)
        elif kind=="sw":
            fig.update_xaxes(**_xax(title),range=[0,1],row=1,col=i)
            fig.add_vline(x=sw_cut,
                          line=dict(color="#C62828",dash="dash",width=1.8),
                          annotation_text=f"<b>Sw cut={sw_cut}</b>",
                          annotation_font=dict(color="#C62828",size=11),
                          annotation_bgcolor="rgba(255,255,255,0.88)",
                          annotation_bordercolor="#C62828")
        else:
            fig.update_xaxes(**_xax(title),range=[0,0.50],row=1,col=i)

    fig.update_yaxes(**_DEPTH_AX, row=1, col=1)
    fig.update_layout(**_base_layout("Fluid Analysis — Water Saturation", height=750),
                      showlegend=True)
    return fig


def plot_triple_combo(df, gr_col=None, rt_col=None, nphi_col=None,
                      rhob_col=None, depth_col="DEPTH"):
    fig = make_subplots(rows=1,cols=3,shared_yaxes=True,
                        subplot_titles=["GR (API)","Resistivity (ohm·m)","NPHI & RHOB"],
                        horizontal_spacing=0.03)
    _fix_subtitles(fig)
    depth = df[depth_col]

    if gr_col and gr_col in df.columns:
        fig.add_trace(go.Scatter(x=df[gr_col],y=depth,mode="lines",name="GR",
                                 line=dict(color="#558B2F",width=1.5),
                                 fill="tozerox",fillcolor="rgba(85,139,47,0.15)"),row=1,col=1)
    fig.update_xaxes(**_xax("GR (API)"),row=1,col=1)

    if rt_col and rt_col in df.columns:
        fig.add_trace(go.Scatter(x=df[rt_col],y=depth,mode="lines",name="RT",
                                 line=dict(color="#C62828",width=1.8),
                                 fill="tozerox",fillcolor="rgba(198,40,40,0.10)"),row=1,col=2)
    fig.update_xaxes(**_xax("RT (ohm·m)",log=True),row=1,col=2)

    if nphi_col and nphi_col in df.columns:
        nphi = df[nphi_col].copy()
        if nphi.dropna().median()>1.0: nphi = nphi/100.0
        fig.add_trace(go.Scatter(x=nphi,y=depth,mode="lines",name="NPHI",
                                 line=dict(color="#1565C0",width=1.5,dash="dash")),row=1,col=3)
    if rhob_col and rhob_col in df.columns:
        fig.add_trace(go.Scatter(x=df[rhob_col],y=depth,mode="lines",name="RHOB",
                                 line=dict(color="#212121",width=1.6)),row=1,col=3)
    fig.update_xaxes(**_xax("NPHI (v/v) / RHOB (g/cc)"),row=1,col=3)

    fig.update_yaxes(**_DEPTH_AX, row=1, col=1)
    fig.update_layout(**_base_layout("Triple Combo Log", height=850), showlegend=True)
    return fig


def plot_final_interpretation(df, pay_flag, gr_col=None, rt_col=None,
                               nphi_col=None, rhob_col=None,
                               phie_col=None, sw_col=None, depth_col="DEPTH"):
    depth = df[depth_col]
    ordered = []
    if gr_col   and gr_col   in df.columns: ordered.append(("GR",  gr_col,  "GR (API)",   "#558B2F","linear"))
    ordered.append(("COMBO",None,"NPHI+RHOB","","linear"))
    if rt_col   and rt_col   in df.columns: ordered.append(("RT",  rt_col,  "RT (ohm·m)", "#C62828","log"))
    if phie_col and phie_col in df.columns: ordered.append(("PHIE",phie_col,"PHIE (v/v)", "#2E7D32","linear"))
    if sw_col   and sw_col   in df.columns: ordered.append(("SW",  sw_col,  "Sw",         "#1565C0","linear"))
    ordered.append(("PAY",None,"Pay Flag","#00897B","linear"))

    n_cols = len(ordered)
    fig = make_subplots(rows=1,cols=n_cols,shared_yaxes=True,
                        subplot_titles=[t[2] for t in ordered],
                        horizontal_spacing=max(0.01,min(0.03,0.18/n_cols)))
    _fix_subtitles(fig)

    for i,(kind,col,title,clr,tp) in enumerate(ordered,1):
        if kind=="COMBO":
            if nphi_col and nphi_col in df.columns:
                nphi = df[nphi_col].copy()
                if nphi.dropna().median()>1.0: nphi=nphi/100.0
                fig.add_trace(go.Scatter(x=nphi,y=depth,mode="lines",name="NPHI",
                                         line=dict(color="#1565C0",width=1.3,dash="dash")),row=1,col=i)
            if rhob_col and rhob_col in df.columns:
                fig.add_trace(go.Scatter(x=df[rhob_col],y=depth,mode="lines",name="RHOB",
                                         line=dict(color="#212121",width=1.5)),row=1,col=i)
            fig.update_xaxes(**_xax("NPHI/RHOB"),row=1,col=i)
        elif kind=="PAY":
            pv = pay_flag.astype(float)
            fig.add_trace(go.Scatter(x=pv,y=depth,mode="lines",name="Pay",
                                     line=dict(color="#00897B",width=2.5),
                                     fill="tozerox",fillcolor="rgba(0,137,123,0.28)"),row=1,col=i)
            fig.update_xaxes(**_xax("Pay"),range=[-0.1,1.5],row=1,col=i)
        else:
            fm={"GR":("tozerox","rgba(85,139,47,0.15)"),
                "PHIE":("tozerox","rgba(46,125,50,0.15)"),
                "SW":("tozerox","rgba(21,101,192,0.10)")}
            fill,fill_c = fm.get(kind,(None,None))
            kw = dict(fill=fill,fillcolor=fill_c) if fill else {}
            fig.add_trace(go.Scatter(x=df[col],y=depth,mode="lines",name=title,
                                     line=dict(color=clr,width=1.5),**kw),row=1,col=i)
            fig.update_xaxes(**_xax(title,log=(tp=="log")),row=1,col=i)

    pay_np   = pay_flag.values.astype(bool)
    depth_np = depth.values
    in_zone, zone_top = False, None
    for d,p in zip(depth_np,pay_np):
        if p and not in_zone:  zone_top=d; in_zone=True
        elif not p and in_zone:
            fig.add_hrect(y0=zone_top,y1=d,fillcolor="rgba(0,137,123,0.10)",line_width=0)
            in_zone=False
    if in_zone and zone_top is not None:
        fig.add_hrect(y0=zone_top,y1=depth_np[-1],fillcolor="rgba(0,137,123,0.10)",line_width=0)

    fig.update_yaxes(**_DEPTH_AX, row=1, col=1)
    fig.update_layout(**_base_layout("Final Interpretation — Composite Log", height=880),
                      showlegend=True)
    return fig


def plot_core_vs_log(log_phi, core_phi, slope, intercept):
    valid = pd.DataFrame({"log": log_phi, "core": core_phi}).dropna()
    x_reg = np.linspace(float(valid["log"].min()), float(valid["log"].max()), 200)
    y_reg = slope*x_reg + intercept
    lim   = float(max(valid.max().max(), 0.01)) * 1.05
    fig   = go.Figure()
    fig.add_trace(go.Scatter(x=valid["log"],y=valid["core"],mode="markers",name="Core data",
                             marker=dict(color="#1565C0",size=8,opacity=0.75)))
    fig.add_trace(go.Scatter(x=x_reg,y=y_reg,mode="lines",
                             name=f"Fit  y={slope:.3f}x+{intercept:.4f}",
                             line=dict(color="#C62828",width=2.2)))
    fig.add_trace(go.Scatter(x=[0,lim],y=[0,lim],mode="lines",name="1:1 line",
                             line=dict(color="#757575",dash="dash",width=1.5)))
    fig.update_xaxes(**_xax("Log Porosity (v/v)", range=[0,lim]))
    fig.update_yaxes(**_xax("Core Porosity (v/v)", range=[0,lim]))
    fig.update_layout(**_base_layout("Core vs Log Porosity Calibration", height=480))
    return fig
