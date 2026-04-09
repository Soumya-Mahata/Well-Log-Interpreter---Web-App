"""
dg_utils.py  —  Shared utilities for the Missing Sonic Data Generation system
==============================================================================
Provides:
  - Column-finder helpers
  - Metric computation
  - Scaler factory
  - All Plotly figure builders
  - Unified _show_results() display block
  - Theme-safe section header

Imported by: dg_conventional.py, dg_unconventional.py, dg_comparison.py
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ── Palette ───────────────────────────────────────────────────────────────────
COLORS = ["#1976D2", "#E53935", "#2E7D32", "#F57F17",
          "#6A1B9A", "#00838F", "#AD1457"]


# ═════════════════════════════════════════════════════════════════════════════
# THEME-SAFE SECTION HEADER
# ═════════════════════════════════════════════════════════════════════════════

def section_header(icon: str, title: str, subtitle: str = "") -> None:
    sub = (f"<br><span style='font-size:0.85rem;opacity:0.7;'>{subtitle}</span>"
           if subtitle else "")
    st.markdown(
        f"""<div style="border-left:4px solid #1976D2;padding:8px 14px;
                        margin:18px 0 10px 0;border-radius:0 6px 6px 0;">
              <span style="font-size:1.25rem;font-weight:700;">{icon}&nbsp;{title}</span>
              {sub}
            </div>""",
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
# DATAFRAME HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return first matching column name (case-insensitive)."""
    upper = {c.upper(): c for c in df.columns}
    for c in candidates:
        if c.upper() in upper:
            return upper[c.upper()]
    return None


def numeric_cols(df: pd.DataFrame) -> list[str]:
    """All numeric columns except DEPTH."""
    return [c for c in df.columns
            if c != "DEPTH" and pd.api.types.is_numeric_dtype(df[c])]


# ═════════════════════════════════════════════════════════════════════════════
# METRICS
# ═════════════════════════════════════════════════════════════════════════════

def safe_metric(true: np.ndarray, pred: np.ndarray) -> dict:
    mask = ~(np.isnan(true) | np.isnan(pred))
    if mask.sum() < 2:
        return {"RMSE": float("nan"), "MAE": float("nan"), "R²": float("nan")}
    t, p = true[mask], pred[mask]
    return {
        "RMSE": float(np.sqrt(mean_squared_error(t, p))),
        "MAE":  float(mean_absolute_error(t, p)),
        "R²":   float(r2_score(t, p)),
    }


# ═════════════════════════════════════════════════════════════════════════════
# SCALER FACTORY
# ═════════════════════════════════════════════════════════════════════════════

def get_scaler(name: str):
    return StandardScaler() if name == "StandardScaler" else MinMaxScaler()


# ═════════════════════════════════════════════════════════════════════════════
# PLOTLY FIGURE BUILDERS
# ═════════════════════════════════════════════════════════════════════════════

def plotly_depth_track(df: pd.DataFrame,
                       curves: list[str],
                       labels: list[str],
                       colors: list[str],
                       title: str,
                       xlabel: str) -> go.Figure:
    """Multi-curve depth track (y = DEPTH, x = log value, inverted y-axis)."""
    fig = go.Figure()
    for curve, label, color in zip(curves, labels, colors):
        if curve in df.columns:
            fig.add_trace(go.Scatter(
                x=df[curve], y=df["DEPTH"],
                mode="lines", name=label,
                line=dict(color=color, width=1.6),
                hovertemplate=(
                    f"<b>{label}</b><br>{xlabel}: %{{x:.4f}}"
                    "<br>Depth: %{y:.2f}<extra></extra>"
                ),
            ))
    fig.update_yaxes(autorange="reversed", title="Depth")
    fig.update_xaxes(title=xlabel)
    fig.update_layout(
        title=title, height=520, hovermode="y unified",
        legend=dict(orientation="h", yanchor="bottom",
                    y=1.01, xanchor="left", x=0),
        margin=dict(l=60, r=20, t=60, b=40),
    )
    return fig


def plotly_scatter(true: np.ndarray, pred: np.ndarray,
                   label: str, model_name: str) -> go.Figure:
    """True vs Predicted scatter with 1:1 line and R² / RMSE in title."""
    mask = ~(np.isnan(true) | np.isnan(pred))
    t, p = true[mask], pred[mask]
    lo, hi = min(t.min(), p.min()), max(t.max(), p.max())
    m = safe_metric(true, pred)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=p, mode="markers",
        marker=dict(color="#1976D2", size=4, opacity=0.45),
        name="Samples",
        hovertemplate="True: %{x:.4f}<br>Pred: %{y:.4f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], mode="lines",
        line=dict(color="#E53935", dash="dash", width=1.5),
        name="1 : 1",
    ))
    fig.update_layout(
        title=(f"{model_name} — {label}<br>"
               f"<sup>R²={m['R²']:.4f}  RMSE={m['RMSE']:.4f}</sup>"),
        xaxis_title=f"True {label}",
        yaxis_title=f"Predicted {label}",
        height=380, margin=dict(l=60, r=20, t=70, b=40),
    )
    return fig


def plotly_residual_hist(true: np.ndarray, pred: np.ndarray,
                         label: str) -> go.Figure:
    """Residual (Pred − True) histogram."""
    mask = ~(np.isnan(true) | np.isnan(pred))
    err = pred[mask] - true[mask]
    fig = go.Figure(go.Histogram(
        x=err, nbinsx=60, marker_color="#1976D2", opacity=0.8,
        hovertemplate="Residual: %{x:.4f}<br>Count: %{y}<extra></extra>",
    ))
    fig.add_vline(x=0, line=dict(color="#E53935", dash="dash", width=1.5))
    fig.update_layout(
        title=f"Residual Distribution — {label}",
        xaxis_title=f"Pred − True  [{label}]",
        yaxis_title="Count",
        height=320, margin=dict(l=60, r=20, t=50, b=40),
    )
    return fig


def plotly_loss_curve(train_l: list, val_l: list,
                      model_name: str) -> go.Figure:
    """Training / validation MSE loss curve."""
    epochs = list(range(1, len(train_l) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=train_l, mode="lines",
                             name="Train MSE",
                             line=dict(color="#1976D2", width=2)))
    if val_l:
        fig.add_trace(go.Scatter(x=epochs, y=val_l, mode="lines",
                                 name="Val MSE",
                                 line=dict(color="#E53935",
                                           dash="dash", width=2)))
    fig.update_layout(
        title=f"{model_name} — Training Loss",
        xaxis_title="Epoch", yaxis_title="MSE",
        height=320,
        legend=dict(orientation="h", y=1.02, x=0),
        margin=dict(l=60, r=20, t=60, b=40),
    )
    return fig


def plotly_uncertainty_band(df: pd.DataFrame,
                             pred_col: str, unc_col: str,
                             true_col: str | None,
                             label: str) -> go.Figure:
    """Depth track with ±1σ shaded uncertainty band."""
    unc   = df[unc_col].values.astype(float)
    pred  = df[pred_col].values.astype(float)
    depth = df["DEPTH"].values
    fig = go.Figure()
    # Band
    fig.add_trace(go.Scatter(
        x=np.concatenate([pred + unc, (pred - unc)[::-1]]),
        y=np.concatenate([depth, depth[::-1]]),
        fill="toself", fillcolor="rgba(25,118,210,0.15)",
        line=dict(color="rgba(0,0,0,0)"), name="±1σ",
    ))
    fig.add_trace(go.Scatter(
        x=pred, y=depth, mode="lines",
        line=dict(color="#1976D2", width=1.5), name="Predicted",
    ))
    if true_col and true_col in df.columns:
        fig.add_trace(go.Scatter(
            x=df[true_col].values, y=depth, mode="lines",
            line=dict(color="#E53935", dash="dot", width=1.2),
            name="Original",
        ))
    fig.update_yaxes(autorange="reversed", title="Depth")
    fig.update_xaxes(title=label)
    fig.update_layout(
        title=f"{label} — Prediction with Uncertainty",
        height=480, hovermode="y unified",
        margin=dict(l=60, r=20, t=50, b=40),
    )
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# UNIFIED RESULT DISPLAY
# ═════════════════════════════════════════════════════════════════════════════

def show_results(df: pd.DataFrame,
                 pred_col: str,
                 true_col: str | None,
                 label: str,
                 model_name: str = "",
                 unc_col: str | None = None,
                 key_prefix: str = "res") -> None:
    """
    Renders: KPI metrics | depth track | scatter + residual | uncertainty band.
    All plot keys are namespaced with key_prefix to avoid duplicate-widget errors.
    """
    section_header("📈", f"{label} Prediction Results", model_name)

    # KPI row
    if true_col and true_col in df.columns and pred_col in df.columns:
        ta = df[true_col].values.astype(float)
        pa = df[pred_col].values.astype(float)
        m  = safe_metric(ta, pa)
        kc1, kc2, kc3 = st.columns(3)
        kc1.metric("RMSE", f"{m['RMSE']:.5f}")
        kc2.metric("MAE",  f"{m['MAE']:.5f}")
        kc3.metric("R²",   f"{m['R²']:.4f}")
    else:
        ta = pa = None
        if pred_col in df.columns:
            st.info("No ground-truth column — prediction statistics only.")

    # Depth track
    if pred_col in df.columns:
        curves  = [pred_col]
        clabels = ["Predicted"]
        cols    = [COLORS[1]]
        if true_col and true_col in df.columns:
            curves  = [true_col, pred_col]
            clabels = ["Original", "Predicted"]
            cols    = [COLORS[0], COLORS[1]]
        st.plotly_chart(
            plotly_depth_track(df, curves, clabels, cols,
                               f"{label} vs Depth", label),
            use_container_width=True,
            key=f"{key_prefix}_depth_{pred_col}_{label}",
        )

    # Scatter + residual
    if ta is not None and pa is not None:
        sc1, sc2 = st.columns(2)
        with sc1:
            st.plotly_chart(
                plotly_scatter(ta, pa, label, model_name or pred_col),
                use_container_width=True,
                key=f"{key_prefix}_sc_{pred_col}_{label}",
            )
        with sc2:
            st.plotly_chart(
                plotly_residual_hist(ta, pa, label),
                use_container_width=True,
                key=f"{key_prefix}_res_{pred_col}_{label}",
            )

    # Uncertainty band (collapsible)
    if unc_col and unc_col in df.columns and pred_col in df.columns:
        with st.expander("📊 Uncertainty Band (±1σ)"):
            st.plotly_chart(
                plotly_uncertainty_band(df, pred_col, unc_col,
                                        true_col, label),
                use_container_width=True,
                key=f"{key_prefix}_unc_{pred_col}_{label}",
            )

    # Stats table (collapsible)
    with st.expander("Detailed Statistics"):
        cols_show = [c for c in [true_col, pred_col, unc_col]
                     if c and c in df.columns]
        if cols_show:
            st.dataframe(df[cols_show].describe().style.format("{:.5f}"),
                         use_container_width=True)
