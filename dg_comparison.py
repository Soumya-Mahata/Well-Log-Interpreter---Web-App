"""
dg_comparison.py  —  Model Comparison & Evaluation
====================================================
Compares all Vp/Vs prediction columns stored in session_state.df:

  A. Metrics summary table  (Model | Target | RMSE | MAE | R²)
  B. Grouped bar charts     (one chart per metric)
  C. Per-model detail       depth track overlay + scatter + residual histogram
  D. Uncertainty summary    bar chart of mean ±σ per prediction column
  E. Inter-model stability  prediction variance across methods

Public entry point:
    from dg_comparison import render
    render(df)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from dg_utils import (
    find_col, numeric_cols, safe_metric,
    section_header,
    plotly_depth_track, plotly_scatter, plotly_residual_hist,
    COLORS,
)


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _pred_cols(df: pd.DataFrame, tag: str) -> list[str]:
    """Return all columns whose name contains `tag` and 'pred' (case-insensitive)."""
    return [c for c in df.columns
            if tag in c and "pred" in c.lower()]


def _build_metrics_rows(df: pd.DataFrame,
                        pred_vp: list[str], pred_vs: list[str],
                        vp_true: str | None, vs_true: str | None) -> list[dict]:
    rows = []
    for pc in pred_vp:
        if vp_true and vp_true in df.columns:
            m = safe_metric(df[vp_true].values.astype(float),
                            df[pc].values.astype(float))
        else:
            m = {"RMSE": None, "MAE": None, "R²": None}
        rows.append({"Model / Method": pc, "Target": "Vp", **m})

    for pc in pred_vs:
        if vs_true and vs_true in df.columns:
            m = safe_metric(df[vs_true].values.astype(float),
                            df[pc].values.astype(float))
        else:
            m = {"RMSE": None, "MAE": None, "R²": None}
        rows.append({"Model / Method": pc, "Target": "Vs", **m})

    return rows


# ═════════════════════════════════════════════════════════════════════════════
# SECTION RENDERERS
# ═════════════════════════════════════════════════════════════════════════════

def _render_metrics_table(rows: list[dict]) -> pd.DataFrame:
    """Section A — styled dataframe."""
    st.markdown("### A — Metrics Summary")
    mdf = pd.DataFrame(rows)
    fmt = {c: "{:.5f}" for c in ["RMSE", "MAE", "R²"] if c in mdf.columns}
    st.dataframe(
        mdf.style.format(fmt, na_rep="—"),
        use_container_width=True,
        hide_index=True,
    )
    return mdf


def _render_bar_charts(mdf: pd.DataFrame) -> None:
    """Section B — grouped bar chart per metric."""
    numeric_mdf = mdf.dropna(subset=["RMSE"])
    if numeric_mdf.empty:
        st.info("Assign true Vp/Vs columns above to see metric charts.")
        return

    st.markdown("### B — Metric Comparison Charts")
    for metric in ["RMSE", "MAE", "R²"]:
        if metric not in numeric_mdf.columns:
            continue
        fig = px.bar(
            numeric_mdf,
            x="Model / Method", y=metric, color="Target",
            barmode="group", text_auto=".4f",
            color_discrete_map={"Vp": COLORS[0], "Vs": COLORS[1]},
            title=f"{metric} — all models & targets",
        )
        fig.update_layout(
            height=320, xaxis_tickangle=-30,
            margin=dict(l=40, r=20, t=50, b=90),
            legend_title_text="Target",
        )
        st.plotly_chart(fig, use_container_width=True,
                        key=f"dg_cmp_bar_{metric}")


def _render_per_model(df: pd.DataFrame,
                      pred_vp: list[str], pred_vs: list[str],
                      vp_true: str | None, vs_true: str | None) -> None:
    """Section C — depth overlay + per-model scatter & residual."""
    st.markdown("### C — Prediction vs True  (per model)")

    for tgt_lbl, true_col, preds_t in [
        ("Vp", vp_true, pred_vp),
        ("Vs", vs_true, pred_vs),
    ]:
        if not preds_t:
            continue

        st.markdown(f"#### {tgt_lbl}")

        # Overlay depth track (original + all predictions)
        curves, clabels, colors = [], [], []
        if true_col and true_col in df.columns:
            curves.append(true_col)
            clabels.append("Original")
            colors.append("#212121")
        for i, pc in enumerate(preds_t):
            curves.append(pc)
            clabels.append(pc)
            colors.append(COLORS[(i + 1) % len(COLORS)])

        st.plotly_chart(
            plotly_depth_track(df, curves, clabels, colors,
                               f"{tgt_lbl} — All Predictions vs Depth",
                               tgt_lbl),
            use_container_width=True,
            key=f"dg_cmp_depth_{tgt_lbl}",
        )

        # Dropdown → focused scatter + residual for one chosen model
        sel = st.selectbox(
            f"Inspect **{tgt_lbl}** prediction in detail",
            preds_t,
            key=f"dg_cmp_sel_{tgt_lbl}",
        )

        if true_col and true_col in df.columns and sel in df.columns:
            t_arr = df[true_col].values.astype(float)
            p_arr = df[sel].values.astype(float)
            sc1, sc2 = st.columns(2)
            with sc1:
                st.plotly_chart(
                    plotly_scatter(t_arr, p_arr, tgt_lbl, sel),
                    use_container_width=True,
                    key=f"dg_cmp_sc_{tgt_lbl}_{sel}",
                )
            with sc2:
                st.plotly_chart(
                    plotly_residual_hist(t_arr, p_arr, tgt_lbl),
                    use_container_width=True,
                    key=f"dg_cmp_res_{tgt_lbl}_{sel}",
                )
        elif preds_t:
            st.info(f"Assign a **True {tgt_lbl}** column above to see scatter & residual plots.")


def _render_uncertainty(df: pd.DataFrame) -> None:
    """Section D — uncertainty bar chart."""
    unc_cols = [c for c in df.columns if "uncertainty" in c.lower()]
    if not unc_cols:
        return

    st.markdown("### D — Uncertainty Summary (±σ)")
    unc_stats = df[unc_cols].describe().T[["mean", "std", "min", "max"]]
    st.dataframe(unc_stats.style.format("{:.5f}"), use_container_width=True)

    fig = px.bar(
        unc_stats.reset_index().rename(columns={"index": "Column"}),
        x="Column", y="mean", error_y="std", color="Column",
        title="Mean Uncertainty (±σ) per Prediction Column",
        labels={"mean": "Mean ±σ"},
    )
    fig.update_layout(height=300, showlegend=False,
                      margin=dict(l=40, r=20, t=50, b=60))
    st.plotly_chart(fig, use_container_width=True, key="dg_cmp_unc_bar")


def _render_stability(df: pd.DataFrame,
                      pred_vp: list[str], pred_vs: list[str]) -> None:
    """Section E — inter-model prediction variance."""
    shown = False
    for tgt_lbl, preds_t in [("Vp", pred_vp), ("Vs", pred_vs)]:
        if len(preds_t) > 1:
            sub = df[preds_t].dropna()
            if not sub.empty:
                if not shown:
                    st.markdown("### E — Inter-model Stability")
                    shown = True
                st.metric(
                    f"{tgt_lbl} prediction variance across models",
                    f"{sub.var(axis=1).mean():.6f}",
                    help="Lower = more consistent predictions across methods",
                )


# ═════════════════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def render(df: pd.DataFrame) -> None:
    section_header("📊", "Model Comparison & Evaluation",
                   "All Vp / Vs predictions compared side-by-side")

    num  = numeric_cols(df)
    opts = ["None"] + num
    auto_vp = find_col(df, ["VP", "VEL", "DT", "DTCO"])
    auto_vs = find_col(df, ["VS", "DTS", "DTSM"])

    # Ground-truth column selectors
    c1, c2 = st.columns(2)
    vp_true = c1.selectbox(
        "True Vp column", opts, key="dg_cmp_vp_true",
        index=(opts.index(auto_vp) if auto_vp in opts else 0),
    )
    vs_true = c2.selectbox(
        "True Vs column", opts, key="dg_cmp_vs_true",
        index=(opts.index(auto_vs) if auto_vs in opts else 0),
    )
    vp_true = None if vp_true == "None" else vp_true
    vs_true = None if vs_true == "None" else vs_true

    pred_vp = _pred_cols(df, "Vp")
    pred_vs = _pred_cols(df, "Vs")

    if not pred_vp and not pred_vs:
        st.info(
            "No prediction columns found yet.\n\n"
            "Run **Conventional**, **Machine Learning**, or **Deep Learning** "
            "first to generate predictions, then return here."
        )
        return

    st.divider()

    rows = _build_metrics_rows(df, pred_vp, pred_vs, vp_true, vs_true)
    mdf  = _render_metrics_table(rows)

    st.divider()
    _render_bar_charts(mdf)

    st.divider()
    _render_per_model(df, pred_vp, pred_vs, vp_true, vs_true)

    st.divider()
    _render_uncertainty(df)

    st.divider()
    _render_stability(df, pred_vp, pred_vs)
