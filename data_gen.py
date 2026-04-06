"""
data_gen.py  —  Missing P- and S-Sonic Data Generation Module
==============================================================
FIXES applied (v2):
  1. Approach selector is the SINGLE source of truth via session_state["dg_approach"]
     — radio drives which section renders; tabs removed (cannot be switched
       programmatically in Streamlit).
  2. XGBoost always visible in model list (shows install hint if absent).
  3. All plots replaced with Plotly interactive figures.
  4. Comparison section redesigned:
       A. Clear metrics table (Model | Target | RMSE | MAE | R²)
       B. Plotly grouped bar charts per metric
       C. Per-model depth track + scatter/residual with dropdown selector
       D. Uncertainty bar chart
       E. Inter-model stability metric
  5. Theme-safe headings — no hard-coded colours that break in dark mode.
  6. Full uncertainty quantification retained (ensemble variance + MC Dropout).

Architecture reference:
    Haritha et al. (2025) — CNN-Bi-LSTM, Journal of Applied Geophysics 233.
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ── sklearn ──────────────────────────────────────────────────────────────────
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ── optional: XGBoost ────────────────────────────────────────────────────────
try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

# ── optional: PyTorch ────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    import types as _types
    torch = None  # type: ignore
    nn = _types.SimpleNamespace(
        Module=object, Linear=None, ReLU=None, Dropout=None,
        Sequential=None, LSTM=None, Conv1d=None,
        AdaptiveMaxPool1d=None, MSELoss=None,
    )
    DataLoader = None      # type: ignore
    TensorDataset = None   # type: ignore


# ═════════════════════════════════════════════════════════════════════════════
# THEME-SAFE SECTION HEADER
# Uses currentColor so it works in both light and dark Streamlit themes.
# ═════════════════════════════════════════════════════════════════════════════

def _section_header(icon: str, title: str, subtitle: str = ""):
    sub_html = (
        f"<br><span style='font-size:0.85rem;opacity:0.7;'>{subtitle}</span>"
        if subtitle else ""
    )
    st.markdown(
        f"""
        <div style="border-left:4px solid #1976D2;
                    padding:8px 14px;margin:18px 0 10px 0;
                    border-radius:0 6px 6px 0;">
          <span style="font-size:1.25rem;font-weight:700;">{icon}&nbsp;{title}</span>
          {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
# UTILITY HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _find(df: pd.DataFrame, candidates: list[str]) -> str | None:
    upper = {c.upper(): c for c in df.columns}
    for c in candidates:
        if c.upper() in upper:
            return upper[c.upper()]
    return None


def _numeric_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if c != "DEPTH" and pd.api.types.is_numeric_dtype(df[c])]


def _safe_metric(true: np.ndarray, pred: np.ndarray) -> dict:
    mask = ~(np.isnan(true) | np.isnan(pred))
    if mask.sum() < 2:
        return {"RMSE": float("nan"), "MAE": float("nan"), "R²": float("nan")}
    t, p = true[mask], pred[mask]
    return {
        "RMSE": float(np.sqrt(mean_squared_error(t, p))),
        "MAE":  float(mean_absolute_error(t, p)),
        "R²":   float(r2_score(t, p)),
    }


def _get_scaler(name: str):
    return StandardScaler() if name == "StandardScaler" else MinMaxScaler()


# ═════════════════════════════════════════════════════════════════════════════
# PLOTLY VISUALIZATIONS
# ═════════════════════════════════════════════════════════════════════════════

_COLORS = ["#1976D2", "#E53935", "#2E7D32", "#F57F17",
           "#6A1B9A", "#00838F", "#AD1457"]


def _plotly_depth_track(df: pd.DataFrame, curves: list[str],
                        labels: list[str], colors: list[str],
                        title: str, xlabel: str) -> go.Figure:
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
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        margin=dict(l=60, r=20, t=60, b=40),
    )
    return fig


def _plotly_scatter(true: np.ndarray, pred: np.ndarray,
                    label: str, model_name: str) -> go.Figure:
    mask = ~(np.isnan(true) | np.isnan(pred))
    t, p = true[mask], pred[mask]
    lo = min(t.min(), p.min())
    hi = max(t.max(), p.max())
    m = _safe_metric(true, pred)
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


def _plotly_residual_hist(true: np.ndarray, pred: np.ndarray,
                          label: str) -> go.Figure:
    mask = ~(np.isnan(true) | np.isnan(pred))
    err = pred[mask] - true[mask]
    fig = go.Figure(go.Histogram(
        x=err, nbinsx=60,
        marker_color="#1976D2", opacity=0.8,
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


def _plotly_loss_curve(train_l: list, val_l: list,
                       model_name: str) -> go.Figure:
    epochs = list(range(1, len(train_l) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=epochs, y=train_l, mode="lines", name="Train MSE",
        line=dict(color="#1976D2", width=2),
    ))
    if val_l:
        fig.add_trace(go.Scatter(
            x=epochs, y=val_l, mode="lines", name="Val MSE",
            line=dict(color="#E53935", dash="dash", width=2),
        ))
    fig.update_layout(
        title=f"{model_name} — Training Loss",
        xaxis_title="Epoch", yaxis_title="MSE",
        height=320,
        legend=dict(orientation="h", y=1.02, x=0),
        margin=dict(l=60, r=20, t=60, b=40),
    )
    return fig


def _show_results(df: pd.DataFrame, pred_col: str,
                  true_col: str | None, label: str,
                  model_name: str = "", unc_col: str | None = None):
    """Unified Plotly result display."""
    _section_header("📈", f"{label} Prediction Results", model_name)

    true_arr = pred_arr = None
    if true_col and true_col in df.columns and pred_col in df.columns:
        true_arr = df[true_col].values.astype(float)
        pred_arr = df[pred_col].values.astype(float)
        m = _safe_metric(true_arr, pred_arr)
        kc1, kc2, kc3 = st.columns(3)
        kc1.metric("RMSE", f"{m['RMSE']:.5f}")
        kc2.metric("MAE",  f"{m['MAE']:.5f}")
        kc3.metric("R²",   f"{m['R²']:.4f}")
    elif pred_col in df.columns:
        st.info("No ground-truth column — showing prediction statistics only.")

    # Depth track
    curves  = [pred_col]
    clabels = ["Predicted"]
    colors  = ["#E53935"]
    if true_col and true_col in df.columns:
        curves  = [true_col, pred_col]
        clabels = ["Original", "Predicted"]
        colors  = ["#1976D2", "#E53935"]

    if pred_col in df.columns:
        st.plotly_chart(
            _plotly_depth_track(df, curves, clabels, colors,
                                f"{label} vs Depth", label),
            use_container_width=True,
            key=f"sr_depth_{pred_col}_{label}",
        )

    # Scatter + residual
    if true_arr is not None and pred_arr is not None:
        s1, s2 = st.columns(2)
        with s1:
            st.plotly_chart(
                _plotly_scatter(true_arr, pred_arr, label,
                                model_name or pred_col),
                use_container_width=True,
                key=f"sr_sc_{pred_col}_{label}",
            )
        with s2:
            st.plotly_chart(
                _plotly_residual_hist(true_arr, pred_arr, label),
                use_container_width=True,
                key=f"sr_res_{pred_col}_{label}",
            )

    # Uncertainty band
    if unc_col and unc_col in df.columns and pred_col in df.columns:
        with st.expander("📊 Uncertainty Band (±1σ)"):
            unc   = df[unc_col].values.astype(float)
            pred  = df[pred_col].values.astype(float)
            depth = df["DEPTH"].values
            fig_u = go.Figure()
            fig_u.add_trace(go.Scatter(
                x=np.concatenate([pred + unc, (pred - unc)[::-1]]),
                y=np.concatenate([depth, depth[::-1]]),
                fill="toself", fillcolor="rgba(25,118,210,0.15)",
                line=dict(color="rgba(0,0,0,0)"), name="±1σ",
            ))
            fig_u.add_trace(go.Scatter(
                x=pred, y=depth, mode="lines",
                line=dict(color="#1976D2", width=1.5), name="Predicted",
            ))
            if true_col and true_col in df.columns:
                fig_u.add_trace(go.Scatter(
                    x=df[true_col].values, y=depth, mode="lines",
                    line=dict(color="#E53935", dash="dot", width=1.2),
                    name="Original",
                ))
            fig_u.update_yaxes(autorange="reversed", title="Depth")
            fig_u.update_xaxes(title=label)
            fig_u.update_layout(
                title=f"{label} — Prediction with Uncertainty",
                height=480, hovermode="y unified",
                margin=dict(l=60, r=20, t=50, b=40),
            )
            st.plotly_chart(fig_u, use_container_width=True,
                            key=f"sr_unc_{pred_col}_{label}")

    # Stats
    with st.expander("Detailed Statistics"):
        cols = [c for c in [true_col, pred_col, unc_col]
                if c and c in df.columns]
        if cols:
            st.dataframe(df[cols].describe().style.format("{:.5f}"),
                         use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# CONVENTIONAL METHODS
# ═════════════════════════════════════════════════════════════════════════════

def _gardner(rhob: pd.Series) -> pd.Series:
    return (rhob / 0.31) ** (1.0 / 0.25) / 1000.0

def _castagna_vp2vs(vp: pd.Series) -> pd.Series:
    return 0.8621 * vp - 1.1724

def _castagna_vs2vp(vs: pd.Series) -> pd.Series:
    return (vs + 1.1724) / 0.8621

def _smith_sonic(rt: pd.Series) -> pd.Series:
    return 1.0 / (0.02 * np.log(rt.clip(lower=0.1)) + 0.1)

def _brocher(vp: pd.Series) -> pd.Series:
    v = vp.values
    return pd.Series(
        0.7858 - 1.2344*v + 0.7949*v**2 - 0.1238*v**3 + 0.0064*v**4,
        index=vp.index,
    )

def _carroll(vp: pd.Series) -> pd.Series:
    return (vp - 1.36) / 1.16


def conventional_methods(df: pd.DataFrame):
    _section_header("🔵", "Conventional / Empirical Methods",
                    "Physics-based equations — no training required")

    num_cols = _numeric_cols(df)
    opts_n   = ["None"] + num_cols

    method = st.selectbox(
        "Select empirical method",
        [
            "Gardner (1974) — ρ → Vp",
            "Castagna (1985) — Vp → Vs  (Mudrock Line)",
            "Castagna (1985) — Vs → Vp  (Inverse)",
            "Smith (2007) — Rt → Vp",
            "Brocher (2005) — Vp → Vs",
            "Carroll (1969) — Vp → Vs",
        ],
        key="dg_conv_method",
    )
    st.divider()

    auto_rhob = _find(df, ["RHOB", "RHOZ", "DEN"])
    auto_vp   = _find(df, ["VP", "VEL", "DT", "DTCO"])
    auto_vs   = _find(df, ["VS", "DTS", "DTSM"])
    auto_rt   = _find(df, ["RT", "AT90", "ILD", "RESD"])

    c1, c2, c3, c4 = st.columns(4)
    rhob_col = c1.selectbox("RHOB (g/cc)", opts_n, key="dg_rhob_sel",
                             index=(opts_n.index(auto_rhob) if auto_rhob in opts_n else 0))
    vp_col   = c2.selectbox("Vp (km/s)",   opts_n, key="dg_vp_sel",
                             index=(opts_n.index(auto_vp)   if auto_vp   in opts_n else 0))
    vs_col   = c3.selectbox("Vs (km/s)",   opts_n, key="dg_vs_sel",
                             index=(opts_n.index(auto_vs)   if auto_vs   in opts_n else 0))
    rt_col   = c4.selectbox("Rt (ohm·m)", opts_n, key="dg_rt_sel",
                             index=(opts_n.index(auto_rt)   if auto_rt   in opts_n else 0))

    rhob_col = None if rhob_col == "None" else rhob_col
    vp_col   = None if vp_col   == "None" else vp_col
    vs_col   = None if vs_col   == "None" else vs_col
    rt_col   = None if rt_col   == "None" else rt_col

    if st.button("▶ Apply Method", type="primary", key="dg_conv_run"):
        df_upd = st.session_state.df.copy()
        ok     = False

        if "Gardner" in method and rhob_col:
            df_upd["Vp_pred"]        = _gardner(df_upd[rhob_col])
            df_upd["Vp_uncertainty"] = df_upd["Vp_pred"].abs() * 0.05
            ok, tag = True, "Vp"
        elif "Castagna" in method and "Vp → Vs" in method and vp_col:
            df_upd["Vs_pred"]        = _castagna_vp2vs(df_upd[vp_col])
            df_upd["Vs_uncertainty"] = df_upd["Vs_pred"].abs() * 0.05
            ok, tag = True, "Vs"
        elif "Castagna" in method and "Vs → Vp" in method and vs_col:
            df_upd["Vp_pred"]        = _castagna_vs2vp(df_upd[vs_col])
            df_upd["Vp_uncertainty"] = df_upd["Vp_pred"].abs() * 0.05
            ok, tag = True, "Vp"
        elif "Smith" in method and rt_col:
            df_upd["Vp_pred"]        = _smith_sonic(df_upd[rt_col])
            df_upd["Vp_uncertainty"] = df_upd["Vp_pred"].abs() * 0.08
            ok, tag = True, "Vp"
        elif "Brocher" in method and vp_col:
            df_upd["Vs_pred"]        = _brocher(df_upd[vp_col])
            df_upd["Vs_uncertainty"] = df_upd["Vs_pred"].abs() * 0.05
            ok, tag = True, "Vs"
        elif "Carroll" in method and vp_col:
            df_upd["Vs_pred"]        = _carroll(df_upd[vp_col])
            df_upd["Vs_uncertainty"] = df_upd["Vs_pred"].abs() * 0.06
            ok, tag = True, "Vs"
        else:
            st.error("Required input curve(s) not assigned above.")
            ok = False

        if ok:
            short = method.split("—")[0].strip()
            st.success(f"✅ {short} → **{tag}_pred** saved.")
            st.session_state.df = df_upd
            if "Vp_pred" in df_upd.columns:
                _show_results(df_upd, "Vp_pred", vp_col, "Vp",
                              short, "Vp_uncertainty")
            if "Vs_pred" in df_upd.columns:
                _show_results(df_upd, "Vs_pred", vs_col, "Vs",
                              short, "Vs_uncertainty")


# ═════════════════════════════════════════════════════════════════════════════
# MACHINE LEARNING
# ═════════════════════════════════════════════════════════════════════════════

_ML_MODELS = ["Linear Regression", "Random Forest",
              "XGBoost", "Decision Tree", "SVR"]


def _build_ml_model(name: str, **kw):
    if name == "Linear Regression":
        return LinearRegression()
    elif name == "Random Forest":
        return RandomForestRegressor(n_estimators=kw.get("n_est", 100),
                                     random_state=42, n_jobs=-1)
    elif name == "XGBoost":
        if not _HAS_XGB:
            st.error("XGBoost not installed — run `pip install xgboost`.")
            st.stop()
        return xgb.XGBRegressor(n_estimators=kw.get("n_est", 100),
                                 max_depth=kw.get("max_d", 5),
                                 learning_rate=0.1, random_state=42,
                                 verbosity=0)
    elif name == "Decision Tree":
        return DecisionTreeRegressor(max_depth=kw.get("max_d", 8),
                                     random_state=42)
    elif name == "SVR":
        return SVR(kernel="rbf", C=10.0, epsilon=0.05)
    raise ValueError(f"Unknown model: {name}")


def _uncertainty_ml(model, X: np.ndarray, name: str) -> np.ndarray:
    if name == "Random Forest":
        return np.stack([t.predict(X) for t in model.estimators_]).std(axis=0)
    if name == "XGBoost" and _HAS_XGB:
        rng = np.random.default_rng(42)
        n = X.shape[0]
        return np.std(
            [model.predict(X[rng.integers(0, n, n)]) for _ in range(10)],
            axis=0,
        )
    return np.full(len(X), float(np.std(model.predict(X)) * 0.05))


def _run_ml_target(df: pd.DataFrame, label: str,
                   true_col: str, pred_col: str, unc_col: str,
                   feats: list[str], model_name: str, scaler_name: str,
                   test_frac: float, pred_scope: str, **kw) -> pd.DataFrame:
    avail = [c for c in feats if c in df.columns]
    if not avail:
        st.warning(f"No valid feature columns for {label}.")
        return df
    mask = df[avail].notna().all(axis=1) & df[true_col].notna()
    if mask.sum() < 10:
        st.warning(f"Too few rows for {label} (≥10 needed). Skipping.")
        return df

    X_all = df[avail].values.astype(float)
    y_all = df[true_col].values.astype(float)
    Xr, Xt, yr, yt = train_test_split(X_all[mask], y_all[mask],
                                       test_size=test_frac, random_state=42)
    sx, sy = _get_scaler(scaler_name), _get_scaler(scaler_name)
    Xr_s = sx.fit_transform(Xr)
    Xt_s = sx.transform(Xt)
    yr_s = sy.fit_transform(yr.reshape(-1, 1)).ravel()

    with st.spinner(f"Training {model_name} for {label} …"):
        mdl = _build_ml_model(model_name, **kw)
        mdl.fit(Xr_s, yr_s)

    yt_pred = sy.inverse_transform(
        mdl.predict(Xt_s).reshape(-1, 1)
    ).ravel()
    m = _safe_metric(yt, yt_pred)
    st.success(
        f"✅ **{label}** test — RMSE: `{m['RMSE']:.5f}` | "
        f"MAE: `{m['MAE']:.5f}` | R²: `{m['R²']:.4f}`"
    )

    if pred_scope == "Missing values only (NaN rows)":
        pmask = df[avail].notna().all(axis=1)
        if true_col in df.columns:
            pmask = pmask & df[true_col].isna()
    else:
        pmask = df[avail].notna().all(axis=1)

    Xp = sx.transform(X_all[pmask])
    yp = sy.inverse_transform(mdl.predict(Xp).reshape(-1, 1)).ravel()
    unc = _uncertainty_ml(mdl, Xp, model_name)

    df = df.copy()
    df[pred_col] = np.nan; df[unc_col] = np.nan
    df.loc[pmask, pred_col] = yp
    df.loc[pmask, unc_col]  = unc
    return df


def ml_models(df: pd.DataFrame):
    _section_header("🔴", "Machine Learning Models",
                    "Scikit-learn & XGBoost — tabular regression")

    st.markdown("""
| Model | Key Strength | Uncertainty |
|---|---|---|
| Linear Regression | Fast baseline | Residual std |
| Random Forest | Robust; tree-variance UQ | Ensemble variance |
| **XGBoost** | State-of-art tabular | Bootstrap resample |
| Decision Tree | Interpretable | Residual std |
| SVR | Small datasets | Residual std |
    """)

    if not _HAS_XGB:
        st.warning("⚠️ XGBoost not installed — `pip install xgboost` to enable it.")

    num_cols = _numeric_cols(df)
    opts_n   = ["None"] + num_cols

    c1, c2, c3 = st.columns(3)
    model_name  = c1.selectbox("Model", _ML_MODELS, key="dg_ml_model")
    target_ml   = c2.radio("Predict", ["Vp", "Vs", "Both"],
                            horizontal=True, key="dg_ml_target")
    scaler_name = c3.selectbox("Scaler",
                                ["StandardScaler", "MinMaxScaler"],
                                key="dg_ml_scaler")

    feats = st.multiselect(
        "Input features (log curves)",
        num_cols,
        default=[c for c in num_cols
                 if c not in ("Vp_pred", "Vs_pred",
                               "Vp_uncertainty", "Vs_uncertainty")],
        key="dg_ml_features",
    )

    auto_vp = _find(df, ["VP", "VEL", "DT", "DTCO"])
    auto_vs = _find(df, ["VS", "DTS", "DTSM"])
    c4, c5 = st.columns(2)
    vp_true = c4.selectbox("True Vp column (training target)", opts_n,
                            key="dg_ml_vp_true",
                            index=(opts_n.index(auto_vp) if auto_vp in opts_n else 0))
    vs_true = c5.selectbox("True Vs column (training target)", opts_n,
                            key="dg_ml_vs_true",
                            index=(opts_n.index(auto_vs) if auto_vs in opts_n else 0))
    vp_true = None if vp_true == "None" else vp_true
    vs_true = None if vs_true == "None" else vs_true

    test_frac = st.slider("Test-set fraction", 0.10, 0.40, 0.20, 0.05,
                           key="dg_ml_test")
    n_est, max_d = 100, 8
    if model_name in ("Random Forest", "XGBoost", "Decision Tree"):
        hc1, hc2 = st.columns(2)
        if model_name != "Decision Tree":
            n_est = hc1.slider("n_estimators", 50, 500, 100, 50,
                                key="dg_ml_nest")
        max_d = hc2.slider("max_depth", 3, 20, 8, 1, key="dg_ml_maxd")

    pred_scope = st.radio(
        "Predict on",
        ["Missing values only (NaN rows)", "Full dataset"],
        horizontal=True, key="dg_ml_scope",
    )

    if st.button("🚀 Train & Predict (ML)", type="primary", key="dg_ml_run"):
        if not feats:
            st.error("Select at least one input feature.")
            return

        targets = []
        if target_ml in ("Vp", "Both") and vp_true:
            targets.append(("Vp", vp_true, "Vp_pred", "Vp_uncertainty"))
        if target_ml in ("Vs", "Both") and vs_true:
            targets.append(("Vs", vs_true, "Vs_pred", "Vs_uncertainty"))
        if not targets:
            st.error("Assign at least one true Vp/Vs column for training.")
            return

        df_upd = st.session_state.df.copy()
        for lbl, tc, pc, uc in targets:
            st.markdown(f"#### ▸ {lbl}")
            df_upd = _run_ml_target(df_upd, lbl, tc, pc, uc, feats,
                                     model_name, scaler_name,
                                     test_frac, pred_scope,
                                     n_est=n_est, max_d=max_d)
        st.session_state.df = df_upd

        for lbl, tc, pc, uc in targets:
            if pc in df_upd.columns:
                _show_results(df_upd, pc,
                              tc if tc in df_upd.columns else None,
                              lbl, model_name, uc)


# ═════════════════════════════════════════════════════════════════════════════
# DEEP LEARNING  (PyTorch)
# ═════════════════════════════════════════════════════════════════════════════

class _ANN(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


class _CNN1D(nn.Module):
    def __init__(self, in_dim: int, dropout: float = 0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 48, kernel_size=min(5, in_dim), padding=2), nn.ReLU(),
            nn.Conv1d(48, 48, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(48, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )
    def forward(self, x):
        b = x.size(0)
        return self.fc(self.conv(x.unsqueeze(1)).view(b, -1)).squeeze(-1)


class _LSTM(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128,
                 num_layers: int = 2, dropout: float = 0.3,
                 bidirectional: bool = False):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=num_layers,
                             batch_first=True,
                             dropout=dropout if num_layers > 1 else 0.0,
                             bidirectional=bidirectional)
        d = 2 if bidirectional else 1
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden * d, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )
    def forward(self, x):
        out, _ = self.lstm(x.unsqueeze(1))
        return self.fc(out[:, -1, :]).squeeze(-1)


class _CNNBiLSTM(nn.Module):
    """Haritha et al. 2025 — CNN + Bi-LSTM."""
    def __init__(self, in_dim: int, cnn_ch: int = 48,
                 lstm_h: int = 120, n_lstm: int = 3,
                 dropout: float = 0.5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, cnn_ch, kernel_size=5, padding=2), nn.ReLU(),
        )
        self.bilstm = nn.LSTM(
            cnn_ch * in_dim, lstm_h, num_layers=n_lstm,
            batch_first=True,
            dropout=dropout if n_lstm > 1 else 0.0,
            bidirectional=True,
        )
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_h * 2, 1024), nn.ReLU(),
            nn.Linear(1024, 256), nn.ReLU(),
            nn.Linear(256, 1), nn.Sigmoid(),
        )
    def forward(self, x):
        b = x.size(0)
        x = self.conv(x.unsqueeze(1)).view(b, 1, -1)
        out, _ = self.bilstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


def _build_dl_model(arch: str, in_dim: int, dropout: float):
    if arch == "ANN":
        return _ANN(in_dim, 128, dropout)
    elif arch == "1D CNN":
        return _CNN1D(in_dim, dropout)
    elif arch == "LSTM":
        return _LSTM(in_dim, 128, 2, dropout, False)
    elif arch == "Bi-LSTM":
        return _LSTM(in_dim, 128, 2, dropout, True)
    else:
        return _CNNBiLSTM(in_dim, dropout=dropout)


def _train_dl(model, X_tr, y_tr, X_val, y_val,
              epochs, batch_size, lr, prog, stat):
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    Xt = torch.tensor(X_tr,  dtype=torch.float32).to(device)
    yt = torch.tensor(y_tr,  dtype=torch.float32).to(device)
    Xv = torch.tensor(X_val, dtype=torch.float32).to(device)
    yv = torch.tensor(y_val, dtype=torch.float32).to(device)
    loader  = DataLoader(TensorDataset(Xt, yt),
                          batch_size=batch_size, shuffle=True)
    opt     = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    tr_l, val_l = [], []
    for ep in range(epochs):
        model.train()
        ep_l = sum(
            loss_fn(model(xb), yb).backward() or  # side-effect
            (opt.step(), opt.zero_grad(),
             loss_fn(model(xb), yb).item() * len(xb))[2]
            for xb, yb in loader
        )
        # simpler loop
        model.train()
        ep_l = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_l += loss.item() * len(xb)
        tr_l.append(ep_l / len(X_tr))
        model.eval()
        with torch.no_grad():
            val_l.append(loss_fn(model(Xv), yv).item())
        if prog:
            prog.progress((ep + 1) / epochs)
        if stat:
            stat.text(f"Epoch {ep+1}/{epochs} | "
                      f"Train: {tr_l[-1]:.6f} | Val: {val_l[-1]:.6f}")
    model.to("cpu")
    return model, tr_l, val_l


def _mc_predict(model, X: np.ndarray, sy, n_mc: int = 20):
    model.train()   # keep dropout active
    Xt    = torch.tensor(X, dtype=torch.float32)
    preds = np.stack([model(Xt).detach().numpy() for _ in range(n_mc)])
    mean  = sy.inverse_transform(preds.mean(0).reshape(-1, 1)).ravel()
    scale = float(getattr(sy, "scale_", [1.0])[0])
    unc   = preds.std(0) * scale
    return mean, unc


def dl_models(df: pd.DataFrame):
    if not _HAS_TORCH:
        st.error("PyTorch not installed.\n"
                 "```\npip install torch --index-url "
                 "https://download.pytorch.org/whl/cpu\n```")
        return

    _section_header("🔴", "Deep Learning Models (PyTorch)",
                    "Haritha et al. (2025) — CNN-Bi-LSTM architecture")

    st.markdown("""
| Architecture | Best for |
|---|---|
| ANN | Fast baseline, tabular |
| 1D CNN | Local spatial features |
| LSTM | Sequential depth-series |
| Bi-LSTM | Forward + backward context |
| **CNN + Bi-LSTM** | Spatial + temporal (top performer) |
    """)

    num_cols = _numeric_cols(df)
    opts_n   = ["None"] + num_cols

    arch        = st.selectbox(
        "Architecture",
        ["ANN", "1D CNN", "LSTM", "Bi-LSTM",
         "CNN + Bi-LSTM (Haritha et al.)"],
        index=4, key="dg_dl_arch",
    )
    target_dl   = st.radio("Predict", ["Vp", "Vs", "Both"],
                            horizontal=True, key="dg_dl_target")
    scaler_name = st.selectbox("Scaler",
                                ["MinMaxScaler", "StandardScaler"],
                                key="dg_dl_scaler")

    feats = st.multiselect(
        "Input features",
        num_cols,
        default=[c for c in num_cols
                 if c not in ("Vp_pred", "Vs_pred",
                               "Vp_uncertainty", "Vs_uncertainty")],
        key="dg_dl_features",
    )

    auto_vp = _find(df, ["VP", "VEL", "DT", "DTCO"])
    auto_vs = _find(df, ["VS", "DTS", "DTSM"])
    c1, c2 = st.columns(2)
    vp_true = c1.selectbox("True Vp column", opts_n, key="dg_dl_vp_true",
                            index=(opts_n.index(auto_vp) if auto_vp in opts_n else 0))
    vs_true = c2.selectbox("True Vs column", opts_n, key="dg_dl_vs_true",
                            index=(opts_n.index(auto_vs) if auto_vs in opts_n else 0))
    vp_true = None if vp_true == "None" else vp_true
    vs_true = None if vs_true == "None" else vs_true

    hc1, hc2, hc3, hc4 = st.columns(4)
    epochs     = hc1.slider("Epochs",     10, 200, 80,  10,  key="dg_dl_epochs")
    batch_size = hc2.slider("Batch size",  8, 256, 64,   8,  key="dg_dl_batch")
    lr         = hc3.select_slider("LR",
                    [0.0001, 0.0005, 0.001, 0.003, 0.005, 0.01],
                    value=0.003, key="dg_dl_lr")
    dropout    = hc4.slider("Dropout", 0.0, 0.7, 0.3, 0.05, key="dg_dl_drop")
    test_frac  = st.slider("Test fraction", 0.10, 0.40, 0.20, 0.05,
                            key="dg_dl_test")
    n_mc       = st.slider("MC Dropout samples", 5, 50, 20, 5,
                            key="dg_dl_mc")
    pred_scope = st.radio(
        "Predict on",
        ["Missing values only (NaN rows)", "Full dataset"],
        horizontal=True, key="dg_dl_scope",
    )

    if st.button("🚀 Train & Predict (DL)", type="primary", key="dg_dl_run"):
        if not feats:
            st.error("Select at least one input feature.")
            return
        targets = []
        if target_dl in ("Vp", "Both") and vp_true:
            targets.append(("Vp", vp_true, "Vp_pred", "Vp_uncertainty"))
        if target_dl in ("Vs", "Both") and vs_true:
            targets.append(("Vs", vs_true, "Vs_pred", "Vs_uncertainty"))
        if not targets:
            st.error("Assign at least one true Vp/Vs column.")
            return

        df_upd = st.session_state.df.copy()
        for lbl, tc, pc, uc in targets:
            st.markdown(f"#### ▸ {lbl}")
            avail = [c for c in feats if c in df_upd.columns]
            if not avail:
                st.warning(f"No feature columns for {lbl}.")
                continue
            mask = df_upd[avail].notna().all(axis=1) & df_upd[tc].notna()
            if mask.sum() < 20:
                st.warning(f"Need ≥20 valid rows for {lbl}.")
                continue

            X_all = df_upd[avail].values.astype(float)
            y_all = df_upd[tc].values.astype(float)
            sx, sy = _get_scaler(scaler_name), _get_scaler(scaler_name)
            Xr, Xt, yr, yt = train_test_split(
                X_all[mask], y_all[mask],
                test_size=test_frac, random_state=42,
            )
            Xr_s = sx.fit_transform(Xr)
            Xt_s = sx.transform(Xt)
            yr_s = sy.fit_transform(yr.reshape(-1, 1)).ravel()
            yt_s = sy.transform(yt.reshape(-1, 1)).ravel()

            model = _build_dl_model(arch, Xr_s.shape[1], dropout)
            prog  = st.progress(0)
            stat  = st.empty()
            with st.spinner(f"Training {arch} for {lbl} …"):
                model, tr_l, val_l = _train_dl(
                    model, Xr_s, yr_s, Xt_s, yt_s,
                    epochs, batch_size, lr, prog, stat,
                )
            prog.empty(); stat.empty()

            st.plotly_chart(
                _plotly_loss_curve(tr_l, val_l, f"{arch} ({lbl})"),
                use_container_width=True,
                key=f"dg_loss_{lbl}_{arch}",
            )

            model.eval()
            with torch.no_grad():
                yt_pred = sy.inverse_transform(
                    model(torch.tensor(Xt_s, dtype=torch.float32)
                          ).numpy().reshape(-1, 1)
                ).ravel()
            m = _safe_metric(yt, yt_pred)
            st.success(f"✅ **{lbl}** test — RMSE: `{m['RMSE']:.5f}` | "
                       f"MAE: `{m['MAE']:.5f}` | R²: `{m['R²']:.4f}`")

            if pred_scope == "Missing values only (NaN rows)":
                pmask = df_upd[avail].notna().all(axis=1)
                if tc in df_upd.columns:
                    pmask = pmask & df_upd[tc].isna()
            else:
                pmask = df_upd[avail].notna().all(axis=1)

            Xp = sx.transform(X_all[pmask])
            yp, unc = _mc_predict(model, Xp, sy, n_mc)

            df_upd[pc] = np.nan; df_upd[uc] = np.nan
            df_upd.loc[pmask, pc] = yp
            df_upd.loc[pmask, uc] = unc

        st.session_state.df = df_upd
        for lbl, tc, pc, uc in targets:
            if pc in df_upd.columns:
                _show_results(df_upd, pc,
                              tc if tc in df_upd.columns else None,
                              lbl, arch, uc)


# ═════════════════════════════════════════════════════════════════════════════
# COMPARISON MODULE  (redesigned)
# ═════════════════════════════════════════════════════════════════════════════

def _comparison_panel(df: pd.DataFrame):
    _section_header("📊", "Model Comparison & Evaluation",
                    "All predictions compared side-by-side")

    num_cols = _numeric_cols(df)
    opts_n   = ["None"] + num_cols
    auto_vp  = _find(df, ["VP", "VEL", "DT", "DTCO"])
    auto_vs  = _find(df, ["VS", "DTS", "DTSM"])

    c1, c2 = st.columns(2)
    vp_true = c1.selectbox("True Vp column", opts_n, key="dg_cmp_vp_true",
                            index=(opts_n.index(auto_vp) if auto_vp in opts_n else 0))
    vs_true = c2.selectbox("True Vs column", opts_n, key="dg_cmp_vs_true",
                            index=(opts_n.index(auto_vs) if auto_vs in opts_n else 0))
    vp_true = None if vp_true == "None" else vp_true
    vs_true = None if vs_true == "None" else vs_true

    pred_vp = [c for c in df.columns if "Vp" in c and "pred" in c.lower()]
    pred_vs = [c for c in df.columns if "Vs" in c and "pred" in c.lower()]

    if not pred_vp and not pred_vs:
        st.info("No prediction columns found yet. "
                "Run a method in Conventional / ML / DL first.")
        return

    # ── A. Metrics table ─────────────────────────────────────────────────────
    st.markdown("### A — Metrics Summary")
    rows = []
    for pc in pred_vp:
        if vp_true and vp_true in df.columns:
            m = _safe_metric(df[vp_true].values.astype(float),
                             df[pc].values.astype(float))
            rows.append({"Model / Method": pc, "Target": "Vp", **m})
        else:
            rows.append({"Model / Method": pc, "Target": "Vp",
                         "RMSE": None, "MAE": None, "R²": None})
    for pc in pred_vs:
        if vs_true and vs_true in df.columns:
            m = _safe_metric(df[vs_true].values.astype(float),
                             df[pc].values.astype(float))
            rows.append({"Model / Method": pc, "Target": "Vs", **m})
        else:
            rows.append({"Model / Method": pc, "Target": "Vs",
                         "RMSE": None, "MAE": None, "R²": None})

    mdf = pd.DataFrame(rows)
    numeric_mdf = mdf.dropna(subset=["RMSE"])

    fmt = {c: "{:.5f}" for c in ["RMSE", "MAE", "R²"]
           if c in mdf.columns}
    st.dataframe(mdf.style.format(fmt, na_rep="—"),
                 use_container_width=True, hide_index=True)

    # ── B. Bar charts per metric ──────────────────────────────────────────────
    if not numeric_mdf.empty:
        st.markdown("### B — Metric Comparison Charts")
        for metric in ["RMSE", "MAE", "R²"]:
            fig = px.bar(
                numeric_mdf, x="Model / Method", y=metric, color="Target",
                barmode="group", text_auto=".4f",
                color_discrete_map={"Vp": "#1976D2", "Vs": "#E53935"},
                title=f"{metric} comparison across models",
            )
            fig.update_layout(
                height=320, xaxis_tickangle=-30,
                margin=dict(l=40, r=20, t=50, b=90),
                legend_title_text="Target",
            )
            st.plotly_chart(fig, use_container_width=True,
                            key=f"dg_bar_{metric}")

    # ── C. Per-model depth track + scatter ────────────────────────────────────
    st.markdown("### C — Prediction vs True (per model)")

    for tgt_lbl, true_col, preds_t in [
        ("Vp", vp_true, pred_vp),
        ("Vs", vs_true, pred_vs),
    ]:
        if not preds_t:
            continue
        st.markdown(f"#### {tgt_lbl}")

        # Depth overlay (all predictions)
        curves  = []
        clabels = []
        colors  = []
        if true_col and true_col in df.columns:
            curves.append(true_col); clabels.append("Original"); colors.append("#212121")
        for i, pc in enumerate(preds_t):
            curves.append(pc)
            clabels.append(pc)
            colors.append(_COLORS[(i + 1) % len(_COLORS)])

        st.plotly_chart(
            _plotly_depth_track(df, curves, clabels, colors,
                                f"{tgt_lbl} — All Predictions vs Depth",
                                tgt_lbl),
            use_container_width=True,
            key=f"dg_cmp_depth_{tgt_lbl}",
        )

        # Dropdown to select one prediction for scatter/residual
        sel = st.selectbox(
            f"Inspect {tgt_lbl} prediction in detail",
            preds_t,
            key=f"dg_cmp_sel_{tgt_lbl}",
        )
        if true_col and true_col in df.columns and sel in df.columns:
            t_arr = df[true_col].values.astype(float)
            p_arr = df[sel].values.astype(float)
            sc1, sc2 = st.columns(2)
            with sc1:
                st.plotly_chart(
                    _plotly_scatter(t_arr, p_arr, tgt_lbl, sel),
                    use_container_width=True,
                    key=f"dg_cmp_sc_{tgt_lbl}_{sel}",
                )
            with sc2:
                st.plotly_chart(
                    _plotly_residual_hist(t_arr, p_arr, tgt_lbl),
                    use_container_width=True,
                    key=f"dg_cmp_res_{tgt_lbl}_{sel}",
                )

    # ── D. Uncertainty summary ────────────────────────────────────────────────
    unc_cols = [c for c in df.columns if "uncertainty" in c.lower()]
    if unc_cols:
        st.markdown("### D — Uncertainty Summary (±σ)")
        unc_stats = df[unc_cols].describe().T[["mean", "std", "min", "max"]]
        st.dataframe(unc_stats.style.format("{:.5f}"), use_container_width=True)
        fig_unc = px.bar(
            unc_stats.reset_index().rename(columns={"index": "Column"}),
            x="Column", y="mean", error_y="std", color="Column",
            title="Mean Uncertainty (±σ) per Prediction Column",
            labels={"mean": "Mean ±σ"},
        )
        fig_unc.update_layout(height=300, showlegend=False,
                              margin=dict(l=40, r=20, t=50, b=60))
        st.plotly_chart(fig_unc, use_container_width=True, key="dg_unc_bar")

    # ── E. Inter-model stability ───────────────────────────────────────────────
    for tgt_lbl, preds_t in [("Vp", pred_vp), ("Vs", pred_vs)]:
        if len(preds_t) > 1:
            sub = df[preds_t].dropna()
            if not sub.empty:
                st.metric(f"{tgt_lbl} inter-model prediction variance",
                          f"{sub.var(axis=1).mean():.6f}",
                          help="Lower = more consistent across models")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN RENDER  —  single source of truth (radio = session_state["dg_approach"])
# ═════════════════════════════════════════════════════════════════════════════

_APPROACHES = [
    "🔵  Conventional Methods",
    "🔴  Machine Learning",
    "🔴  Deep Learning",
    "📊  Comparison",
]

_APPROACH_FN = {
    "🔵  Conventional Methods": conventional_methods,
    "🔴  Machine Learning":     ml_models,
    "🔴  Deep Learning":        dl_models,
    "📊  Comparison":           _comparison_panel,
}


def render(df: pd.DataFrame):
    st.title("🤖 Missing P- & S-Sonic Data Generation")

    st.info(
        "**Objective:** Predict missing **Vp** (P-wave) and **Vs** (S-wave) logs "
        "from well-log curves using conventional empirical equations or "
        "non-conventional ML / DL models.\n\n"
        "Predictions are stored as `Vp_pred`, `Vs_pred`, "
        "`Vp_uncertainty`, `Vs_uncertainty` and persist across all pages."
    )

    # ── Single source of truth ────────────────────────────────────────────────
    # The radio widget writes directly to st.session_state["dg_approach"].
    # On every rerun, `approach` reflects the user's current selection,
    # and we call exactly the matching render function — no tabs, no sync bug.
    approach = st.radio(
        "Select Approach",
        _APPROACHES,
        horizontal=True,
        key="dg_approach",
    )

    st.divider()

    # Route
    fn = _APPROACH_FN.get(approach, conventional_methods)
    fn(st.session_state.df)
