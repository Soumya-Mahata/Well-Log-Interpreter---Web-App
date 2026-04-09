"""
dg_unconventional.py  —  Non-Conventional ML & DL Methods
==========================================================
Machine Learning models (scikit-learn + XGBoost):
  Linear Regression | Random Forest | XGBoost | Decision Tree | SVR

Deep Learning models (PyTorch):
  ANN | 1D CNN | LSTM | Bi-LSTM | CNN + Bi-LSTM (Haritha et al. 2025)

Uncertainty quantification:
  ML  → ensemble variance (RF) / bootstrap (XGBoost) / residual std
  DL  → Monte Carlo Dropout (n_mc forward passes with dropout ON)

Public entry points:
    from dg_unconventional import render_ml, render_dl
    render_ml(df)
    render_dl(df)
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

from dg_utils import (
    find_col, numeric_cols, get_scaler, safe_metric,
    section_header, show_results, plotly_loss_curve,
)

# ── Optional: XGBoost ─────────────────────────────────────────────────────────
try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

# ── Optional: PyTorch ─────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    import types as _t
    torch = None  # type: ignore
    nn = _t.SimpleNamespace(
        Module=object, Linear=None, ReLU=None, Dropout=None,
        Sequential=None, LSTM=None, Conv1d=None,
        AdaptiveMaxPool1d=None, MSELoss=None,
    )
    DataLoader = TensorDataset = None  # type: ignore


# ═════════════════════════════════════════════════════════════════════════════
# ── MACHINE LEARNING ─────────────────────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════

ML_MODEL_LIST = ["Linear Regression", "Random Forest",
                 "XGBoost", "Decision Tree", "SVR"]


def _build_ml_model(name: str, n_est: int = 100, max_d: int = 8):
    if name == "Linear Regression":
        return LinearRegression()
    if name == "Random Forest":
        return RandomForestRegressor(n_estimators=n_est,
                                     random_state=42, n_jobs=-1)
    if name == "XGBoost":
        if not _HAS_XGB:
            st.error("XGBoost not installed — run `pip install xgboost`.")
            st.stop()
        return xgb.XGBRegressor(n_estimators=n_est, max_depth=max_d,
                                 learning_rate=0.1, random_state=42,
                                 verbosity=0)
    if name == "Decision Tree":
        return DecisionTreeRegressor(max_depth=max_d, random_state=42)
    if name == "SVR":
        return SVR(kernel="rbf", C=10.0, epsilon=0.05)
    raise ValueError(f"Unknown ML model: {name}")


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
    # fallback: 5 % of prediction std
    return np.full(len(X), float(np.std(model.predict(X)) * 0.05))


def _train_and_predict_ml(df: pd.DataFrame,
                           label: str, true_col: str,
                           pred_col: str, unc_col: str,
                           feats: list[str],
                           model_name: str,
                           scaler_name: str,
                           test_frac: float,
                           pred_scope: str,
                           n_est: int = 100,
                           max_d: int = 8) -> pd.DataFrame:
    """Train one ML model for one target (Vp or Vs) and write predictions."""
    avail = [c for c in feats if c in df.columns]
    if not avail:
        st.warning(f"No valid feature columns found for **{label}**.")
        return df

    mask = df[avail].notna().all(axis=1) & df[true_col].notna()
    if mask.sum() < 10:
        st.warning(f"Too few labelled rows for **{label}** (≥ 10 needed). Skipping.")
        return df

    X_all = df[avail].values.astype(float)
    y_all = df[true_col].values.astype(float)

    Xr, Xt, yr, yt = train_test_split(
        X_all[mask], y_all[mask], test_size=test_frac, random_state=42,
    )
    sx, sy = get_scaler(scaler_name), get_scaler(scaler_name)
    Xr_s = sx.fit_transform(Xr)
    Xt_s = sx.transform(Xt)
    yr_s = sy.fit_transform(yr.reshape(-1, 1)).ravel()

    with st.spinner(f"Training **{model_name}** for {label} …"):
        mdl = _build_ml_model(model_name, n_est, max_d)
        mdl.fit(Xr_s, yr_s)

    yt_pred = sy.inverse_transform(
        mdl.predict(Xt_s).reshape(-1, 1)
    ).ravel()
    m = safe_metric(yt, yt_pred)
    st.success(
        f"✅ **{label}** test — RMSE: `{m['RMSE']:.5f}` | "
        f"MAE: `{m['MAE']:.5f}` | R²: `{m['R²']:.4f}`"
    )

    # Prediction mask
    if pred_scope == "Missing values only (NaN rows)":
        pmask = df[avail].notna().all(axis=1)
        if true_col in df.columns:
            pmask = pmask & df[true_col].isna()
    else:
        pmask = df[avail].notna().all(axis=1)

    Xp  = sx.transform(X_all[pmask])
    yp  = sy.inverse_transform(mdl.predict(Xp).reshape(-1, 1)).ravel()
    unc = _uncertainty_ml(mdl, Xp, model_name)

    df = df.copy()
    df[pred_col] = np.nan
    df[unc_col]  = np.nan
    df.loc[pmask, pred_col] = yp
    df.loc[pmask, unc_col]  = unc
    return df


def render_ml(df: pd.DataFrame) -> None:
    section_header("🔴", "Machine Learning Models",
                   "Scikit-learn & XGBoost — tabular regression with uncertainty")

    st.markdown("""
| Model | Key Strength | Uncertainty Method |
|---|---|---|
| Linear Regression | Fast baseline | Residual std (±5 %) |
| Random Forest | Robust; interpretable UQ | Ensemble tree variance |
| **XGBoost** | State-of-art tabular | Bootstrap resample |
| Decision Tree | Interpretable rules | Residual std |
| SVR | Small / noisy datasets | Residual std |
    """)

    if not _HAS_XGB:
        st.warning("⚠️ XGBoost not installed — run `pip install xgboost` to enable it.")

    num  = numeric_cols(df)
    opts = ["None"] + num

    # ── Controls ──────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    model_name  = c1.selectbox("Model", ML_MODEL_LIST, key="dg_ml_model")
    target_ml   = c2.radio("Predict", ["Vp", "Vs", "Both"],
                            horizontal=True, key="dg_ml_target")
    scaler_name = c3.selectbox("Scaler",
                                ["StandardScaler", "MinMaxScaler"],
                                key="dg_ml_scaler")

    feats = st.multiselect(
        "Input features (log curves)",
        num,
        default=[c for c in num
                 if c not in ("Vp_pred", "Vs_pred",
                               "Vp_uncertainty", "Vs_uncertainty")],
        key="dg_ml_features",
    )

    auto_vp = find_col(df, ["VP", "VEL", "DT", "DTCO"])
    auto_vs = find_col(df, ["VS", "DTS", "DTSM"])
    c4, c5 = st.columns(2)
    vp_true = c4.selectbox(
        "True Vp column (training target)", opts, key="dg_ml_vp_true",
        index=(opts.index(auto_vp) if auto_vp in opts else 0),
    )
    vs_true = c5.selectbox(
        "True Vs column (training target)", opts, key="dg_ml_vs_true",
        index=(opts.index(auto_vs) if auto_vs in opts else 0),
    )
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

    if not st.button("🚀 Train & Predict", type="primary", key="dg_ml_run"):
        return

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
        st.markdown(f"##### ▸ {lbl}")
        df_upd = _train_and_predict_ml(
            df_upd, lbl, tc, pc, uc, feats,
            model_name, scaler_name, test_frac, pred_scope,
            n_est=n_est, max_d=max_d,
        )
    st.session_state.df = df_upd

    for lbl, tc, pc, uc in targets:
        if pc in df_upd.columns:
            show_results(df_upd, pc,
                         true_col=(tc if tc in df_upd.columns else None),
                         label=lbl, model_name=model_name,
                         unc_col=uc, key_prefix=f"ml_{model_name}")


# ═════════════════════════════════════════════════════════════════════════════
# ── DEEP LEARNING  (PyTorch) ──────────────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════

# ── Model architectures ──────────────────────────────────────────────────────

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
    """
    Convolutional Bidirectional LSTM — Haritha et al. (2025)
    Journal of Applied Geophysics 233, 105628.
    Architecture: Conv1D → Bi-LSTM × 3 → FC × 3 → Sigmoid output
    """
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
    if arch == "1D CNN":
        return _CNN1D(in_dim, dropout)
    if arch == "LSTM":
        return _LSTM(in_dim, 128, 2, dropout, bidirectional=False)
    if arch == "Bi-LSTM":
        return _LSTM(in_dim, 128, 2, dropout, bidirectional=True)
    # CNN + Bi-LSTM (default / best)
    return _CNNBiLSTM(in_dim, dropout=dropout)


# ── Training loop ─────────────────────────────────────────────────────────────

def _train_dl(model, X_tr, y_tr, X_val, y_val,
              epochs: int, batch_size: int, lr: float,
              prog, stat):
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
                      f"Train MSE: {tr_l[-1]:.6f} | Val MSE: {val_l[-1]:.6f}")

    model.to("cpu")
    return model, tr_l, val_l


def _mc_predict(model, X: np.ndarray, sy, n_mc: int = 20):
    """Monte Carlo Dropout: keep dropout ON for n_mc passes → mean ± std."""
    model.train()
    Xt    = torch.tensor(X, dtype=torch.float32)
    preds = np.stack([model(Xt).detach().numpy() for _ in range(n_mc)])
    mean  = sy.inverse_transform(preds.mean(0).reshape(-1, 1)).ravel()
    scale = float(getattr(sy, "scale_", [1.0])[0])
    unc   = preds.std(0) * scale
    return mean, unc


# ── Streamlit render ──────────────────────────────────────────────────────────

def render_dl(df: pd.DataFrame) -> None:
    if not _HAS_TORCH:
        st.error(
            "PyTorch is not installed.\n\n"
            "```\npip install torch --index-url "
            "https://download.pytorch.org/whl/cpu\n```"
        )
        return

    section_header("🔴", "Deep Learning Models (PyTorch)",
                   "Haritha et al. (2025) — CNN-Bi-LSTM architecture reference")

    st.markdown("""
| Architecture | Best for | Uncertainty |
|---|---|---|
| ANN | Fast baseline, tabular | MC Dropout |
| 1D CNN | Local spatial features | MC Dropout |
| LSTM | Sequential depth-series | MC Dropout |
| Bi-LSTM | Forward + backward context | MC Dropout |
| **CNN + Bi-LSTM** | Spatial + temporal (top performer) | MC Dropout |
    """)

    num  = numeric_cols(df)
    opts = ["None"] + num

    # ── Controls ──────────────────────────────────────────────────────────────
    arch = st.selectbox(
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
        num,
        default=[c for c in num
                 if c not in ("Vp_pred", "Vs_pred",
                               "Vp_uncertainty", "Vs_uncertainty")],
        key="dg_dl_features",
    )

    auto_vp = find_col(df, ["VP", "VEL", "DT", "DTCO"])
    auto_vs = find_col(df, ["VS", "DTS", "DTSM"])
    c1, c2 = st.columns(2)
    vp_true = c1.selectbox("True Vp column", opts, key="dg_dl_vp_true",
                            index=(opts.index(auto_vp) if auto_vp in opts else 0))
    vs_true = c2.selectbox("True Vs column", opts, key="dg_dl_vs_true",
                            index=(opts.index(auto_vs) if auto_vs in opts else 0))
    vp_true = None if vp_true == "None" else vp_true
    vs_true = None if vs_true == "None" else vs_true

    hc1, hc2, hc3, hc4 = st.columns(4)
    epochs     = hc1.slider("Epochs",     10, 200, 80,  10,  key="dg_dl_epochs")
    batch_size = hc2.slider("Batch size",  8, 256, 64,   8,  key="dg_dl_batch")
    lr         = hc3.select_slider(
        "Learning rate",
        [0.0001, 0.0005, 0.001, 0.003, 0.005, 0.01],
        value=0.003, key="dg_dl_lr",
    )
    dropout    = hc4.slider("Dropout", 0.0, 0.7, 0.3, 0.05, key="dg_dl_drop")
    test_frac  = st.slider("Test fraction", 0.10, 0.40, 0.20, 0.05,
                            key="dg_dl_test")
    n_mc       = st.slider("MC Dropout samples (uncertainty)", 5, 50, 20, 5,
                            key="dg_dl_mc")
    pred_scope = st.radio(
        "Predict on",
        ["Missing values only (NaN rows)", "Full dataset"],
        horizontal=True, key="dg_dl_scope",
    )

    if not st.button("🚀 Train & Predict", type="primary", key="dg_dl_run"):
        return

    if not feats:
        st.error("Select at least one input feature.")
        return

    targets = []
    if target_dl in ("Vp", "Both") and vp_true:
        targets.append(("Vp", vp_true, "Vp_pred", "Vp_uncertainty"))
    if target_dl in ("Vs", "Both") and vs_true:
        targets.append(("Vs", vs_true, "Vs_pred", "Vs_uncertainty"))
    if not targets:
        st.error("Assign at least one true Vp/Vs column for training.")
        return

    df_upd = st.session_state.df.copy()

    for lbl, tc, pc, uc in targets:
        st.markdown(f"##### ▸ {lbl}")
        avail = [c for c in feats if c in df_upd.columns]
        if not avail:
            st.warning(f"No feature columns for {lbl}.")
            continue

        mask = df_upd[avail].notna().all(axis=1) & df_upd[tc].notna()
        if mask.sum() < 20:
            st.warning(f"Need ≥ 20 labelled rows for {lbl}. Skipping.")
            continue

        X_all = df_upd[avail].values.astype(float)
        y_all = df_upd[tc].values.astype(float)
        sx, sy = get_scaler(scaler_name), get_scaler(scaler_name)

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

        with st.spinner(f"Training **{arch}** for {lbl} …"):
            model, tr_l, val_l = _train_dl(
                model, Xr_s, yr_s, Xt_s, yt_s,
                epochs, batch_size, lr, prog, stat,
            )
        prog.empty()
        stat.empty()

        # Loss curve
        st.plotly_chart(
            plotly_loss_curve(tr_l, val_l, f"{arch} ({lbl})"),
            use_container_width=True,
            key=f"dg_dl_loss_{lbl}_{arch}",
        )

        # Test metrics
        model.eval()
        with torch.no_grad():
            yt_pred = sy.inverse_transform(
                model(torch.tensor(Xt_s, dtype=torch.float32)
                      ).numpy().reshape(-1, 1)
            ).ravel()
        m = safe_metric(yt, yt_pred)
        st.success(
            f"✅ **{lbl}** test — RMSE: `{m['RMSE']:.5f}` | "
            f"MAE: `{m['MAE']:.5f}` | R²: `{m['R²']:.4f}`"
        )

        # MC Dropout prediction
        if pred_scope == "Missing values only (NaN rows)":
            pmask = df_upd[avail].notna().all(axis=1)
            if tc in df_upd.columns:
                pmask = pmask & df_upd[tc].isna()
        else:
            pmask = df_upd[avail].notna().all(axis=1)

        Xp = sx.transform(X_all[pmask])
        yp, unc = _mc_predict(model, Xp, sy, n_mc=n_mc)

        df_upd[pc] = np.nan
        df_upd[uc] = np.nan
        df_upd.loc[pmask, pc] = yp
        df_upd.loc[pmask, uc] = unc

    st.session_state.df = df_upd

    for lbl, tc, pc, uc in targets:
        if pc in df_upd.columns:
            show_results(df_upd, pc,
                         true_col=(tc if tc in df_upd.columns else None),
                         label=lbl, model_name=arch,
                         unc_col=uc, key_prefix=f"dl_{arch}")
