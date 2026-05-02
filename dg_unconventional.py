"""
dg_unconventional.py — Unconventional Missing Data Generation
=============================================================

Guided workflow:
  1. Select Target  (DTC / DTS)
  2. Select Model
  3. View Model Theory
  4. View Required Input Logs
  5. Map Raw Logs → backend computes ALL engineered features
  6. Predict → store in session state → visualise

MODELS SUPPORTED (v4):
----------------------
1. Random Forest   — random_forest_{dtc/dts}.pkl
2. XGBoost         — xgboost_{dtc/dts}.pkl
3. CNN-BiLSTM      — cnn_bilstm_{dtc/dts}.pkl

FEATURE SET (5 features, identical for all models):
  [GR, Vsh, RHOB, NPHI, RT_log]
  - GR  : raw gamma ray (mapped by user)
  - Vsh : computed in backend from GR via p5-p95 per-well normalisation
  - RHOB: bulk density (mapped by user)
  - NPHI: neutron porosity (mapped by user)
  - RT_log: log10(RT) — computed in backend from the user-mapped RT column

PER-MODEL PREDICTION COLUMNS:
  Each model writes to <target>_<model_key>_pred (e.g. DTC_xgboost_pred).
  Prevents overwriting and enables proper multi-model comparison.
"""

from __future__ import annotations

import os
import pickle
import warnings

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

from dg_utils import (
    find_col,
    numeric_cols,
    safe_metric,
    section_header,
    show_results,
)

# TorchWrapper and CNNBiLSTMModel are imported from torch_models.py so that
# pickle resolves the class as  torch_models.TorchWrapper  both when the .pkl
# is saved in Colab and when it is loaded here in Streamlit.
# Without this shared module, pickle raises:
#   "Can't get attribute 'TorchWrapper' on <module 'main' from '...main.py'>"
try:
    from torch_models import TorchWrapper, CNNBiLSTMModel  # noqa: F401
except ImportError:
    TorchWrapper   = None  # type: ignore  — CNN-BiLSTM will error at load time
    CNNBiLSTMModel = None  # type: ignore

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS  (must mirror sonic_training_colab.ipynb)
# ─────────────────────────────────────────────────────────────────────────────

_BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
_MODEL_DIR = os.path.join(_BASE_DIR, "models")

MODEL_OPTIONS: list[str] = [
    "Random Forest",
    "XGBoost",
    "CNN-BiLSTM",
]

_MODEL_KEYS: dict[str, str] = {
    "Random Forest": "random_forest",
    "XGBoost":       "xgboost",
    "CNN-BiLSTM":    "cnn_bilstm",
}

_DL_MODELS: set[str] = {"CNN-BiLSTM"}

# All logs shown in UI (GR always shown so Vsh can be derived)
# User maps 4 raw logs. Backend computes Vsh (from GR) and RT_log (from RT).
_RAW_LOGS: list[str] = ["GR", "RHOB", "NPHI", "RT"]

# ── Feature orders — MUST match training notebook ─────────────────────────────
# ML: 4 features   DL: 5 features
# 5 features — identical for all models (ML and DL).
# GR + Vsh together are NOT redundant: GR = absolute gamma magnitude,
# Vsh = well-normalised relative shale fraction (0-1 per-well).
# RT_log = log10(RT) — computed in backend from user-mapped RT column.
_ML_FEATURES: list[str] = ["GR", "Vsh", "RHOB", "NPHI", "RT_log"]
_DL_FEATURES: list[str] = ["GR", "Vsh", "RHOB", "NPHI", "RT_log"]

_MODEL_THEORY: dict[str, str] = {
    "Random Forest": (
        "**Random Forest** is a bagging ensemble of decision trees trained on "
        "random subsets of training rows and features. It is robust to outliers, "
        "naturally handles non-linear interactions between GR/Vsh, RHOB, NPHI "
        "and RT_log, and provides interpretable feature importances. It serves "
        "as the interpretable baseline against which XGBoost and CNN-BiLSTM are compared."
    ),
    "XGBoost": (
        "**Extreme Gradient Boosting** builds trees sequentially, each correcting "
        "the residuals of the previous one. It typically achieves the highest "
        "accuracy of the three models on tabular petrophysical data, with "
        "built-in L1/L2 regularisation and early stopping to prevent overfitting."
    ),
    "CNN-BiLSTM": (
        "**Hybrid CNN-BiLSTM** is the deep learning model. A 1-D CNN extracts "
        "local bedding patterns from a sliding window of 32 consecutive depth "
        "samples (~16 m), then a Bidirectional LSTM captures long-range depth "
        "trends both downward and upward. Unlike the ML models which treat each "
        "depth sample independently, CNN-BiLSTM exploits the sequential structure "
        "of well logs — making it particularly suited to transitional and "
        "cyclically interbedded lithologies."
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# BACKEND FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def _engineer_features(
    df: pd.DataFrame,
    col_map: dict[str, str],
    is_dl: bool,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """
    Build feature matrix from user's raw column mapping.

    Returns
    -------
    X        : 2-D float array (n_rows, n_features)
    feat_list: ordered feature names
    mask     : boolean array — True = row included in X
    """
    work = pd.DataFrame(index=df.index)

    # Pull raw logs
    for log in ["GR", "RHOB", "NPHI", "RT"]:
        src = col_map.get(log)
        if src and src != "None" and src in df.columns:
            work[log] = df[src].values.astype(float)
        else:
            work[log] = np.nan

    # Compute Vsh from GR  (p5–p95 per-well normalisation, clipped 0–1)
    gr = work["GR"]
    valid_gr = gr.dropna()
    if len(valid_gr) < 2:
        work["Vsh"] = np.nan
    else:
        gr_min  = float(valid_gr.quantile(0.05))
        gr_max  = float(valid_gr.quantile(0.95))
        denom   = (gr_max - gr_min) if (gr_max - gr_min) != 0 else 1.0
        work["Vsh"] = ((gr - gr_min) / denom).clip(0.0, 1.0)

    # Compute RT_log = log10(RT) — linearises the 4-decade resistivity range
    # clip(0.01) guards against zero / negative sentinel values
    rt = work["RT"].clip(lower=0.01)
    work["RT_log"] = np.log10(rt)

    # All models (ML and DL) use the same 5-feature set
    feat_list = list(_DL_FEATURES) if is_dl else list(_ML_FEATURES)
    X_df  = work[feat_list]
    mask  = ~X_df.isnull().any(axis=1).values
    X     = X_df.values.astype(float)

    # Fill residual NaNs with column median (safety net)
    for j in range(X.shape[1]):
        col_vals = X[:, j]
        nan_mask = np.isnan(col_vals)
        if nan_mask.any():
            med = float(np.nanmedian(col_vals))
            X[nan_mask, j] = med

    return X, feat_list, mask


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_key: str, target: str):
    fname = f"{model_key}_{target.lower()}.pkl"
    path  = os.path.join(_MODEL_DIR, fname)

    if not os.path.isfile(path):
        st.warning(
            f"Model file not found: `models/{fname}`\n\n"
            "Train with the Colab notebook and place the `.pkl` in `models/`."
        )
        return None

    try:
        with open(path, "rb") as fh:
            return pickle.load(fh)
    except Exception as exc:
        st.error(f"Failed to load `{fname}`: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def predict_with_model(model, X: np.ndarray, model_name: str) -> np.ndarray:
    try:
        y_pred = model.predict(X)
        return np.asarray(y_pred, dtype=float).ravel()
    except ValueError as exc:
        st.error(f"**Feature mismatch** ({model_name}): {exc}")
        return np.full(X.shape[0], np.nan)
    except Exception as exc:
        st.error(f"Prediction error ({model_name}): {exc}")
        return np.full(X.shape[0], np.nan)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN UI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def render_dl(df: pd.DataFrame) -> None:
    section_header(
        "🤖",
        "Unconventional Missing Data Generation",
        "Generate missing sonic logs (DTC / DTS) using pre-trained ML / DL models",
    )

    num_cols = numeric_cols(df)
    if not num_cols:
        st.error("No numeric columns found in the uploaded dataset.")
        return

    # STEP 1 — Target
    st.markdown("### Step 1 · Select Sonic Target")
    target_display = st.selectbox(
        "Sonic Target",
        ["DTC — P-wave Sonic (Compressional)", "DTS — S-wave Sonic (Shear)"],
        key="unc_target_select",
    )
    target_col = "DTC" if target_display.startswith("DTC") else "DTS"
    st.divider()

    # STEP 2 — Model
    st.markdown("### Step 2 · Select Model")
    selected_model = st.selectbox(
        "Model",
        MODEL_OPTIONS,
        index=MODEL_OPTIONS.index("XGBoost"),
        key="unc_model_select",
    )
    model_key = _MODEL_KEYS[selected_model]
    is_dl     = selected_model in _DL_MODELS

    # Per-model output column name — avoids overwriting between models
    pred_col = f"{target_col}_{model_key}_pred"

    _fname = f"{model_key}_{target_col.lower()}.pkl"
    _fpath = os.path.join(_MODEL_DIR, _fname)
    if os.path.isfile(_fpath):
        st.success(f"Model file found: `models/{_fname}`")
    else:
        st.warning(f"`models/{_fname}` not found. Run the Colab training notebook.")
    st.divider()

    # STEP 3 — Theory
    st.markdown("### Step 3 · Model Theory")
    with st.expander(f"About {selected_model}", expanded=False):
        st.markdown(_MODEL_THEORY[selected_model])
        feat_names = _DL_FEATURES if is_dl else _ML_FEATURES
        st.info(
            f"**Features used:** `{'`, `'.join(feat_names)}`\n\n"
            "GR is mapped by the user and converted to **Vsh** internally "
            "(p5–p95 normalisation, clipped 0–1)."
        )
    st.divider()

    # STEP 4 — Required logs info
    st.markdown("### Step 4 · Required Input Logs")
    _log_desc = {
        "GR":   "Gamma Ray (API) — auto-converted to Vsh",
        "RHOB": "Bulk Density (g/cc)",
        "NPHI": "Neutron Porosity (v/v)",
        "RT":   "True Resistivity / LLD (ohm·m) — log10-transformed in backend",
    }
    c0, c1 = st.columns([1, 3])
    c0.markdown("**Log**"); c1.markdown("**Description**")
    for log in _RAW_LOGS:
        ca, cb = st.columns([1, 3])
        ca.markdown(f"`{log}`")
        cb.markdown(_log_desc[log])
    st.caption("Vsh and RT_log are computed automatically in the backend — you only map GR, RHOB, NPHI, RT.")
    st.divider()

    # STEP 5 — Map logs
    st.markdown("### Step 5 · Map Input Logs")
    options_none = ["None"] + num_cols

    _candidates = {
        "GR":   ["GR", "GAMMA", "GAM", "GRD"],
        "RHOB": ["RHOB", "DRHOB", "RHO", "DEN", "DENSITY"],
        "NPHI": ["NPHI", "TNPH", "NEU", "NEUTRON"],
        "RT":   ["RT", "LLD", "ILD", "RES", "RESIST", "RDEEP"],
    }

    col_map: dict[str, str] = {}
    half = (len(_RAW_LOGS) + 1) // 2
    lc, rc = st.columns(2)

    def _sel(log, container):
        default = find_col(df, _candidates.get(log, [log])) or "None"
        idx     = options_none.index(default) if default in options_none else 0
        return container.selectbox(log, options_none, index=idx, key=f"unc_map_{log}",
                                   help=_log_desc.get(log, log))

    for log in _RAW_LOGS[:half]:
        col_map[log] = _sel(log, lc)
    for log in _RAW_LOGS[half:]:
        col_map[log] = _sel(log, rc)

    # Ground-truth (optional)
    st.markdown("**Ground-Truth Column** *(optional)*")
    gt_cands = {"DTC": ["DTC", "DTCO", "DT", "AC", "SONIC"],
                "DTS": ["DTS", "DTSM", "DT4S", "SS"]}
    auto_gt   = find_col(df, gt_cands.get(target_col, [target_col]))
    gt_opts   = ["None"] + num_cols
    gt_idx    = gt_opts.index(auto_gt) if (auto_gt and auto_gt in gt_opts) else 0
    true_col  = st.selectbox(f"True {target_col} column", gt_opts, index=gt_idx,
                              key="unc_true_col")
    true_col  = None if true_col == "None" else true_col
    st.divider()

    # STEP 6 — Predict
    st.markdown("### Step 6 · Predict")
    core_mapped = [l for l in ["GR", "RHOB", "NPHI"] if col_map.get(l, "None") != "None"]
    if len(core_mapped) < 3:
        st.warning("Map at least **GR, RHOB, and NPHI** before running prediction.")

    if not st.button("Run Prediction", type="primary", key="unc_predict_btn"):
        return

    df_work = st.session_state.df.copy()

    # Ensure DEPTH is a proper column (lasio sometimes leaves it as index)
    if "DEPTH" not in df_work.columns:
        if df_work.index.name in ("DEPTH", "DEPT", "MD", "TVD"):
            df_work = df_work.reset_index()
            df_work.rename(columns={df_work.columns[0]: "DEPTH"}, inplace=True)
        else:
            df_work.insert(0, "DEPTH", np.arange(len(df_work), dtype=float))

    # Sort by depth so depth-track lines are continuous, not scrambled
    if "DEPTH" in df_work.columns:
        df_work = df_work.sort_values("DEPTH").reset_index(drop=True)

    with st.spinner("Engineering features…"):
        try:
            X_all, feat_list, mask = _engineer_features(df_work, col_map, is_dl)
        except Exception as exc:
            st.error(f"Feature engineering failed: {exc}")
            return

    n_valid = int(mask.sum())
    n_skip  = int((~mask).sum())
    if n_valid == 0:
        st.error("No complete rows found after feature engineering.")
        return

    st.info(
        f"Feature matrix: **{len(feat_list)} features** × **{n_valid} rows** "
        f"({n_skip} skipped)\n\nFeatures: `{'`, `'.join(feat_list)}`"
    )

    with st.spinner(f"Loading {selected_model} model…"):
        model = load_model(model_key, target_col)
    if model is None:
        return

    with st.spinner(f"Running inference ({selected_model})…"):
        y_pred = predict_with_model(model, X_all[mask], selected_model)

    df_work[pred_col]           = np.nan
    df_work.loc[mask, pred_col] = y_pred
    st.session_state.df         = df_work

    n_predicted = int((~np.isnan(df_work[pred_col])).sum())
    st.success(f"Done — **{n_predicted}** rows written to `{pred_col}`.")

    # Sanity check
    valid_preds = df_work[pred_col].dropna()
    if len(valid_preds):
        p_min, p_max, p_mean = valid_preds.min(), valid_preds.max(), valid_preds.mean()
        expected = (40, 200) if target_col == "DTC" else (60, 400)
        if p_min < expected[0] or p_max > expected[1]:
            st.warning(
                f"⚠️ Some predictions outside expected range "
                f"{expected[0]}–{expected[1]} µs/ft "
                f"(min={p_min:.1f}, max={p_max:.1f}, mean={p_mean:.1f}). "
                "Verify model file and feature column mapping."
            )

    # Metrics (shown inside show_results below — no duplicate here)

    show_results(
        df_work, pred_col=pred_col, true_col=true_col,
        label=target_col, model_name=selected_model, unc_col=None,
        key_prefix=f"unc_{model_key}_{target_col.lower()}",
    )

