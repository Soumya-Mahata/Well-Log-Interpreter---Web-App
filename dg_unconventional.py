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

Model file naming:
  models/{model_key}_{target_lower}.pkl
  e.g.  models/cnn_bilstm_dtc.pkl
        models/xgboost_dts.pkl

Feature engineering is handled 100% in backend (invisible to user).
See train_models.py for the matching training-time feature pipeline.
"""

from __future__ import annotations

import os
import pickle
import warnings
from typing import Optional

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

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS  (must mirror train_models.py)
# ─────────────────────────────────────────────────────────────────────────────

_BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
_MODEL_DIR = os.path.join(_BASE_DIR, "models")

# UI labels → internal key
MODEL_OPTIONS: list[str] = [
    "Regression",
    "Decision Tree",
    "XGBoost",
    "ANN",
    "CNN",
    "BiLSTM",
    "CNN-BiLSTM",
]

_MODEL_KEYS: dict[str, str] = {
    "Regression":    "regression",
    "Decision Tree": "decision_tree",
    "XGBoost":       "xgboost",
    "ANN":           "ann",
    "CNN":           "cnn",
    "BiLSTM":        "bilstm",
    "CNN-BiLSTM":    "cnn_bilstm",
}

# DL models — expect 3-D input AND use extra engineered features
_DL_MODELS: set[str] = {"CNN", "BiLSTM", "CNN-BiLSTM"}

# Raw logs the user maps (shown in UI)
_RAW_LOGS_ML: list[str] = ["GR", "RHOB", "NPHI", "RT", "PEF"]
_RAW_LOGS_DL: list[str] = ["GR", "RHOB", "NPHI", "RT", "PEF", "DEPTH"]

# ML feature order (must match train_models.ML_FEATURES)
_ML_FEATURES: list[str] = ["GR", "RHOB", "NPHI", "RT", "PEF", "Vsh"]

# DL feature order (must match train_models.DL_FEATURES)
_DL_FEATURES: list[str] = [
    "GR", "RHOB", "NPHI", "RT", "PEF",
    "Vsh",
    "GR_grad", "RHOB_grad", "NPHI_grad",
    "DEPTH_norm",
]

# Model theory blurbs
_MODEL_THEORY: dict[str, str] = {
    "Regression": (
        "**Ridge Regression** fits a linear relationship between the input logs "
        "and the sonic target, with L2 regularisation to reduce overfitting. "
        "Fast and interpretable; best suited to formations where sonic responds "
        "approximately linearly to bulk density and porosity."
    ),
    "Decision Tree": (
        "**Decision Tree Regressor** partitions the feature space into rectangular "
        "regions via recursive binary splits. Captures non-linear thresholds "
        "(e.g. lithology boundaries) without scaling. Prone to overfitting at "
        "large depths; controlled here with `max_depth` and `min_samples_leaf`."
    ),
    "XGBoost": (
        "**Extreme Gradient Boosting** builds an ensemble of decision trees "
        "sequentially, each correcting the residuals of the previous one. "
        "Delivers strong performance on tabular petrophysical data with "
        "natural handling of non-linear interactions between GR, RHOB, and NPHI."
    ),
    "ANN": (
        "**Artificial Neural Network** (MLPRegressor) uses fully-connected layers "
        "with ReLU activations to learn complex non-linear mappings. Includes "
        "early stopping to prevent overfitting. Recommended when the "
        "GR–RHOB–NPHI–RT relationship is complex or varies across facies."
    ),
    "CNN": (
        "**1-D Convolutional Neural Network** treats the engineered feature vector "
        "at each depth level as a 1-sample sequence, applying convolutional "
        "filters to extract local patterns. Gradient features (GR_grad, etc.) "
        "help the CNN detect rapid lithology transitions."
    ),
    "BiLSTM": (
        "**Bidirectional Long Short-Term Memory** network processes the depth "
        "sequence both top-down and bottom-up, capturing long-range depth "
        "dependencies. Ideal for cyclically interbedded lithologies where "
        "context above and below a given depth matters."
    ),
    "CNN-BiLSTM": (
        "**Hybrid CNN-BiLSTM** first extracts local feature patterns with "
        "convolutional layers, then feeds the result into a Bidirectional LSTM "
        "for sequence-level context. Combines the spatial feature-extraction "
        "strength of CNN with the temporal memory of BiLSTM — typically "
        "the highest accuracy architecture for sonic log prediction."
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# BACKEND FEATURE ENGINEERING
# Must mirror train_models.build_features() exactly.
# ─────────────────────────────────────────────────────────────────────────────

def _engineer_features(
    df: pd.DataFrame,
    col_map: dict[str, str],
    is_dl: bool,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """
    Build the engineered feature matrix from the user's raw column mapping.

    Parameters
    ----------
    df      : working copy of st.session_state.df
    col_map : {log_name → df_column}  e.g. {"GR": "GAMMA_RAY", "RHOB": "RHOB"}
    is_dl   : True for CNN / BiLSTM / CNN-BiLSTM

    Returns
    -------
    X        : 2-D float array (n_rows, n_features) — complete rows only
    feat_list: ordered feature name list
    mask     : boolean array length n_rows — True = row included in X
    """
    n = len(df)
    work = pd.DataFrame(index=df.index)

    # ── Pull raw logs into work ───────────────────────────────────────────────
    for log in ["GR", "RHOB", "NPHI", "RT", "PEF"]:
        src = col_map.get(log)
        if src and src in df.columns:
            work[log] = df[src].values.astype(float)
        else:
            work[log] = np.nan  # missing → rows will be skipped

    # ── 1. Vsh ───────────────────────────────────────────────────────────────
    gr = work["GR"]
    gr_min  = float(gr.quantile(0.05))
    gr_max  = float(gr.quantile(0.95))
    denom   = (gr_max - gr_min) if (gr_max - gr_min) != 0 else 1.0
    work["Vsh"] = ((gr - gr_min) / denom).clip(0.0, 1.0)

    feat_list = list(_ML_FEATURES)   # GR RHOB NPHI RT PEF Vsh

    # ── 2. DL extras ─────────────────────────────────────────────────────────
    if is_dl:
        work["GR_grad"]   = work["GR"].diff().fillna(0.0)
        work["RHOB_grad"] = work["RHOB"].diff().fillna(0.0)
        work["NPHI_grad"] = work["NPHI"].diff().fillna(0.0)

        depth_src = col_map.get("DEPTH")
        if depth_src and depth_src in df.columns:
            depth = df[depth_src].values.astype(float)
            std   = float(np.nanstd(depth))
            work["DEPTH_norm"] = (depth - np.nanmean(depth)) / (std if std != 0 else 1.0)
        else:
            work["DEPTH_norm"] = 0.0

        feat_list = list(_DL_FEATURES)

    # ── Build matrix, drop NaN rows ──────────────────────────────────────────
    X_df = work[feat_list]
    mask = ~X_df.isnull().any(axis=1).values
    X    = X_df.values.astype(float)

    # Fill any remaining NaN with column median (safety net)
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
    """
    Load models/{model_key}_{target_lower}.pkl

    Returns model object or None on failure (error shown via st.warning).
    """
    fname = f"{model_key}_{target.lower()}.pkl"
    path  = os.path.join(_MODEL_DIR, fname)

    if not os.path.isfile(path):
        st.warning(
            f"Model file not found: `models/{fname}`\n\n"
            "Train the model with `train_models.py` and place the `.pkl` "
            "in the `models/` directory."
        )
        return None

    try:
        with open(path, "rb") as fh:
            return pickle.load(fh)
    except Exception as exc:
        st.error(
            f"Failed to load `{fname}`: {exc}\n\n"
            "Check library version compatibility with `requirements.txt`."
        )
        return None


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def predict_with_model(
    model,
    X: np.ndarray,
    model_name: str,
) -> np.ndarray:
    """
    Run .predict() with correct input shape.

    ML models  : X shape (n, features)
    DL models  : wrapper handles reshaping internally via _TorchWrapper.predict()
                 but we still pass 2-D here — wrapper reshapes.
    """
    try:
        y_pred = model.predict(X)
        return np.asarray(y_pred, dtype=float).ravel()
    except ValueError as exc:
        st.error(
            f"**Feature mismatch** ({model_name}): {exc}\n\n"
            "The feature count or order does not match training. "
            "Ensure you are using the correct model file for this target."
        )
        return np.full(X.shape[0], np.nan)
    except Exception as exc:
        st.error(f"Prediction error ({model_name}): {exc}")
        return np.full(X.shape[0], np.nan)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN UI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def render_dl(df: pd.DataFrame) -> None:
    """
    Entry point called by data_gen.py.
    Backward-compatible name kept; now supports DTC + DTS + 7 model types.
    """
    section_header(
        "🤖",
        "Unconventional Missing Data Generation",
        "Generate missing sonic logs (DTC / DTS) using pre-trained ML / DL models",
    )

    num_cols = numeric_cols(df)
    if not num_cols:
        st.error("No numeric columns found in the uploaded dataset.")
        return

    # ═══════════════════════════════════════════════════════════════════════ #
    # STEP 1 — Target Selection
    # ═══════════════════════════════════════════════════════════════════════ #
    st.markdown("### Step 1 · Select Sonic Target")

    target_display = st.selectbox(
        "Sonic Target",
        ["DTC — P-wave Sonic (Compressional)", "DTS — S-wave Sonic (Shear)"],
        key="unc_target_select",
        help=(
            "DTC = compressional slowness (μs/ft). "
            "DTS = shear slowness (μs/ft)."
        ),
    )
    target_col = "DTC" if target_display.startswith("DTC") else "DTS"
    pred_col   = f"{target_col}_pred"

    st.divider()

    # ═══════════════════════════════════════════════════════════════════════ #
    # STEP 2 — Model Selection
    # ═══════════════════════════════════════════════════════════════════════ #
    st.markdown("### Step 2 · Select Model")

    selected_model = st.selectbox(
        "Model",
        MODEL_OPTIONS,
        index=MODEL_OPTIONS.index("CNN-BiLSTM"),
        key="unc_model_select",
        help="Choose a pre-trained model architecture.",
    )
    model_key = _MODEL_KEYS[selected_model]
    is_dl     = selected_model in _DL_MODELS

    # File status badge
    _fname = f"{model_key}_{target_col.lower()}.pkl"
    _fpath = os.path.join(_MODEL_DIR, _fname)
    if os.path.isfile(_fpath):
        st.success(f"Model file found: `models/{_fname}`")
    else:
        st.warning(
            f"`models/{_fname}` not found. "
            "Run `train_models.py` to generate it."
        )

    st.divider()

    # ═══════════════════════════════════════════════════════════════════════ #
    # STEP 3 — Model Theory
    # ═══════════════════════════════════════════════════════════════════════ #
    st.markdown("### Step 3 · Model Theory")

    with st.expander(f"About {selected_model}", expanded=False):
        st.markdown(_MODEL_THEORY[selected_model])

        if is_dl:
            st.info(
                "**DL models** use additional engineered features "
                "(Vsh, depth-normalised, gradient logs) computed automatically "
                "in the backend — you only need to map the raw logs below."
            )
        else:
            st.info(
                "**ML models** use Vsh (derived from GR) as an additional "
                "engineered feature alongside the raw logs — computed automatically."
            )

    st.divider()

    # ═══════════════════════════════════════════════════════════════════════ #
    # STEP 4 — Required Input Logs
    # ═══════════════════════════════════════════════════════════════════════ #
    st.markdown("### Step 4 · Required Input Logs")

    required_logs = _RAW_LOGS_DL if is_dl else _RAW_LOGS_ML
    _log_desc: dict[str, str] = {
        "GR":    "Gamma Ray (API)",
        "RHOB":  "Bulk Density (g/cc)",
        "NPHI":  "Neutron Porosity (v/v)",
        "RT":    "True Resistivity / LLD (ohm·m)",
        "PEF":   "Photoelectric Factor (b/e⁻)",
        "DEPTH": "Measured Depth (m or ft) — DL only",
    }

    log_rows = [(log, _log_desc.get(log, log)) for log in required_logs]
    # Display as a clean info table
    cols_info = st.columns([1, 3])
    cols_info[0].markdown("**Log**")
    cols_info[1].markdown("**Description**")
    for log, desc in log_rows:
        c1, c2 = st.columns([1, 3])
        c1.markdown(f"`{log}`")
        c2.markdown(desc)

    st.caption(
        "Engineered features (Vsh, gradients, DEPTH_norm) are computed "
        "automatically in the backend — you do **not** need to provide them."
    )

    st.divider()

    # ═══════════════════════════════════════════════════════════════════════ #
    # STEP 5 — Map Input Logs
    # ═══════════════════════════════════════════════════════════════════════ #
    st.markdown("### Step 5 · Map Input Logs")
    st.caption(
        "Select the column in your dataset that corresponds to each "
        "required log. Missing logs will be filled with the column median."
    )

    col_map: dict[str, str] = {}
    options_none = ["None"] + num_cols

    # Render 2 columns of selectboxes for neatness
    n_logs    = len(required_logs)
    half      = (n_logs + 1) // 2
    left_logs = required_logs[:half]
    rght_logs = required_logs[half:]

    lc, rc = st.columns(2)

    def _selectbox_for(log: str, container) -> str:
        candidates = {
            "GR":    ["GR", "GAMMA", "GAM", "GRD"],
            "RHOB":  ["RHOB", "DRHOB", "RHO", "DEN", "DENSITY"],
            "NPHI":  ["NPHI", "TNPH", "NEU", "NEUTRON"],
            "RT":    ["RT", "LLD", "ILD", "RES", "RESIST", "RDEEP"],
            "PEF":   ["PEF", "PE", "PHOTOELECTRIC"],
            "DEPTH": ["DEPTH", "MD", "DEPT", "TVD"],
        }
        default = find_col(df, candidates.get(log, [log])) or "None"
        idx     = options_none.index(default) if default in options_none else 0
        return container.selectbox(
            f"{log}",
            options_none,
            index=idx,
            key=f"unc_map_{log}",
            help=_log_desc.get(log, log),
        )

    for log in left_logs:
        col_map[log] = _selectbox_for(log, lc)
    for log in rght_logs:
        col_map[log] = _selectbox_for(log, rc)

    # ── Optional: ground-truth column for evaluation ──────────────────────
    st.markdown("**Ground-Truth Column** *(optional — for accuracy evaluation)*")
    gt_candidates = {
        "DTC": ["DTC", "DTCO", "DT", "AC", "SONIC"],
        "DTS": ["DTS", "DTSM", "DT4S", "SS"],
    }
    auto_gt   = find_col(df, gt_candidates.get(target_col, [target_col]))
    gt_opts   = ["None"] + num_cols
    gt_def_idx = gt_opts.index(auto_gt) if (auto_gt and auto_gt in gt_opts) else 0
    true_col  = st.selectbox(
        f"True {target_col} column",
        gt_opts,
        index=gt_def_idx,
        key="unc_true_col",
        help="If available, the app will compute RMSE / MAE / R² after prediction.",
    )
    true_col = None if true_col == "None" else true_col

    st.divider()

    # ═══════════════════════════════════════════════════════════════════════ #
    # STEP 6 — Run Prediction
    # ═══════════════════════════════════════════════════════════════════════ #
    st.markdown("### Step 6 · Predict")

    # Validate that at least the 3 core logs are mapped
    core_mapped = [
        log for log in ["GR", "RHOB", "NPHI"]
        if col_map.get(log, "None") != "None"
    ]
    if len(core_mapped) < 3:
        st.warning(
            "Map at least **GR, RHOB, and NPHI** before running prediction. "
            "Missing core logs degrade accuracy significantly."
        )

    if not st.button("Run Prediction", type="primary", key="unc_predict_btn"):
        return

    # ── Feature engineering ───────────────────────────────────────────────
    df_work = st.session_state.df.copy()

    with st.spinner("Engineering features…"):
        try:
            X_all, feat_list, mask = _engineer_features(df_work, col_map, is_dl)
        except Exception as exc:
            st.error(f"Feature engineering failed: {exc}")
            return

    n_valid = int(mask.sum())
    n_skip  = int((~mask).sum())

    if n_valid == 0:
        st.error(
            "No complete rows found after feature engineering. "
            "Check that the mapped columns contain data."
        )
        return

    st.info(
        f"Feature matrix: **{len(feat_list)} features** × **{n_valid} rows** "
        f"({n_skip} rows skipped — missing mapped logs)"
    )

    # ── Load model ────────────────────────────────────────────────────────
    with st.spinner(f"Loading {selected_model} model for {target_col}…"):
        model = load_model(model_key, target_col)

    if model is None:
        return

    # ── Predict ───────────────────────────────────────────────────────────
    with st.spinner(f"Running inference ({selected_model})…"):
        X_valid = X_all[mask]
        y_pred  = predict_with_model(model, X_valid, selected_model)

    # ── Store in session state ─────────────────────────────────────────────
    df_work[pred_col]      = np.nan
    df_work.loc[mask, pred_col] = y_pred
    st.session_state.df    = df_work

    n_predicted = int((~np.isnan(df_work[pred_col])).sum())
    st.success(
        f"Prediction complete — **{n_predicted}** rows filled in `{pred_col}`."
    )

    # ── Metrics banner ────────────────────────────────────────────────────
    if true_col and true_col in df_work.columns:
        y_true_arr = df_work[true_col].values.astype(float)
        y_pred_arr = df_work[pred_col].values.astype(float)
        both_valid = ~(np.isnan(y_true_arr) | np.isnan(y_pred_arr))
        if both_valid.sum() >= 2:
            m = safe_metric(y_true_arr[both_valid], y_pred_arr[both_valid])
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("RMSE", f"{m['RMSE']:.4f}")
            mc2.metric("MAE",  f"{m['MAE']:.4f}")
            key = "R\u00b2"
            mc3.metric(key, f"{m[key]:.4f}")

    # ── Full results panel ────────────────────────────────────────────────
    key_sfx = f"{model_key}_{target_col.lower()}"
    show_results(
        df_work,
        pred_col=pred_col,
        true_col=true_col,
        label=target_col,
        model_name=selected_model,
        unc_col=None,
        key_prefix=f"unc_{key_sfx}",
    )
