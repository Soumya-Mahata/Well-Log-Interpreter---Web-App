from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import json
import pickle
import tempfile
import os

from dg_utils import (
    numeric_cols, find_col, safe_metric,
    section_header, show_results
)

# Optional TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model as _keras_load_model
    _HAS_TF = True
except ImportError:
    _HAS_TF = False


# ==========================================================
# LOAD ARTIFACTS
# ==========================================================


def _load_keras_artifacts():

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(BASE_DIR, "models", "cnn_bilstm_sonic.h5")
    scaler_x_path = os.path.join(BASE_DIR, "models", "scalerX.pkl")
    scaler_y_path = os.path.join(BASE_DIR, "models", "scalerY.pkl")
    config_path = os.path.join(BASE_DIR, "models", "model_config.json")

    if not _HAS_TF:
        st.error("Install TensorFlow: pip install tensorflow")
        return None, None, None, None

    try:
        model = _keras_load_model(model_path, compile=False)

        with open(scaler_x_path, "rb") as f:
            scaler_X = pickle.load(f)

        with open(scaler_y_path, "rb") as f:
            scaler_y = pickle.load(f)

        with open(config_path, "r") as f:
            config = json.load(f)

    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None, None, None

    return model, scaler_X, scaler_y, config
# ==========================================================
# MC DROPOUT (UNCERTAINTY)
# ==========================================================


def _mc_predict(model, X, scaler_y, n_mc=20):

    preds = np.stack([
        model(X, training=True).numpy().flatten()
        for _ in range(n_mc)
    ])

    mean = preds.mean(axis=0).reshape(-1, 1)
    std = preds.std(axis=0)

    y = scaler_y.inverse_transform(mean).ravel()
    scale = float(getattr(scaler_y, "scale_", [1.0])[0])
    unc = std * scale

    return y, unc


# ==========================================================
# MAIN UI
# ==========================================================

def render_dl(df: pd.DataFrame) -> None:

    section_header("🔴", "CNN-BiLSTM (Pre-trained)",
                   "Load trained model from Colab and predict sonic logs")

    st.info(
        "Using pre-trained CNN-BiLSTM model stored locally in the /models folder.\n\n"
        "Make sure the following files exist:\n"
        "- cnn_bilstm_sonic.h5\n"
        "- scalerX.pkl\n"
        "- scalerY.pkl\n"
        "- model_config.json"
    )

    # FILE UPLOADS
    st.success("✅ Using pre-loaded model from /models folder")

    # LOAD ONCE
    if "keras_model" not in st.session_state:
        with st.spinner("Loading model..."):
            m, sx, sy, cfg = _load_keras_artifacts()
        if m is None:
            return

        st.session_state.keras_model = m
        st.session_state.scaler_X = sx
        st.session_state.scaler_y = sy
        st.session_state.config = cfg

        st.success("Model loaded")

    model = st.session_state.keras_model
    sx = st.session_state.scaler_X
    sy = st.session_state.scaler_y
    cfg = st.session_state.config

    input_cols = cfg.get("input_cols", [])
    n_features = len(input_cols)

    st.markdown("### Column Mapping")

    num = numeric_cols(df)
    col_map = {}

    for col in input_cols:
        col_map[col] = st.selectbox(
            f"{col} →",
            ["None"] + num,
            key=f"map_{col}"
        )

    auto_vp = find_col(df, ["VP", "VEL", "DT"])
    opts = ["None"] + num

    vp_true = st.selectbox(
        "True DTCO (optional)",
        opts,
        index=(opts.index(auto_vp) if auto_vp in opts else 0)
    )

    vp_true = None if vp_true == "None" else vp_true

    n_mc = st.slider("MC samples", 5, 50, 20)

    if not st.button("Predict", type="primary"):
        return

    if "None" in col_map.values():
        st.error("Map all columns")
        return

    cols = [col_map[c] for c in input_cols]

    df_upd = st.session_state.df.copy()

    X = df_upd[cols].values.astype(float)
    mask = df_upd[cols].notna().all(axis=1)

    if mask.sum() == 0:
        st.error("No valid rows")
        return

    X_valid = X[mask]

    X_norm = sx.transform(X_valid)
    X_norm = X_norm.reshape(X_norm.shape[0], 1, n_features)

    with st.spinner("Predicting..."):
        y_pred, unc = _mc_predict(model, X_norm, sy, n_mc)

    df_upd["Vp_pred"] = np.nan
    df_upd["Vp_uncertainty"] = np.nan

    df_upd.loc[mask, "Vp_pred"] = y_pred
    df_upd.loc[mask, "Vp_uncertainty"] = unc

    st.session_state.df = df_upd

    if vp_true and vp_true in df_upd.columns:
        if "DT" in vp_true.upper() or "DTCO" in vp_true.upper():
            # Convert DTCO (μs/ft) to Vp (m/s)
            df_upd["Vp_true"] = 304800.0 / df_upd[vp_true]
            true_col = "Vp_true"
        else:
            true_col = vp_true
    else:
        true_col = None

    if vp_true and vp_true in df_upd.columns:
        y_true = df_upd[true_col].values
        y_p = df_upd["Vp_pred"].values

        valid = ~np.isnan(y_true) & ~np.isnan(y_p)

        if valid.sum() > 0:
            m = safe_metric(y_true[valid], y_p[valid])
            st.success(
                f"RMSE: {m['RMSE']:.2f} | MAE: {m['MAE']:.2f} | R2: {m['R²']:.4f}"
            )

    show_results(
        df_upd,
        "Vp_pred",
        true_col=true_col,
        label="Vp",
        model_name="CNN-BiLSTM",
        unc_col="Vp_uncertainty",
        key_prefix="cnn"
    )
