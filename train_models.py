"""
train_models.py
===============
Training scripts for all ML and DL models used by the petrophysics platform.

Targets:  DTC (P-wave sonic)  |  DTS (S-wave sonic)
Models:   Regression, Decision Tree, XGBoost, ANN, CNN, BiLSTM, CNN-BiLSTM

HOW TO USE
----------
1. Prepare your dataset as a pandas DataFrame with at minimum:
       GR, RHOB, NPHI, RT (or LLD), PEF, DEPTH, DTC, DTS
2. Import any trainer you need:
       from train_models import train_xgboost
       train_xgboost(df, target="DTC")
3. Trained .pkl files land in  models/

FEATURE ENGINEERING
-------------------
ALL feature engineering is handled internally by `build_features()`.
Users / callers only need to supply raw logs.

NAMING CONVENTION
-----------------
models/{model_key}_{target_lower}.pkl
  e.g.  models/regression_dtc.pkl
        models/cnn_bilstm_dts.pkl
"""

from __future__ import annotations

import os
import pickle
import warnings
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS — single source of truth for feature lists
# ─────────────────────────────────────────────────────────────────────────────

# Raw input logs expected from the user / dataset
RAW_LOGS = ["GR", "RHOB", "NPHI", "RT", "PEF"]

# Engineered features added by build_features() for ML models (no depth)
ML_FEATURES = [
    "GR", "RHOB", "NPHI", "RT", "PEF",
    "Vsh",                          # derived from GR
]

# Additional features for DL models
DL_EXTRA_FEATURES = ["GR_grad", "RHOB_grad", "NPHI_grad", "DEPTH_norm"]

DL_FEATURES = ML_FEATURES + DL_EXTRA_FEATURES  # 10 features total

# Targets
TARGETS = Literal["DTC", "DTS"]

# Output directory
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(_BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def build_features(
    df: pd.DataFrame,
    target: str,
    include_depth: bool = False,
    cross_sonic: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Derive all engineered features from raw logs.

    Parameters
    ----------
    df           : DataFrame containing raw logs (GR, RHOB, NPHI, RT, PEF,
                   optionally DEPTH, DTC, DTS)
    target       : "DTC" or "DTS"
    include_depth: True for DL models — adds DEPTH_norm + gradient features
    cross_sonic  : If True, add the opposing sonic log as an input feature
                   (only when that column is present in df)

    Returns
    -------
    feat_df      : DataFrame with exactly the engineered columns
    feature_cols : Ordered list of column names (used as model input)
    """
    out = df.copy()

    # ── 1. Vsh (Shale Volume from GR) ────────────────────────────────────────
    gr = out["GR"].copy()
    gr_min, gr_max = gr.quantile(0.05), gr.quantile(0.95)
    denom = gr_max - gr_min if (gr_max - gr_min) != 0 else 1.0
    out["Vsh"] = ((gr - gr_min) / denom).clip(0.0, 1.0)

    feature_cols: list[str] = list(ML_FEATURES)  # GR RHOB NPHI RT PEF Vsh

    # ── 2. Gradient features + depth (DL only) ───────────────────────────────
    if include_depth:
        out["GR_grad"]   = out["GR"].diff().fillna(0.0)
        out["RHOB_grad"] = out["RHOB"].diff().fillna(0.0)
        out["NPHI_grad"] = out["NPHI"].diff().fillna(0.0)

        if "DEPTH" in out.columns:
            depth = out["DEPTH"].astype(float)
            std = depth.std()
            out["DEPTH_norm"] = (depth - depth.mean()) / (std if std != 0 else 1.0)
        else:
            out["DEPTH_norm"] = 0.0  # fallback if depth not provided

        feature_cols = list(DL_FEATURES)  # includes grads + depth_norm

    # ── 3. Optional cross-sonic ───────────────────────────────────────────────
    cross_col = "DTS" if target == "DTC" else "DTC"
    if cross_sonic and cross_col in out.columns:
        out[cross_col + "_input"] = out[cross_col]
        feature_cols = feature_cols + [cross_col + "_input"]

    return out[feature_cols], feature_cols


def prepare_dataset(
    df: pd.DataFrame,
    target: str,
    include_depth: bool = False,
    cross_sonic: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Full dataset preparation pipeline.

    1. Build engineered features
    2. Drop rows where target is NaN
    3. Impute remaining NaNs with median
    4. Return X, y, feature_cols
    """
    if target not in df.columns:
        raise ValueError(
            f"Target column '{target}' not found in DataFrame. "
            "Ensure your dataset contains the target sonic log."
        )

    feat_df, feature_cols = build_features(
        df, target=target, include_depth=include_depth, cross_sonic=cross_sonic
    )

    # Align with target, drop missing target rows
    combined = feat_df.copy()
    combined["__target__"] = df[target].values
    combined = combined.dropna(subset=["__target__"])

    y = combined["__target__"].values.astype(float)
    X_raw = combined[feature_cols].values.astype(float)

    # Impute any remaining NaNs in features
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X_raw)

    print(
        f"[prepare_dataset] target={target} | features={len(feature_cols)} | "
        f"rows={len(y)} | NaN rows dropped={len(df) - len(y)}"
    )
    return X, y, feature_cols


def _print_metrics(y_test: np.ndarray, y_pred: np.ndarray, label: str) -> None:
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2   = float(r2_score(y_test, y_pred))
    print(f"  [{label}]  RMSE={rmse:.4f}  R²={r2:.4f}")


def _save(model, model_key: str, target: str) -> str:
    fname = f"{model_key}_{target.lower()}.pkl"
    path  = os.path.join(MODEL_DIR, fname)
    with open(path, "wb") as fh:
        pickle.dump(model, fh)
    print(f"  Saved → {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# ML TRAINERS
# ─────────────────────────────────────────────────────────────────────────────

def train_regression(df: pd.DataFrame, target: str = "DTC") -> str:
    """
    Train Ridge Regression and save to models/regression_{target}.pkl

    Parameters
    ----------
    df     : DataFrame with raw logs + target column
    target : "DTC" or "DTS"

    Returns
    -------
    path to saved .pkl
    """
    print(f"\n[Regression] Training for target={target}")
    X, y, _ = prepare_dataset(df, target, include_depth=False)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("ridge",   Ridge(alpha=1.0)),
    ])
    model.fit(X_tr, y_tr)
    _print_metrics(y_te, model.predict(X_te), "Regression")
    return _save(model, "regression", target)


def train_decision_tree(df: pd.DataFrame, target: str = "DTC") -> str:
    """
    Train Decision Tree Regressor and save to models/decision_tree_{target}.pkl
    """
    print(f"\n[Decision Tree] Training for target={target}")
    X, y, _ = prepare_dataset(df, target, include_depth=False)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("dt",      DecisionTreeRegressor(
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
        )),
    ])
    model.fit(X_tr, y_tr)
    _print_metrics(y_te, model.predict(X_te), "Decision Tree")
    return _save(model, "decision_tree", target)


def train_xgboost(df: pd.DataFrame, target: str = "DTC") -> str:
    """
    Train XGBoost Regressor and save to models/xgboost_{target}.pkl
    """
    print(f"\n[XGBoost] Training for target={target}")
    X, y, _ = prepare_dataset(df, target, include_depth=False)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("xgb",     XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbosity=0,
        )),
    ])
    model.fit(X_tr, y_tr)
    _print_metrics(y_te, model.predict(X_te), "XGBoost")
    return _save(model, "xgboost", target)


def train_ann(df: pd.DataFrame, target: str = "DTC") -> str:
    """
    Train MLP (ANN) Regressor and save to models/ann_{target}.pkl
    """
    print(f"\n[ANN] Training for target={target}")
    X, y, _ = prepare_dataset(df, target, include_depth=False)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("ann",     MLPRegressor(
            hidden_layer_sizes=(256, 128, 64, 32),
            activation="relu",
            solver="adam",
            learning_rate_init=1e-3,
            max_iter=800,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=30,
            random_state=42,
        )),
    ])
    model.fit(X_tr, y_tr)
    _print_metrics(y_te, model.predict(X_te), "ANN")
    return _save(model, "ann", target)


# ─────────────────────────────────────────────────────────────────────────────
# PYTORCH WRAPPERS + DL TRAINERS
# ─────────────────────────────────────────────────────────────────────────────

# ── Wrapper (sklearn-compatible, pickle-safe) ─────────────────────────────────

class _TorchWrapper:
    """
    Wraps a PyTorch model for sklearn-style .predict(X_2d).

    Handles scaling internally so the app can call predict(X_raw_2d).
    Sequential models (CNN / BiLSTM / CNN-BiLSTM) also accept
    X shaped (n, 1, features) — the wrapper normalises to 2-D first.
    """

    def __init__(self, model, scaler_X, scaler_y, feature_cols: list[str],
                 model_type: str):
        self.model        = model
        self.scaler_X     = scaler_X
        self.scaler_y     = scaler_y
        self.feature_cols = feature_cols
        self.model_type   = model_type  # "cnn" | "bilstm" | "cnn_bilstm"

    # Called by dg_unconventional.py → predict_with_model()
    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch
        # Flatten to 2-D regardless of what shape arrives
        X2d = np.asarray(X, dtype=float).reshape(X.shape[0], -1)
        Xn  = self.scaler_X.transform(X2d)

        # Shape depends on architecture
        if self.model_type == "cnn":
            # Conv1d expects (B, C_in=features, L=1)
            Xt = torch.tensor(Xn, dtype=torch.float32).unsqueeze(-1)
        else:
            # BiLSTM / CNN-BiLSTM expect (B, seq_len=1, features)
            Xt = torch.tensor(Xn, dtype=torch.float32).unsqueeze(1)

        self.model.eval()
        with torch.no_grad():
            out = self.model(Xt).cpu().numpy()

        return self.scaler_y.inverse_transform(
            out.reshape(-1, 1)
        ).ravel()


# ── Architecture definitions ──────────────────────────────────────────────────

def _build_cnn(n_features: int, dropout: float = 0.2):
    """1-D CNN for petrophysical logs."""
    import torch.nn as nn

    class CNNModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(n_features, 128, kernel_size=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(128, 64, kernel_size=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(64, 32, kernel_size=1),
                nn.ReLU(),
            )
            self.head = nn.Linear(32, 1)

        def forward(self, x):        # x: (B, features, 1)
            h = self.net(x)          # (B, 32, 1)
            return self.head(h.squeeze(-1)).squeeze(-1)  # (B,)

    return CNNModel()


def _build_bilstm(n_features: int, hidden: int = 128, dropout: float = 0.2):
    """Bidirectional LSTM."""
    import torch.nn as nn

    class BiLSTMModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=n_features,
                hidden_size=hidden,
                num_layers=2,
                batch_first=True,
                bidirectional=True,
                dropout=dropout,
            )
            self.head = nn.Sequential(
                nn.Linear(hidden * 2, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1),
            )

        def forward(self, x):        # x: (B, 1, features)
            out, _ = self.lstm(x)    # (B, 1, hidden*2)
            return self.head(out[:, -1, :]).squeeze(-1)  # (B,)

    return BiLSTMModel()


def _build_cnn_bilstm(n_features: int, hidden: int = 128, dropout: float = 0.2):
    """CNN feature extractor → Bidirectional LSTM."""
    import torch.nn as nn

    class CNNBiLSTMModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.cnn = nn.Sequential(
                nn.Conv1d(n_features, 64, kernel_size=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(64, 32, kernel_size=1),
                nn.ReLU(),
            )
            self.lstm = nn.LSTM(
                input_size=32,
                hidden_size=hidden,
                num_layers=2,
                batch_first=True,
                bidirectional=True,
                dropout=dropout,
            )
            self.head = nn.Sequential(
                nn.Linear(hidden * 2, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1),
            )

        def forward(self, x):
            # x: (B, 1, features) → transpose for Conv1d → (B, features, 1)
            xc = x.transpose(1, 2)          # (B, features, 1)
            xc = self.cnn(xc)               # (B, 32, 1)
            xl = xc.transpose(1, 2)          # (B, 1, 32)  — seq for LSTM
            out, _ = self.lstm(xl)           # (B, 1, hidden*2)
            return self.head(out[:, -1, :]).squeeze(-1)  # (B,)

    return CNNBiLSTMModel()


# ── Generic DL training loop ──────────────────────────────────────────────────

def _train_torch(
    model,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_type: str,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 20,
) -> list[float]:
    """
    Adam training loop with early stopping.

    Returns list of validation losses (for optional plotting).
    """
    import torch
    import torch.nn as nn

    def _to_tensor(arr, model_type):
        t = torch.tensor(arr, dtype=torch.float32)
        if model_type == "cnn":
            return t.unsqueeze(-1)   # (B, features, 1)
        else:
            return t.unsqueeze(1)    # (B, 1, features)

    Xtr_t  = _to_tensor(X_tr,  model_type)
    ytr_t  = torch.tensor(y_tr,  dtype=torch.float32)
    Xval_t = _to_tensor(X_val, model_type)
    yval_t = torch.tensor(y_val, dtype=torch.float32)

    opt       = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=10, factor=0.5, verbose=False
    )
    criterion = nn.MSELoss()
    dataset   = torch.utils.data.TensorDataset(Xtr_t, ytr_t)
    loader    = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    best_val   = float("inf")
    best_state = None
    patience_c = 0
    val_losses = []

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(Xval_t), yval_t))
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val - 1e-6:
            best_val   = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_c = 0
        else:
            patience_c += 1
            if patience_c >= patience:
                print(f"    Early stop at epoch {epoch}, best val_loss={best_val:.6f}")
                break

        if epoch % 50 == 0:
            print(f"    Epoch {epoch:4d}  val_loss={val_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return val_losses


# ── DL trainer factory ────────────────────────────────────────────────────────

def _train_dl(
    df: pd.DataFrame,
    target: str,
    model_key: str,
    build_fn,
    epochs: int = 200,
    batch_size: int = 64,
) -> str:
    """
    Shared training routine for CNN / BiLSTM / CNN-BiLSTM.

    Parameters
    ----------
    df        : raw log DataFrame
    target    : "DTC" or "DTS"
    model_key : "cnn" | "bilstm" | "cnn_bilstm"
    build_fn  : callable(n_features) -> nn.Module
    """
    from sklearn.preprocessing import StandardScaler

    print(f"\n[{model_key.upper()}] Training for target={target}")

    X, y, feature_cols = prepare_dataset(df, target, include_depth=True)
    n_features = X.shape[1]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tr, y_tr, test_size=0.15, random_state=42
    )

    # Scale features and target separately
    scaler_X = StandardScaler()
    X_tr_n   = scaler_X.fit_transform(X_tr)
    X_val_n  = scaler_X.transform(X_val)
    X_te_n   = scaler_X.transform(X_te)

    scaler_y = StandardScaler()
    y_tr_n   = scaler_y.fit_transform(y_tr.reshape(-1, 1)).ravel()
    y_val_n  = scaler_y.transform(y_val.reshape(-1, 1)).ravel()

    net = build_fn(n_features)
    print(
        f"  Architecture: {model_key}  |  "
        f"features={n_features}  |  train={len(y_tr)}  "
        f"val={len(y_val)}  test={len(y_te)}"
    )

    _train_torch(
        net, X_tr_n, y_tr_n, X_val_n, y_val_n,
        model_type=model_key,
        epochs=epochs,
        batch_size=batch_size,
    )

    # Evaluate on test set
    wrapper = _TorchWrapper(net, scaler_X, scaler_y, feature_cols, model_key)
    y_pred  = wrapper.predict(X_te)
    _print_metrics(y_te, y_pred, model_key.upper())

    return _save(wrapper, model_key, target)


# ── Public DL trainers ────────────────────────────────────────────────────────

def train_cnn(df: pd.DataFrame, target: str = "DTC",
              epochs: int = 200, batch_size: int = 64) -> str:
    """Train CNN and save to models/cnn_{target}.pkl"""
    return _train_dl(
        df, target, "cnn",
        build_fn=lambda n: _build_cnn(n),
        epochs=epochs, batch_size=batch_size,
    )


def train_bilstm(df: pd.DataFrame, target: str = "DTC",
                 epochs: int = 200, batch_size: int = 64) -> str:
    """Train BiLSTM and save to models/bilstm_{target}.pkl"""
    return _train_dl(
        df, target, "bilstm",
        build_fn=lambda n: _build_bilstm(n),
        epochs=epochs, batch_size=batch_size,
    )


def train_cnn_bilstm(df: pd.DataFrame, target: str = "DTC",
                     epochs: int = 200, batch_size: int = 64) -> str:
    """Train CNN-BiLSTM and save to models/cnn_bilstm_{target}.pkl"""
    return _train_dl(
        df, target, "cnn_bilstm",
        build_fn=lambda n: _build_cnn_bilstm(n),
        epochs=epochs, batch_size=batch_size,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE: train ALL models for a given target
# ─────────────────────────────────────────────────────────────────────────────

def train_all(df: pd.DataFrame, target: str = "DTC",
              dl_epochs: int = 200) -> dict[str, str]:
    """
    Train every model for a single target and return paths.

    Parameters
    ----------
    df        : DataFrame with raw logs + target column
    target    : "DTC" or "DTS"
    dl_epochs : training epochs for CNN / BiLSTM / CNN-BiLSTM

    Returns
    -------
    dict mapping model_key → saved .pkl path
    """
    paths: dict[str, str] = {}
    paths["regression"]    = train_regression(df,    target)
    paths["decision_tree"] = train_decision_tree(df, target)
    paths["xgboost"]       = train_xgboost(df,       target)
    paths["ann"]           = train_ann(df,            target)
    paths["cnn"]           = train_cnn(df,            target, epochs=dl_epochs)
    paths["bilstm"]        = train_bilstm(df,         target, epochs=dl_epochs)
    paths["cnn_bilstm"]    = train_cnn_bilstm(df,     target, epochs=dl_epochs)
    return paths


# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE USAGE (do NOT execute in production — import and call instead)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Quick smoke-test with synthetic data ──────────────────────────────────
    # Replace this block with your real LAS / CSV dataset.
    print("Generating synthetic dataset for smoke-test …")
    np.random.seed(0)
    n = 500
    synthetic = pd.DataFrame({
        "DEPTH": np.linspace(1000, 3000, n),
        "GR":    np.random.uniform(20, 150, n),
        "RHOB":  np.random.uniform(2.0, 2.8, n),
        "NPHI":  np.random.uniform(0.05, 0.40, n),
        "RT":    np.random.uniform(1, 200, n),
        "PEF":   np.random.uniform(1.5, 5.0, n),
    })
    # Synthetic targets (loosely correlated with inputs)
    synthetic["DTC"] = (
        180 - 30 * synthetic["RHOB"] + 0.2 * synthetic["GR"]
        + np.random.normal(0, 5, n)
    )
    synthetic["DTS"] = synthetic["DTC"] * 1.7 + np.random.normal(0, 8, n)

    print("\n=== Training all models for DTC ===")
    paths_dtc = train_all(synthetic, target="DTC", dl_epochs=50)

    print("\n=== Training all models for DTS ===")
    paths_dts = train_all(synthetic, target="DTS", dl_epochs=50)

    print("\n=== Saved models ===")
    for k, p in {**paths_dtc, **paths_dts}.items():
        print(f"  {k}: {p}")
