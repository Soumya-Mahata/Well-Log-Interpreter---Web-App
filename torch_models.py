"""
torch_models.py
===============
Shared module for CNN-BiLSTM architecture and TorchWrapper.

WHY THIS FILE EXISTS
--------------------
Python's pickle stores the *fully-qualified class path* of every object it
serialises.  When the Colab notebook trains a model and saves the TorchWrapper,
pickle records the class as  __main__.TorchWrapper  (because Colab cells run in
__main__).  When Streamlit loads the .pkl, it looks for TorchWrapper in
__main__ — which is now main.py, not the notebook — and raises:

    Can't get attribute 'TorchWrapper' on <module 'main' from '...main.py'>

The solution: define TorchWrapper (and CNNBiLSTMModel) in this shared file.
Both the Colab notebook and dg_unconventional.py import from here, so pickle
always resolves the class as  torch_models.TorchWrapper  regardless of where
the .pkl was created or loaded.

USAGE
-----
Colab notebook (add at top of architecture cell):
    from torch_models import CNNBiLSTMModel, TorchWrapper

dg_unconventional.py (already done):
    from torch_models import TorchWrapper
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class CNNBiLSTMModel(nn.Module):
    """
    CNN-BiLSTM hybrid for sonic log prediction.

    Input:  (B, SEQ_LEN, n_features)
    Output: (B,)  — predicted DTC or DTS at the last depth step

    Architecture
    ------------
    Conv1D (x2, kernel=3)   — local bedding pattern extraction across depth
    BiLSTM (x2, hidden=96)  — bidirectional depth-sequence context
    Dense head               — scalar prediction
    """

    def __init__(self, n_features: int, hidden: int = 96, dropout: float = 0.25):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32), nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=32, hidden_size=hidden,
            num_layers=2, batch_first=True,
            bidirectional=True, dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, 64), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, SEQ_LEN, n_features)
        xc = x.transpose(1, 2)        # → (B, n_features, SEQ_LEN)
        xc = self.cnn(xc)              # → (B, 32, SEQ_LEN)
        xl = xc.transpose(1, 2)        # → (B, SEQ_LEN, 32)
        out, _ = self.lstm(xl)
        return self.head(out[:, -1, :]).squeeze(-1)   # last time-step → scalar


class TorchWrapper:
    """
    Sklearn-compatible wrapper stored inside every CNN-BiLSTM .pkl file.

    dg_unconventional.py calls  wrapper.predict(X_2d)  where X_2d is the
    raw (unscaled) feature matrix with shape (N, n_features), N rows ordered
    by depth.

    Internally
    ----------
    1. Impute NaN values using stored SimpleImputer (fitted on training data)
    2. Scale features with stored StandardScaler (fitted on training data)
    3. Build sliding windows: (n_windows, seq_len, n_features)
    4. Run CNNBiLSTMModel in mini-batches
    5. Inverse-scale predictions back to µs/ft
    6. Fill the leading (seq_len - 1) positions with nearest-neighbour so
       output length always equals input length N
    """

    def __init__(
        self,
        model: CNNBiLSTMModel,
        scaler_X,
        scaler_y,
        feature_cols: list[str],
        model_type: str,
        seq_len: int,
        imputer=None,
    ):
        self.model        = model.cpu()
        self.scaler_X     = scaler_X
        self.scaler_y     = scaler_y
        self.feature_cols = feature_cols
        self.model_type   = model_type
        self.seq_len      = seq_len
        self.imputer      = imputer

    def predict(self, X) -> np.ndarray:
        X2d = np.asarray(X, dtype=float).reshape(len(X), -1)

        # 1. Impute NaNs — prefer stored imputer, fall back to column-median
        #    (fallback handles sklearn version mismatches where the pickled
        #    SimpleImputer raises "_fill_dtype" AttributeError)
        try:
            if self.imputer is not None:
                X2d = self.imputer.transform(X2d)
        except Exception:
            # sklearn version mismatch — impute with column median directly
            for j in range(X2d.shape[1]):
                col = X2d[:, j]
                nan_mask = np.isnan(col)
                if nan_mask.any():
                    median = float(np.nanmedian(col))
                    X2d[nan_mask, j] = median if not np.isnan(median) else 0.0

        # 2. Scale
        Xn = self.scaler_X.transform(X2d)

        n_orig, seq = len(Xn), self.seq_len

        # 3. Pad if fewer samples than seq_len
        if n_orig < seq:
            pad = np.tile(Xn[0:1], (seq - n_orig, 1))
            Xn  = np.vstack([pad, Xn])

        # 4. Build sliding windows
        n_windows = len(Xn) - seq + 1
        X_seq = np.stack(
            [Xn[i : i + seq] for i in range(n_windows)], axis=0
        ).astype(np.float32)   # (n_windows, seq, n_features)

        # 5. Inference in mini-batches
        self.model.eval()
        batch_size   = 2048
        preds_scaled = []
        with torch.no_grad():
            for start in range(0, len(X_seq), batch_size):
                xb = torch.tensor(X_seq[start : start + batch_size])
                preds_scaled.append(self.model(xb).cpu().numpy())
        preds_scaled = np.concatenate(preds_scaled).ravel()

        # 6. Inverse-scale
        preds = self.scaler_y.inverse_transform(
            preds_scaled.reshape(-1, 1)
        ).ravel()

        # 7. Map back to original N rows
        #    Window i predicts depth position (i + seq - 1).
        #    Positions 0 … seq-2 get nearest-neighbour fill (first valid pred).
        if n_orig > len(X2d):          # we padded — trim leading predictions
            preds = preds[n_orig - len(X2d):]

        result           = np.empty(len(X2d))
        result[seq - 1:] = preds[: len(X2d) - seq + 1]
        result[: seq - 1] = preds[0]
        return result
