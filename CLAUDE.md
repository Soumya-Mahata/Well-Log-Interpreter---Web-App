# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Petrophysics Platform is a Streamlit-based web application for academic-grade well log interpretation. It processes LAS (Log ASCII Standard) files to perform petrophysical analysis including lithology identification, porosity estimation, and fluid analysis.

**Live demo:** https://petrophysics-platform.streamlit.app

## Commands

```bash
# Run the application
streamlit run main.py

# Install dependencies
pip install -r requirements.txt
```

**Dependencies:** Streamlit, pandas, numpy, plotly, lasio, scikit-learn, xgboost, scipy, matplotlib

## Architecture

### Entry Point
- `main.py` — Single-file Streamlit entry point with page navigation via `st.radio`. All modules import into `main.py`; persistent state lives in `st.session_state`.

### Module Structure
Each tab/page is a separate module following a consistent pattern:
- `utils.py` — Core calculation engine (pure Python/NumPy/SciPy, no Streamlit dependency). Contains LAS I/O, depth filtering, null/outlier handling, petrophysical equations (Vshale, porosity, Archie's law, M-N plot, MID plot).
- `plots.py` — Plotly visualization library. All functions return `go.Figure` objects ready for `st.plotly_chart()`.
- `qc.py` — Quality control: null values, outlier detection, smoothing, hole quality checks.
- `lithology.py` — Six standard crossplots (NPHI-RHOB, NPHI-DT, RHOB-DT, M-N, MID plots) + K-means clustering.
- `porosity.py` — Density, sonic, neutron porosity calculations with shale correction and core calibration.
- `fluids.py` — Archie's equation for water saturation (Sw) with three Rw estimation methods (direct, Pickett plot, SP log).
- `results.py` — Triple combo logs, net pay calculation, composite interpretation, CSV export.
- `data_gen.py` — Missing sonic data generation (orchestrator module).
  - `dg_utils.py` — Shared helpers for data generation workflow.
  - `dg_conventional.py` — Physics-based empirical methods (Gardner, Castagna, Smith, Brocher, Carroll).
  - `dg_unconventional.py` — ML/DL models (Regression, Decision Tree, XGBoost, ANN, CNN, BiLSTM, CNN-BiLSTM).
  - `dg_comparison.py` — Metrics comparison between methods.
- `train_models.py` — Training scripts for all ML/DL models. Feature engineering via `build_features()`.

### Data Flow
1. LAS file → `utils.load_las()` → `raw_df` (immutable) → `df_full` (after rename) → `df` (depth-filtered, used by all modules)
2. Computed columns (VSH, PHID, PHIN, PHIT, PHIE, SW, SH, PAY_FLAG, CLUSTER, M_LIT, N_LIT, RHOMAA, DTMAA, UMAA) are stored in `st.session_state.df` and persist across navigation.

### Model Files
- `models/model_config.json` — Model configuration (input columns, target, feature count)
- `models/*.pkl` — Trained sklearn/PyTorch models (wrapped via `_TorchWrapper` for sklearn-compatible predict)
- `models/cnn_bilstm_sonic.h5` — Pre-trained deep learning model

### Key Conventions
- All widget keys are prefixed by module (e.g., `lit_`, `por_`, `fl_`, `res_`, `qc_`, `dg_`) to avoid Streamlit key collisions.
- `utils.find_col(df, candidates)` — Auto-detect curve names from common aliases (e.g., `["GR", "GRD"]`, `["RHOB", "RHOZ", "DEN"]`).
- Velocity units: Internal calculations use km/s; conversion helpers (`_to_kms`, `_from_kms`) handle m/s and µs/ft (DT).
- `dg_unconventional.py` handles all feature engineering internally; users only map raw logs (GR, RHOB, NPHI, RT, PEF, DEPTH).
