# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Start

```bash
streamlit run main.py
```

The app is available at https://petrophysics-platform.streamlit.app

## Architecture Overview

**Single-page Streamlit application** for well log interpretation with 6 navigation sections:

1. **Data Loading & QC** (`main.py` + `utils.py`) - LAS file parsing, curve selection, depth filtering
2. **Missing Data Generation** (`data_gen.py` + `dg_*.py`) - Predict Vp/Vs using conventional, ML, or DL methods
3. **Lithology Identification** (`lithology.py`) - Crossplots + K-means clustering
4. **Porosity Estimation** (`porosity.py`) - Density, sonic, neutron porosity calculations
5. **Fluid Analysis** (`fluids.py`) - Archie saturation, pay flag computation
6. **Results** (`results.py`) - Integrated log display

### Core Modules

| File | Purpose |
|------|---------|
| `main.py` | Entry point, session state management, navigation |
| `utils.py` | Petrophysical calculations (no Streamlit dependency) |
| `plots.py` | Plotly figure builders |
| `dg_utils.py` | Shared ML/DL helpers, Plotly builders |
| `dg_conventional.py` | Gardner, Castagna, Brocher equations |
| `dg_unconventional.py` | scikit-learn/XGBoost + PyTorch models |
| `dg_comparison.py` | Model evaluation metrics |

### Session State Pattern

All persistent state lives in `st.session_state` (initialized in `main.py:56-96`):
- `df` - working dataframe (modified by depth filter)
- `df_full` - full depth range after rename
- `raw_df` - original parsed data (never modified)
- Computed columns: `VSH`, `PHID/PHIN/PHIS/PHIE`, `SW`, `CLUSTER`, `Vp_pred`, etc.

## Key Dependencies

- **Core**: `streamlit`, `pandas`, `numpy`, `plotly`
- **ML**: `scikit-learn`, `xgboost`, `torch` (PyTorch)
- **Domain**: `lasio` (LAS file parsing)

Install: `pip install -r requirements.txt`

## Data Flow

```
LAS file → utils.load_las() → raw_df → rename/selection → df_full → depth filter → df
                                    ↓
                            All modules read/write to st.session_state.df
```

## Adding New Features

1. **New computed curve**: Add function to `utils.py`, then call from appropriate module (e.g., `porosity.py`)
2. **New crossplot**: Add to `plots.py`, then add tab in `lithology.py`
3. **New ML model**: Add to `dg_unconventional.py`, follow `_build_ml_model()` pattern
4. **New page**: Add to `main.py` navigation, create module following `render(df)` signature

## Testing

No automated tests. Validate petrophysical calculations against:
- Schlumberger Log Interpretation charts
- IIT ISM lecture notes (Mandal 2026)
- Haritha et al. (2025) for CNN-Bi-LSTM architecture
