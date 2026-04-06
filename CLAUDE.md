# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Well Log Interpreter** — Streamlit web application for petrophysical analysis of well-log LAS files. The app performs formation evaluation including lithology identification, porosity estimation, and fluid saturation calculations.

Live app: https://well-log-interpreter.streamlit.app/

## Commands

```bash
# Run the application
streamlit run main.py

# Install dependencies
pip install -r requirements.txt
```

## Architecture

### Module Structure

| File | Purpose |
|------|---------|
| `main.py` | Single-page Streamlit entry point with sidebar navigation. Manages `st.session_state` for persistent state across tabs. |
| `utils.py` | Pure computation engine (no Streamlit deps). LAS I/O, petrophysical calculations (Vsh, porosity, Archie's law, M-N plots), K-means clustering. |
| `plots.py` | Plotly visualization library. All functions return `go.Figure` ready for `st.plotly_chart()`. |
| `qc.py` | Quality control: null handling, outlier detection (Z-score/MAD), smoothing, hole quality. |
| `lithology.py` | Six standard crossplots (NPHI-RHOB, NPHI-DT, M-N, MID) + K-means clustering. |
| `porosity.py` | Density/neutron/sonic porosity with shale correction and core calibration. |
| `fluids.py` | Archie saturation calculation with 3 Rw estimation methods (direct, Pickett, SP). |
| `results.py` | Triple combo log, net pay calculation, composite interpretation, CSV export. |

### Data Flow

1. **Upload** → `utils.load_las()` → `raw_df` (immutable)
2. **Rename/Select** → `df_full` (working copy with renamed curves)
3. **Depth Filter** → `df` (filtered view used by all modules)
4. **Computation** → New columns added to `st.session_state.df`

### Session State Keys

- `las`, `raw_df`, `df_full`, `df` — data objects
- `rename_map`, `selected_curves` — curve management
- `depth_top`, `depth_base` — depth filter bounds
- `por_*`, `fl_*` — porosity/fluid parameters (persisted across tabs)
- `vsh_done`, `por_done`, `sw_done` — computation flags

### Key Petrophysical Functions

- `compute_vshale_gr()` — Linear GR shale volume (Larionov)
- `density_porosity()`, `sonic_porosity()`, `neutron_porosity()` — standard transforms
- `total_porosity()` — gas-corrected density+neutron combination
- `effective_porosity()` — PHIT corrected for shale
- `compute_M()`, `compute_N()` — M-N plot coordinates
- `compute_rho_maa()`, `compute_dt_maa()`, `compute_U_maa()` — MID plot
- `water_saturation_archie()` — Archie's law with customizable a, m, n, Rw
- `flag_reservoir()` — net pay flag based on cutoffs

### Reference Standards

- Mineral properties from Schlumberger Log Interpretation Charts (2005b)
- Archie (1942), Wyllie Time-Average (1956)
- M-N/MID plots — Burke et al. (1969)
