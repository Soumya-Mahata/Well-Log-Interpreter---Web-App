"""
dg_conventional.py  —  Conventional Empirical Methods
======================================================
Implements physics-based equations to predict Vp / Vs:

  1. Gardner (1974)       ρ → Vp
  2. Castagna (1985)      Vp → Vs  (Mudrock Line)
  3. Castagna (1985)      Vs → Vp  (Inverse)
  4. Smith (2007)         Rt → Vp
  5. Brocher (2005)       Vp → Vs
  6. Carroll (1969)       Vp → Vs

Public entry point:
    from dg_conventional import render
    render(df)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from dg_utils import (
    find_col, numeric_cols,
    section_header, show_results,
)


# ═════════════════════════════════════════════════════════════════════════════
# EMPIRICAL EQUATIONS
# ═════════════════════════════════════════════════════════════════════════════

def _gardner(rhob: pd.Series) -> pd.Series:
    """
    Gardner et al. (1974)
    ρ = 0.31 · Vp^0.25   →   Vp = (ρ / 0.31)^4   [km/s from g/cc]
    """
    return (rhob / 0.31) ** 4.0 / 1000.0


def _castagna_vp2vs(vp: pd.Series) -> pd.Series:
    """
    Castagna et al. (1985) Mudrock Line
    Vs = 0.8621·Vp − 1.1724   [both in km/s]
    """
    return 0.8621 * vp - 1.1724


def _castagna_vs2vp(vs: pd.Series) -> pd.Series:
    """Inverse Castagna: Vp = (Vs + 1.1724) / 0.8621"""
    return (vs + 1.1724) / 0.8621


def _smith(rt: pd.Series) -> pd.Series:
    """
    Smith (2007) — sonic from resistivity (clastic approximation)
    Vp ≈ 1 / (0.02·log(Rt) + 0.1)   [km/s]
    """
    return 1.0 / (0.02 * np.log(rt.clip(lower=0.1)) + 0.1)


def _brocher(vp: pd.Series) -> pd.Series:
    """
    Brocher (2005) polynomial:
    Vs = 0.7858 − 1.2344·Vp + 0.7949·Vp² − 0.1238·Vp³ + 0.0064·Vp⁴
    """
    v = vp.values
    return pd.Series(
        0.7858 - 1.2344*v + 0.7949*v**2 - 0.1238*v**3 + 0.0064*v**4,
        index=vp.index,
    )


def _carroll(vp: pd.Series) -> pd.Series:
    """Carroll (1969): Vs = (Vp − 1.36) / 1.16"""
    return (vp - 1.36) / 1.16


# ── Method registry ───────────────────────────────────────────────────────────
_METHODS = [
    "Gardner (1974) — ρ → Vp",
    "Castagna (1985) — Vp → Vs  (Mudrock Line)",
    "Castagna (1985) — Vs → Vp  (Inverse)",
    "Smith (2007) — Rt → Vp",
    "Brocher (2005) — Vp → Vs",
    "Carroll (1969) — Vp → Vs",
]

# Map: method label → (required_input, output_tag, unc_fraction, fn)
_DISPATCH = {
    "Gardner (1974) — ρ → Vp":                ("rhob", "Vp", 0.05, _gardner),
    "Castagna (1985) — Vp → Vs  (Mudrock Line)": ("vp",  "Vs", 0.05, _castagna_vp2vs),
    "Castagna (1985) — Vs → Vp  (Inverse)":  ("vs",  "Vp", 0.05, _castagna_vs2vp),
    "Smith (2007) — Rt → Vp":                ("rt",   "Vp", 0.08, _smith),
    "Brocher (2005) — Vp → Vs":              ("vp",   "Vs", 0.05, _brocher),
    "Carroll (1969) — Vp → Vs":              ("vp",   "Vs", 0.06, _carroll),
}


# ═════════════════════════════════════════════════════════════════════════════
# STREAMLIT RENDER
# ═════════════════════════════════════════════════════════════════════════════

def render(df: pd.DataFrame) -> None:
    section_header("🔵", "Conventional / Empirical Methods",
                   "Physics-based equations — no training required")

    st.markdown("""
| Method | Input → Output | Uncertainty |
|--------|---------------|-------------|
| Gardner (1974) | ρ (RHOB) → Vp | ±5 % heuristic |
| Castagna (1985) | Vp → Vs | ±5 % heuristic |
| Castagna (1985) inverse | Vs → Vp | ±5 % heuristic |
| Smith (2007) | Rt → Vp | ±8 % heuristic |
| Brocher (2005) | Vp → Vs | ±5 % heuristic |
| Carroll (1969) | Vp → Vs | ±6 % heuristic |
    """)

    st.divider()

    # ── Method selector ───────────────────────────────────────────────────────
    method = st.selectbox("Select empirical method", _METHODS,
                          key="dg_conv_method")

    # ── Curve selectors (show all; only the relevant one is used) ─────────────
    num  = numeric_cols(df)
    opts = ["None"] + num

    auto_rhob = find_col(df, ["RHOB", "RHOZ", "DEN"])
    auto_vp   = find_col(df, ["VP", "VEL", "DT", "DTCO"])
    auto_vs   = find_col(df, ["VS", "DTS", "DTSM"])
    auto_rt   = find_col(df, ["RT", "AT90", "ILD", "RESD"])

    def idx(auto):
        return opts.index(auto) if auto in opts else 0

    c1, c2, c3, c4 = st.columns(4)
    rhob_col = c1.selectbox("RHOB (g/cc)", opts, key="dg_rhob_sel",  index=idx(auto_rhob))
    vp_col   = c2.selectbox("Vp   (km/s)", opts, key="dg_vp_sel",    index=idx(auto_vp))
    vs_col   = c3.selectbox("Vs   (km/s)", opts, key="dg_vs_sel",    index=idx(auto_vs))
    rt_col   = c4.selectbox("Rt (ohm·m)", opts,  key="dg_rt_sel",    index=idx(auto_rt))

    rhob_col = None if rhob_col == "None" else rhob_col
    vp_col   = None if vp_col   == "None" else vp_col
    vs_col   = None if vs_col   == "None" else vs_col
    rt_col   = None if rt_col   == "None" else rt_col

    # Map curve-role strings to actual column names
    input_map = {"rhob": rhob_col, "vp": vp_col, "vs": vs_col, "rt": rt_col}
    # Map output tag to the corresponding "true" reference column for display
    true_map  = {"Vp": vp_col, "Vs": vs_col}

    if not st.button("▶ Apply Method", type="primary", key="dg_conv_run"):
        return

    # ── Dispatch ──────────────────────────────────────────────────────────────
    cfg = _DISPATCH.get(method)
    if cfg is None:
        st.error("Unknown method — please select one from the list.")
        return

    required_role, out_tag, unc_frac, fn = cfg
    in_col = input_map[required_role]

    if not in_col:
        role_labels = {
            "rhob": "RHOB (density)",
            "vp":   "Vp (P-wave velocity)",
            "vs":   "Vs (S-wave velocity)",
            "rt":   "Rt (resistivity)",
        }
        st.error(
            f"This method requires **{role_labels[required_role]}** — "
            "please select the appropriate curve above."
        )
        return

    # Compute prediction
    df_upd = st.session_state.df.copy()
    pred_col = f"{out_tag}_pred"
    unc_col  = f"{out_tag}_uncertainty"

    df_upd[pred_col] = fn(df_upd[in_col])
    df_upd[unc_col]  = df_upd[pred_col].abs() * unc_frac

    short = method.split("—")[0].strip()
    st.success(f"✅ **{short}** → `{pred_col}` saved to dataset.")
    st.session_state.df = df_upd

    # Display results
    show_results(
        df_upd, pred_col,
        true_col=true_map.get(out_tag),
        label=out_tag,
        model_name=short,
        unc_col=unc_col,
        key_prefix="conv",
    )
