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

Unit handling:
  • All internal calculations are performed in **km/s**.
  • Velocity log curves may be supplied in one of three units:
        – km/s         (no conversion needed)
        – m/s          (÷ 1000)
        – µs/ft  (DT)  (1 000 000 / (value × 3280.84) → km/s)
  • Predicted output is always stored in **km/s**; a companion column
    stores the result converted back to the *same unit as the input*
    so users can compare directly in their preferred domain.

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
# UNIT CONVERSION HELPERS
# ═════════════════════════════════════════════════════════════════════════════

# Available velocity units presented to the user
_VEL_UNITS = ["km/s", "m/s", "µs/ft (DT)"]

# Density units (Gardner output / RHOB input)
_DEN_UNITS = ["g/cc (g/cm³)", "kg/m³"]


def _to_kms(series: pd.Series, unit: str) -> pd.Series:
    """Convert a velocity series to km/s from the specified unit."""
    if unit == "km/s":
        return series
    elif unit == "m/s":
        return series / 1_000.0
    elif unit == "µs/ft (DT)":
        # DT [µs/ft] → Vp [km/s]
        # Vp [ft/µs] = 1 / DT[µs/ft]
        # Vp [m/s]   = (1 / DT[µs/ft]) × (1 ft / 1 µs) × (1e6 µs / s) × (0.3048 m / ft)
        # Vp [km/s]  = Vp [m/s] / 1000
        dt_safe = series.clip(lower=1e-6)
        return (1_000_000.0 * 0.3048) / (dt_safe * 1_000.0)
    else:
        raise ValueError(f"Unknown velocity unit: {unit!r}")


def _from_kms(series: pd.Series, unit: str) -> pd.Series:
    """Convert a velocity series from km/s back to the specified unit."""
    if unit == "km/s":
        return series
    elif unit == "m/s":
        return series * 1_000.0
    elif unit == "µs/ft (DT)":
        # Vp [km/s] → DT [µs/ft]
        vp_safe = series.clip(lower=1e-6)
        return (1_000_000.0 * 0.3048) / (vp_safe * 1_000.0)
    else:
        raise ValueError(f"Unknown velocity unit: {unit!r}")


def _to_gcc(series: pd.Series, unit: str) -> pd.Series:
    """Convert density to g/cc."""
    if unit == "g/cc (g/cm³)":
        return series
    elif unit == "kg/m³":
        return series / 1_000.0
    else:
        raise ValueError(f"Unknown density unit: {unit!r}")


def _unit_label(unit: str) -> str:
    """Short label for axis / column names."""
    return {
        "km/s":       "km/s",
        "m/s":        "m/s",
        "µs/ft (DT)": "µs/ft",
    }.get(unit, unit)


# ═════════════════════════════════════════════════════════════════════════════
# EMPIRICAL EQUATIONS  (all inputs / outputs in km/s or g/cc)
# ═════════════════════════════════════════════════════════════════════════════

def _gardner(rhob_gcc: pd.Series) -> pd.Series:
    """
    Gardner et al. (1974) — inverted for Vp
    Standard form (SI):  ρ [g/cc] = 0.31 · Vp [m/s]^0.25
    Inverted:            Vp [m/s] = (ρ / 0.31)^4
                         Vp [km/s] = Vp [m/s] / 1000
    """
    vp_ms = (rhob_gcc / 0.31) ** 4.0
    return vp_ms / 1_000.0


def _castagna_vp2vs(vp_kms: pd.Series) -> pd.Series:
    """
    Castagna et al. (1985) Mudrock Line
    Vs = 0.8621·Vp − 1.1724   [both in km/s]
    """
    return 0.8621 * vp_kms - 1.1724


def _castagna_vs2vp(vs_kms: pd.Series) -> pd.Series:
    """Inverse Castagna: Vp = (Vs + 1.1724) / 0.8621  [km/s]"""
    return (vs_kms + 1.1724) / 0.8621


def _smith(rt: pd.Series) -> pd.Series:
    """
    Smith (2007) — sonic from resistivity (clastic approximation)
    Vp ≈ 1 / (0.02·log(Rt) + 0.1)   [km/s]
    """
    return 1.0 / (0.02 * np.log(rt.clip(lower=0.1)) + 0.1)


def _brocher(vp_kms: pd.Series) -> pd.Series:
    """
    Brocher (2005) polynomial  [Vp and Vs in km/s, valid 1.5–8.5 km/s]
    Vs = 0.7858 − 1.2344·Vp + 0.7949·Vp² − 0.1238·Vp³ + 0.0064·Vp⁴
    """
    v = vp_kms.values
    return pd.Series(
        0.7858 - 1.2344*v + 0.7949*v**2 - 0.1238*v**3 + 0.0064*v**4,
        index=vp_kms.index,
    )


def _carroll(vp_kms: pd.Series) -> pd.Series:
    """
    Carroll (1969)
    Vs = 1.09913326 × Vp^0.9238115336   [km/s]
    Valid for Vp/Vs ratio between 1.61 and 1.85 (Poisson's ratio 0.22–0.28).
    """
    return 1.09913326 * (vp_kms.clip(lower=0.0) ** 0.9238115336)


# ── Method registry ───────────────────────────────────────────────────────────
_METHODS = [
    "Gardner (1974) — ρ → Vp",
    "Castagna (1985) — Vp → Vs  (Mudrock Line)",
    "Castagna (1985) — Vs → Vp  (Inverse)",
    "Smith (2007) — Rt → Vp",
    "Brocher (2005) — Vp → Vs",
    "Carroll (1969) — Vp → Vs",
]

# Map: method label → (required_input_role, output_tag, unc_fraction, fn)
_DISPATCH = {
    "Gardner (1974) — ρ → Vp":                   ("rhob", "Vp", 0.05, _gardner),
    "Castagna (1985) — Vp → Vs  (Mudrock Line)":  ("vp",   "Vs", 0.05, _castagna_vp2vs),
    "Castagna (1985) — Vs → Vp  (Inverse)":       ("vs",   "Vp", 0.05, _castagna_vs2vp),
    "Smith (2007) — Rt → Vp":                     ("rt",   "Vp", 0.08, _smith),
    "Brocher (2005) — Vp → Vs":                   ("vp",   "Vs", 0.05, _brocher),
    "Carroll (1969) — Vp → Vs":                   ("vp",   "Vs", 0.06, _carroll),
}

# Which methods need a velocity input and which role they use
_NEEDS_VP_UNIT = {"vp", "vs"}   # roles whose column needs unit conversion
_NEEDS_RT      = {"rt"}         # resistivity — no velocity unit needed


# ═════════════════════════════════════════════════════════════════════════════
# STREAMLIT RENDER
# ═════════════════════════════════════════════════════════════════════════════

def render(df: pd.DataFrame) -> None:
    section_header(
        "🔵", "Conventional / Empirical Methods",
        "Physics-based equations — no training required",
    )

    st.markdown("""
| Method | Input → Output | Equation | Uncertainty |
|--------|---------------|----------|-------------|
| Gardner (1974) | ρ (RHOB) → Vp | ρ = 0.31·Vp^0.25 (inverted) | ±5 % |
| Castagna (1985) | Vp → Vs | Vs = 0.8621·Vp − 1.1724 | ±5 % |
| Castagna (1985) inverse | Vs → Vp | Vp = (Vs + 1.1724) / 0.8621 | ±5 % |
| Smith (2007) | Rt → Vp | Vp = 1 / (0.02·ln(Rt) + 0.1) | ±8 % |
| Brocher (2005) | Vp → Vs | 4th-order polynomial | ±5 % |
| Carroll (1969) | Vp → Vs | Vs = 1.09913326·Vp^0.9238 | ±6 % |
    """)

    st.divider()

    # ── Method selector ───────────────────────────────────────────────────────
    method = st.selectbox(
        "Select empirical method", _METHODS, key="dg_conv_method",
    )

    required_role = _DISPATCH[method][0]

    # ── Curve selectors ───────────────────────────────────────────────────────
    num  = numeric_cols(df)
    opts = ["None"] + num

    auto_rhob = find_col(df, ["RHOB", "RHOZ", "DEN"])
    auto_vp   = find_col(df, ["VP", "VEL", "DT", "DTCO"])
    auto_vs   = find_col(df, ["VS", "DTS", "DTSM"])
    auto_rt   = find_col(df, ["RT", "AT90", "ILD", "RESD"])

    def idx(auto):
        return opts.index(auto) if auto in opts else 0

    c1, c2, c3, c4 = st.columns(4)
    rhob_col = c1.selectbox("RHOB (density)",  opts, key="dg_rhob_sel", index=idx(auto_rhob))
    vp_col   = c2.selectbox("Vp / DT curve",   opts, key="dg_vp_sel",   index=idx(auto_vp))
    vs_col   = c3.selectbox("Vs / DTS curve",  opts, key="dg_vs_sel",   index=idx(auto_vs))
    rt_col   = c4.selectbox("Rt (ohm·m)",       opts, key="dg_rt_sel",   index=idx(auto_rt))

    rhob_col = None if rhob_col == "None" else rhob_col
    vp_col   = None if vp_col   == "None" else vp_col
    vs_col   = None if vs_col   == "None" else vs_col
    rt_col   = None if rt_col   == "None" else rt_col

    # ── Unit selectors (shown contextually) ──────────────────────────────────
    st.markdown("##### ⚙️ Unit Configuration")
    st.caption(
        "Sonic logs are commonly stored as slowness (DT) in **µs/ft**. "
        "Select the unit that matches your input curve(s); the engine converts "
        "to km/s internally and reports the prediction in the same unit."
    )

    u1, u2, u3 = st.columns(3)

    # Smart defaults: if curve name contains DT → default to µs/ft
    def _smart_vel_idx(col_name):
        if col_name and any(k in col_name.upper() for k in ("DT", "DTCO", "DTSM")):
            return 2   # µs/ft (DT)
        return 0       # km/s

    vp_unit = u1.selectbox(
        "Vp / DT input unit",
        _VEL_UNITS,
        index=_smart_vel_idx(auto_vp),
        key="dg_vp_unit",
        help="Unit of the compressional velocity/slowness curve.",
    )

    vs_unit = u2.selectbox(
        "Vs / DTS input unit",
        _VEL_UNITS,
        index=_smart_vel_idx(auto_vs),
        key="dg_vs_unit",
        help="Unit of the shear velocity/slowness curve.",
    )

    den_unit = u3.selectbox(
        "RHOB / density unit",
        _DEN_UNITS,
        index=0,
        key="dg_den_unit",
        help="Unit of the bulk density curve (Gardner method only).",
    )

    # Output unit: default to µs/ft when input is DT
    _active_role_col = {"vp": auto_vp, "vs": auto_vs, "rhob": None, "rt": None}.get(required_role)
    _out_default = _smart_vel_idx(_active_role_col)
    out_unit = st.selectbox(
        "Predicted output unit",
        _VEL_UNITS,
        index=_out_default,
        key="dg_out_unit",
        help="Unit for the saved prediction column.",
    )

    # Determine which input unit applies to the active method
    _role_unit_map = {
        "vp":   vp_unit,
        "vs":   vs_unit,
        "rhob": den_unit,   # density — not velocity, handled separately
        "rt":   None,       # resistivity — no unit conversion
    }

    # Conversion note
    if required_role in _NEEDS_VP_UNIT:
        active_unit = _role_unit_map[required_role]
        if active_unit == "µs/ft (DT)":
            st.info(
                "🔄 **DT → velocity conversion active**: "
                f"DT [µs/ft] will be converted to km/s using "
                "`Vp [km/s] = (10⁶ × 0.3048) / (DT × 1000)` "
                "before applying the empirical equation.",
                icon=None,
            )

    st.divider()

    # ── Run button ────────────────────────────────────────────────────────────
    if not st.button("▶ Apply Method", type="primary", key="dg_conv_run"):
        return

    # ── Dispatch ──────────────────────────────────────────────────────────────
    cfg = _DISPATCH.get(method)
    if cfg is None:
        st.error("Unknown method — please select one from the list.")
        return

    required_role, out_tag, unc_frac, fn = cfg

    input_map = {
        "rhob": rhob_col,
        "vp":   vp_col,
        "vs":   vs_col,
        "rt":   rt_col,
    }
    true_map = {"Vp": vp_col, "Vs": vs_col}

    in_col = input_map[required_role]

    if not in_col:
        role_labels = {
            "rhob": "RHOB / bulk density",
            "vp":   "Vp / compressional slowness (DT)",
            "vs":   "Vs / shear slowness (DTS)",
            "rt":   "Rt (resistivity)",
        }
        st.error(
            f"This method requires **{role_labels[required_role]}** — "
            "please select the appropriate curve above."
        )
        return

    # ── Build working copy ────────────────────────────────────────────────────
    df_upd = st.session_state.df.copy()

    # ── Convert input to internal unit (km/s or g/cc) ─────────────────────────
    raw_input = df_upd[in_col]

    if required_role in _NEEDS_VP_UNIT:
        input_unit = _role_unit_map[required_role]
        input_kms  = _to_kms(raw_input, input_unit)
    elif required_role == "rhob":
        input_kms = _to_gcc(raw_input, den_unit)   # stays g/cc; fn expects g/cc
    else:
        input_kms = raw_input   # Rt: no conversion

    # ── Apply equation ────────────────────────────────────────────────────────
    pred_kms = fn(input_kms)

    # ── Convert output to user-chosen unit ───────────────────────────────────
    if out_tag in ("Vp", "Vs"):          # velocity output
        pred_out = _from_kms(pred_kms, out_unit)
        unit_lbl = _unit_label(out_unit)
    else:
        pred_out = pred_kms              # density — Gardner returns km/s for Vp
        unit_lbl = "km/s"

    # ── Column names ──────────────────────────────────────────────────────────
    # Map Vp/Vs output to DTC/DTS so the comparison module finds them.
    # Format: DTC_<MethodName>_pred  (matches comparison regex)
    _out_to_dtx = {"Vp": "DTC", "Vs": "DTS"}
    dtx_prefix  = _out_to_dtx.get(out_tag, out_tag)   # DTC / DTS / fallback

    # Sanitise method abbreviation for column name
    method_abbr = (
        method.split("(")[0].strip()
               .split("—")[0].strip()
               .replace(" ", "_")
               .replace(".", "")
    )
    pred_col = f"{dtx_prefix}_{method_abbr}_pred"
    unc_col  = f"{dtx_prefix}_{method_abbr}_uncertainty"

    # Also save the raw value in the user's chosen unit as a companion column
    unit_col = f"{dtx_prefix}_{method_abbr}_{unit_lbl.replace('/', '_')}"
    df_upd[unit_col]  = pred_out          # e.g. in µs/ft
    df_upd[pred_col]  = pred_out          # same data, discoverable by comparison
    df_upd[unc_col]   = pred_out.abs() * unc_frac

    short = method.split("—")[0].strip()
    st.success(f"✅ **{short}** → `{pred_col}` [{unit_lbl}] saved to dataset.")
    st.session_state.df = df_upd

    # ── Display unit-conversion summary ───────────────────────────────────────
    if required_role in _NEEDS_VP_UNIT and input_unit == "µs/ft (DT)":
        with st.expander("📐 Unit conversion details", expanded=False):
            st.markdown(f"""
| Step | Expression | Value (example at median DT) |
|------|-----------|-------------------------------|
| Input DT | `{raw_input.median():.2f}` µs/ft | — |
| → m/s | `Vp = 10⁶ × 0.3048 / DT` | `{(1e6 * 0.3048 / raw_input.median()):.1f}` m/s |
| → km/s | `÷ 1000` | `{(1e6 * 0.3048 / raw_input.median() / 1000):.3f}` km/s |
| Equation applied | {method.split('—')[1].strip()} | — |
| Output unit | {unit_lbl} | — |
""")

    # ── Display results ───────────────────────────────────────────────────────
    show_results(
        df_upd, pred_col,
        true_col=true_map.get(out_tag),
        label=f"{out_tag} [{unit_lbl}]",
        model_name=short,
        unc_col=unc_col,
        key_prefix="conv",
    )