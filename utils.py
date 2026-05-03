"""
utils.py  —  Petrophysical Calculation Engine
==============================================
Pure-Python / NumPy / SciPy functions with no Streamlit dependency.
All functions are stateless: they receive arrays / DataFrames and return results.
No deprecated pandas API is used (ffill/bfill called as methods, not fillna kwargs).
"""

import io
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import zscore
from sklearn.cluster import KMeans
import lasio

# ── Mineral reference (Schlumberger 2005b Table 24.1) ────────────────────────
MINERALS = {
    # name        rho_ma   dt_ma    M      N      Umaa
    "Sandstone":  (2.65,   55.5,   0.84,  0.64,  4.79),
    "Limestone":  (2.71,   47.5,   0.85,  0.60,  13.78),
    "Dolomite":   (2.87,   43.5,   0.80,  0.53,  9.11),
    "Anhydrite":  (2.98,   50.0,   0.72,  0.52,  15.06),
    "Halite":     (2.17,   67.0,   1.24,  1.02,  9.49),
    "Gypsum":     (2.35,   52.0,   1.05,  0.31,  18.76),
}


# ─────────────────────────────────────────────────────────────────────────────
# LAS I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_las(uploaded_file):
    """
    Read a LAS file from a Streamlit UploadedFile object.
    Returns (las_object, DataFrame).
    The index column (DEPTH) is moved to a regular column and rows are sorted ascending.
    """
    raw_bytes = uploaded_file.read()
    text = raw_bytes.decode("utf-8", errors="replace")
    las = lasio.read(io.StringIO(text))
    df = las.df().reset_index()
    df.rename(columns={df.columns[0]: "DEPTH"}, inplace=True)
    df.sort_values("DEPTH", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return las, df


def get_well_info(las):
    """Return well header as {mnemonic: {value, unit, desc}} dict."""
    return {
        item.mnemonic: {"value": item.value, "unit": item.unit, "desc": item.descr}
        for item in las.well
    }


def get_curve_info(las):
    """Return DataFrame with one row per curve: Mnemonic, Unit, Description."""
    return pd.DataFrame(
        [{"Mnemonic": c.mnemonic, "Unit": c.unit, "Description": c.descr}
         for c in las.curves]
    )


# ─────────────────────────────────────────────────────────────────────────────
# DEPTH FILTERING
# ─────────────────────────────────────────────────────────────────────────────

def filter_depth(df, top, base):
    """Return a copy of df filtered to [top, base] depth range."""
    mask = (df["DEPTH"] >= top) & (df["DEPTH"] <= base)
    return df[mask].copy().reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# NULL / MISSING VALUE HANDLING
# ─────────────────────────────────────────────────────────────────────────────

def count_nulls(df):
    """Return DataFrame with null count and % per curve (excludes DEPTH)."""
    cols = [c for c in df.columns if c != "DEPTH"]
    n = len(df)
    counts = df[cols].isna().sum()
    return pd.DataFrame({
        "Curve":      cols,
        "Null Count": counts.values,
        "Null %":     (counts.values / n * 100).round(2),
    })


def fill_nulls(df, method="interpolate"):
    """
    Fill NaN values in all non-DEPTH columns.
    method: 'interpolate' | 'ffill' | 'bfill' | 'mean' | 'drop'
    Uses method-based call syntax (no deprecated fillna(method=...) kwargs).
    """
    df = df.copy()
    cols = [c for c in df.columns if c != "DEPTH"]
    if method == "drop":
        df.dropna(subset=cols, inplace=True)
    elif method == "interpolate":
        df[cols] = df[cols].interpolate(method="linear", limit_direction="both")
    elif method == "ffill":
        df[cols] = df[cols].ffill().bfill()
    elif method == "bfill":
        df[cols] = df[cols].bfill().ffill()
    elif method == "mean":
        for c in cols:
            df[c] = df[c].fillna(df[c].mean())
    df.reset_index(drop=True, inplace=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# OUTLIER DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_outliers_zscore(series, threshold=3.0):
    """Return boolean mask (True = outlier) using global Z-score."""
    mask = pd.Series(False, index=series.index)
    valid = series.dropna()
    if len(valid) < 3:
        return mask
    z = np.abs(zscore(valid.values.astype(float)))
    mask.loc[valid.index] = z > threshold
    return mask


def detect_outliers_median(series, window=7, threshold=3.0):
    """
    Return boolean mask using rolling Median Absolute Deviation (MAD).
    More robust than Z-score for non-Gaussian log distributions.
    """
    rolling_med = series.rolling(window=window, center=True, min_periods=1).median()
    diff = (series - rolling_med).abs()
    mad  = diff.rolling(window=window, center=True, min_periods=1).median()
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(mad > 0, diff / mad, 0.0)
    return pd.Series(ratio > threshold, index=series.index).fillna(False)


def replace_outliers(df, curve, mask):
    """Replace masked samples with linear interpolation."""
    df = df.copy()
    df.loc[mask, curve] = np.nan
    df[curve] = df[curve].interpolate(method="linear", limit_direction="both")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SMOOTHING
# ─────────────────────────────────────────────────────────────────────────────

def smooth_log(series, method="moving_average", window=5):
    """
    Smooth a log curve.
    'moving_average' — simple rolling mean (uniform noise reduction).
    'savgol'         — Savitzky-Golay polynomial filter (preserves peak shapes).
    """
    if method == "moving_average":
        return series.rolling(window=window, center=True, min_periods=1).mean()
    elif method == "savgol":
        w = window if window % 2 == 1 else window + 1
        poly = min(3, w - 1)
        filled = series.interpolate(method="linear", limit_direction="both").values
        smoothed = savgol_filter(filled, window_length=w, polyorder=poly)
        return pd.Series(smoothed, index=series.index)
    return series.copy()


# ─────────────────────────────────────────────────────────────────────────────
# HOLE QUALITY
# ─────────────────────────────────────────────────────────────────────────────

def hole_quality_check(df, cali_col, bit_size):
    """
    Compare CALI to bit size.
    Bad Hole : CALI > 110 % of bit  → log quality degraded (washout)
    Moderate : 100 % < CALI ≤ 110 % of bit  → use with caution
    Good Hole: CALI ≤ 100 % of bit  → in-gauge, high confidence
    """
    if cali_col not in df.columns:
        return pd.Series(["No CALI"] * len(df), index=df.index)
    cali = df[cali_col]
    result = np.select(
        [cali > bit_size * 1.10,
         cali > bit_size * 1.00],
        ["Bad Hole", "Moderate Hole"],
        default="Good Hole"
    )
    return pd.Series(result, index=df.index)


# ─────────────────────────────────────────────────────────────────────────────
# SHALE VOLUME
# ─────────────────────────────────────────────────────────────────────────────

def compute_vshale_gr(gr, gr_clean, gr_shale):
    """
    Linear GR shale volume index (Larionov 1969, linear form):
    Vsh = (GR - GR_clean) / (GR_shale - GR_clean),  clipped [0, 1].
    """
    denom = gr_shale - gr_clean
    if abs(denom) < 1e-6:
        return pd.Series(0.0, index=gr.index)
    return ((gr - gr_clean) / denom).clip(0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# POROSITY
# ─────────────────────────────────────────────────────────────────────────────

def density_porosity(rhob, rho_matrix=2.65, rho_fluid=1.0):
    """PHID = (ρma − ρb) / (ρma − ρf), clipped [0, 1]."""
    return ((rho_matrix - rhob) / (rho_matrix - rho_fluid)).clip(0.0, 1.0)


def sonic_porosity(dt, dt_matrix=55.5, dt_fluid=189.0):
    """Wyllie time-average: PHIS = (Δt − Δtma) / (Δtf − Δtma), clipped [0, 1]."""
    return ((dt - dt_matrix) / (dt_fluid - dt_matrix)).clip(0.0, 1.0)


def neutron_porosity(nphi):
    """
    Return PHIN in fraction v/v [0, 1].
    Auto-detects percent scale (median > 1) and converts.
    """
    nphi = nphi.copy()
    if nphi.dropna().median() > 1.0:
        nphi = nphi / 100.0
    return nphi.clip(0.0, 1.0)


def nd_porosity(phid, phin):
    """Neutron-Density average porosity for MID plot input."""
    return ((phid + phin) / 2.0).clip(0.0, 1.0)


def sn_porosity(phis, phin):
    """Sonic-Neutron average porosity for MID plot input."""
    return ((phis + phin) / 2.0).clip(0.0, 1.0)


def total_porosity(phid, phin):
    """
    Best-estimate total porosity (density + neutron combination).
    Gas correction: if PHID > PHIN use geometric mean (gas crossplot correction).
    Liquid-filled: simple arithmetic mean.
    Reference: Schlumberger Log Interpretation Principles, Vol. 1.
    """
    gas_mask = phid.values > phin.values
    phit = np.where(
        gas_mask,
        np.sqrt((phid.values ** 2 + phin.values ** 2) / 2.0),
        (phid.values + phin.values) / 2.0
    )
    return pd.Series(np.clip(phit, 0.0, 1.0), index=phid.index)


def effective_porosity(phit, vsh, phid_sh=0.10, phin_sh=0.30):
    """
    PHIE = PHIT − Vsh × φsh_avg   (shale correction).
    φsh_avg = (PHID_shale + PHIN_shale) / 2.
    Clipped [0, 1].
    """
    phi_sh_avg = (phid_sh + phin_sh) / 2.0
    return (phit - vsh * phi_sh_avg).clip(0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# M-N PLOT
# ─────────────────────────────────────────────────────────────────────────────

def compute_M(dt, rhob, dt_f=189.0, rho_f=1.0):
    """
    M = (Δtf − Δt) / [(ρb − ρf) × 100]
    Porosity-independent lithology indicator.
    Ref: Burke et al. 1969; Schlumberger 2005b.
    """
    denom = (rhob - rho_f) * 100.0
    with np.errstate(divide="ignore", invalid="ignore"):
        M = (dt_f - dt) / denom
    return pd.Series(np.where(np.abs(rhob - rho_f) > 1e-4, M, np.nan), index=dt.index)


def compute_N(phin, rhob, phin_f=1.0, rho_f=1.0):
    """
    N = (ϕNf − ϕN) / (ρb − ρf)
    phin must be in fraction v/v (limestone scale).
    """
    denom = rhob - rho_f
    with np.errstate(divide="ignore", invalid="ignore"):
        N = (phin_f - phin) / denom
    return pd.Series(np.where(np.abs(denom) > 1e-4, N, np.nan), index=phin.index)


# ─────────────────────────────────────────────────────────────────────────────
# MID PLOT
# ─────────────────────────────────────────────────────────────────────────────

def compute_rho_maa(rhob, phi_nd, rho_f=1.0):
    """
    Apparent matrix grain density:
    ρmaa = (ρb − ϕND × ρf) / (1 − ϕND)
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        rho_maa = (rhob - phi_nd * rho_f) / (1.0 - phi_nd)
    return pd.Series(np.where(phi_nd < 0.999, rho_maa, np.nan), index=rhob.index)


def compute_dt_maa(dt, phi_sn, dt_f=189.0):
    """
    Apparent matrix travel time:
    Δtmaa = (Δt − ϕSN × Δtf) / (1 − ϕSN)
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        dt_maa = (dt - phi_sn * dt_f) / (1.0 - phi_sn)
    return pd.Series(np.where(phi_sn < 0.999, dt_maa, np.nan), index=dt.index)


def compute_U_maa(rhob, pe, phi_nd, U_f=0.398):
    """
    Apparent volumetric photoelectric cross-section:
    Umaa = (ρb × Pe − ϕND × Uf) / (1 − ϕND)
    Uf (freshwater) = 0.398 barns/cm³  (Schlumberger 2005b).
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        U_maa = (rhob * pe - phi_nd * U_f) / (1.0 - phi_nd)
    return pd.Series(np.where(phi_nd < 0.999, U_maa, np.nan), index=rhob.index)


# ─────────────────────────────────────────────────────────────────────────────
# ARCHIE'S EQUATION
# ─────────────────────────────────────────────────────────────────────────────

def water_saturation_archie(rt, phit, rw=0.1, a=1.0, m=2.0, n=2.0):
    """
    Archie (1942):  Sw = [(a × Rw) / (Rt × ϕ^m)]^(1/n)
    Parameters:
        a  — tortuosity factor        (commonly 0.62–1.0)
        m  — cementation exponent     (commonly 1.8–2.2)
        n  — saturation exponent      (commonly 2.0)
        rw — formation water resistivity (ohm·m)
    Clipped to [0, 1].
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        f  = (a * rw) / (rt * np.power(phit, m))
        sw = np.power(np.abs(f), 1.0 / n)
    return pd.Series(np.clip(sw, 0.0, 1.0), index=rt.index)


def hydrocarbon_saturation(sw):
    """Sh = 1 − Sw, clipped [0, 1]."""
    return (1.0 - sw).clip(0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Rw ESTIMATION
# ─────────────────────────────────────────────────────────────────────────────

def estimate_rw_pickett(rt, phit, a=1.0, m=2.0):
    """
    Estimate Rw from Pickett plot regression on the water-bearing cluster.

    Method: bin porosity into 10 equal-log bins; take the P20 Rt value in
    each bin (approximates Sw=1 water-bearing points).  Fit log-log line:
        log10(Rt) = m · log10(1/ϕ) + log10(a·Rw)
    Intercept gives log10(a·Rw) → Rw = 10^intercept / a.

    Returns estimated Rw (ohm·m), clipped [0.001, 10].
    """
    tmp = pd.DataFrame({"rt": rt, "phi": phit}).dropna()
    tmp = tmp[(tmp["rt"] > 0) & (tmp["phi"] > 0.01)]
    if len(tmp) < 10:
        return 0.10

    # Bin into log-spaced porosity bins and take the low-Rt (P20) in each bin
    # to approximate the water-bearing (Sw≈1) population
    tmp["phi_log"] = np.log10(tmp["phi"])
    phi_min, phi_max = tmp["phi_log"].min(), tmp["phi_log"].max()
    bins = np.linspace(phi_min, phi_max, 11)  # 10 bins
    tmp["bin"] = pd.cut(tmp["phi_log"], bins=bins, labels=False)
    water_pts = (
        tmp.dropna(subset=["bin"])
           .groupby("bin", observed=True)
           .apply(lambda g: g.nsmallest(max(1, len(g) // 5), "rt"))  # lowest 20%
           .reset_index(drop=True)
    )

    if len(water_pts) < 5:
        # Fallback: use global P20 of Rt
        rt_p20 = tmp["rt"].quantile(0.20)
        water_pts = tmp[tmp["rt"] <= rt_p20]
    if len(water_pts) < 3:
        water_pts = tmp

    x = np.log10(1.0 / water_pts["phi"].values)
    y = np.log10(water_pts["rt"].values)
    coeffs = np.polyfit(x, y, 1)   # slope ≈ m, intercept = log10(a·Rw)
    rw = 10.0 ** coeffs[1] / a
    return float(np.clip(rw, 0.001, 10.0))


def estimate_rw_sp(sp, rmf=1.0, temp_f=150.0):
    """
    Simplified Schlumberger SP method:
      SSP = −K × log(Rmf_eq / Rw_eq),  K ≈ 61 mV at 25°C
      Rmf_eq = Rmf × (T + 6.77) / (75 + 6.77)
      Rw_eq  = Rmf_eq / 10^(SSP / 61)
      Rw     = Rw_eq  × (75 + 6.77) / (T + 6.77)
    SSP is taken as the median of the SP series in the analysis window.
    Returns estimated Rw (ohm·m), clipped [0.001, 10].
    """
    SSP = float(sp.dropna().median())
    rmf_eq = rmf * (temp_f + 6.77) / (75.0 + 6.77)
    rw_eq  = rmf_eq / (10.0 ** (SSP / 61.0))
    rw     = rw_eq  * (75.0 + 6.77) / (temp_f + 6.77)
    return float(np.clip(rw, 0.001, 10.0))


# ─────────────────────────────────────────────────────────────────────────────
# RESERVOIR EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def flag_reservoir(df, phie_col="", sw_col="", gr_col="",
                   phi_cut=0.10, sw_cut=0.60, gr_cut=75.0):
    """
    Boolean pay flag per sample.
    A sample is net pay only when ALL applicable cutoffs are satisfied:
      PHIE > phi_cut  AND  Sw < sw_cut  AND  GR < gr_cut.
    Any curve that is absent is ignored (its condition defaults to True).
    """
    cond = pd.Series(True, index=df.index)
    if phie_col and phie_col in df.columns:
        cond = cond & (df[phie_col] > phi_cut)
    if sw_col and sw_col in df.columns:
        cond = cond & (df[sw_col] < sw_cut)
    if gr_col and gr_col in df.columns:
        cond = cond & (df[gr_col] < gr_cut)
    return cond


def compute_net_pay(df, pay_flag):
    """Return dict with gross, net pay, NTG statistics."""
    depths = df["DEPTH"].values
    if len(depths) < 2:
        return {}
    step    = float(np.abs(np.median(np.diff(depths))))
    gross   = float(depths[-1] - depths[0])
    net_pay = float(int(pay_flag.sum()) * step)
    ntg     = net_pay / gross if gross > 0 else 0.0
    return {
        "Gross Interval":  round(gross,   2),
        "Net Pay":         round(net_pay, 2),
        "NTG":             round(ntg,     4),
        "Sample Interval": round(step,    4),
        "Pay Samples":     int(pay_flag.sum()),
        "Total Samples":   int(len(df)),
    }


def get_pay_intervals(df, pay_flag):
    """Build a DataFrame of contiguous pay intervals: Top, Base, Thickness."""
    depths = df["DEPTH"].values
    flags  = pay_flag.values.astype(bool)
    rows   = []
    in_zone, top_d = False, None
    for d, f in zip(depths, flags):
        if f and not in_zone:
            top_d   = d
            in_zone = True
        elif not f and in_zone:
            rows.append({"Top": round(top_d, 2), "Base": round(d, 2),
                         "Thickness": round(d - top_d, 2)})
            in_zone = False
    if in_zone and top_d is not None:
        rows.append({"Top": round(top_d, 2), "Base": round(depths[-1], 2),
                     "Thickness": round(depths[-1] - top_d, 2)})
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["Top", "Base", "Thickness"])


# ─────────────────────────────────────────────────────────────────────────────
# K-MEANS CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────

def kmeans_lithology(df, features, n_clusters=4):
    """
    Unsupervised k-means on selected log curves.
    Returns integer cluster labels aligned to df.index; unclassified rows = -1.
    """
    sub   = df[features].copy()
    valid = sub.dropna()
    labels = pd.Series(-1, index=df.index, dtype=int)
    if len(valid) < n_clusters:
        return labels
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels.loc[valid.index] = km.fit_predict(valid)
    return labels


# ─────────────────────────────────────────────────────────────────────────────
# CORE DATA
# ─────────────────────────────────────────────────────────────────────────────

def interpolate_core(core_df, log_depths, core_depth_col="DEPTH", core_phi_col="CPOR"):
    """
    Linearly interpolate core porosity onto the log depth grid.
    Explicitly casts columns to float64 to avoid TypeError when CSV columns
    are read as object/string dtype.
    """
    cs = core_df[[core_depth_col, core_phi_col]].copy()
    cs[core_depth_col] = pd.to_numeric(cs[core_depth_col], errors="coerce")
    cs[core_phi_col]   = pd.to_numeric(cs[core_phi_col],   errors="coerce")
    cs = cs.dropna().sort_values(core_depth_col)
    xp = cs[core_depth_col].values.astype(float)
    fp = cs[core_phi_col].values.astype(float)
    x  = pd.to_numeric(log_depths, errors="coerce").values.astype(float)
    values = np.interp(x, xp, fp, left=np.nan, right=np.nan)
    return pd.Series(values, index=log_depths.index)


def linear_calibration(log_phi, core_phi):
    """
    Least-squares linear fit: Core_PHI = slope × Log_PHI + intercept.
    Returns (slope, intercept, corrected_series).
    """
    tmp = pd.DataFrame({"log": log_phi, "core": core_phi}).dropna()
    if len(tmp) < 3:
        return 1.0, 0.0, log_phi.copy()
    coeffs = np.polyfit(tmp["log"].values, tmp["core"].values, 1)
    slope, intercept = float(coeffs[0]), float(coeffs[1])
    corrected = (log_phi * slope + intercept).clip(0.0, 1.0)
    return slope, intercept, corrected


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def find_col(df, candidates):
    """Return the first candidate column name present in df, else None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None
