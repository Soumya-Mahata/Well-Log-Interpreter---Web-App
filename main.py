"""
main.py  —  Well Log LAS Interpreter  (v3)
==========================================
Single-file entry point.  Page modules are imported and called here.
All persistent state lives in st.session_state; nothing is re-initialised
on page navigation.

Run:
    streamlit run main.py
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np

import utils
import plots
import qc
import lithology
import porosity
import fluids
import results

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be the very first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Well Log Interpreter",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  /* ── layout ── */
  .block-container { padding-top: 0.8rem; padding-bottom: 1rem; }
  section[data-testid="stSidebar"] { min-width: 240px; max-width: 280px; }

  /* ── typography ── */
  h1  { color: #0D47A1; font-size: 1.55rem; }
  h2  { color: #1565C0; font-size: 1.18rem;
        border-bottom: 2px solid #BBDEFB; padding-bottom: 3px; margin-top: 1.2rem; }
  h3  { color: #1976D2; font-size: 1.02rem; margin-top: 0.9rem; }

  /* ── metric cards ── */
  div[data-testid="stMetric"] {
    background: #E3F2FD; border-radius: 8px;
    padding: 8px 12px; border-left: 4px solid #1565C0;
  }

  /* ── info / warn boxes ── */
  .info-box {
    background: #E3F2FD; border-left: 4px solid #1565C0;
    padding: 9px 14px; border-radius: 4px; margin: 6px 0; font-size: 0.91em;
  }
  .warn-box {
    background: #FFF8E1; border-left: 4px solid #F57F17;
    padding: 9px 14px; border-radius: 4px; margin: 6px 0; font-size: 0.91em;
  }
  .success-box {
    background: #E8F5E9; border-left: 4px solid #2E7D32;
    padding: 9px 14px; border-radius: 4px; margin: 6px 0; font-size: 0.91em;
  }

  /* ── tab font ── */
  button[data-baseweb="tab"] { font-size: 0.88rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE — initialise once, never reset on navigation
# ─────────────────────────────────────────────────────────────────────────────

_SS_DEFAULTS = {
    # ── Data objects ──────────────────────────────────────
    "las":            None,   # lasio.LASFile
    "raw_df":         None,   # df as loaded, NEVER modified
    "df_full":        None,   # raw_df after rename (full depth range)
    "df":             None,   # df_full after depth filter  ← used by all modules
    "rename_map":     {},     # {original_name: new_name}
    "selected_curves": None,  # list of curves chosen in the UI
    "curve_info":     None,
    "well_info":      None,
    # ── Depth filter ──────────────────────────────────────
    "depth_top":      None,
    "depth_base":     None,
    # ── QC ────────────────────────────────────────────────
    "df_qc":          None,
    # ── Computation flags ─────────────────────────────────
    "vsh_done":       False,
    "por_done":       False,
    "sw_done":        False,
    # ── Porosity parameters persisted across tabs ─────────
    "por_rhob_col":   None,
    "por_nphi_col":   None,
    "por_dt_col":     None,
    "por_gr_col":     None,
    "por_rho_matrix": 2.65,
    "por_rho_fluid":  1.0,
    "por_dt_matrix":  55.5,
    "por_dt_fluid":   189.0,
    "por_phid_sh":    0.10,
    "por_phin_sh":    0.30,
    # ── Fluid / Archie parameters ─────────────────────────
    "fl_rt_col":      None,
    "fl_phi_col":     None,
    "fl_a":           1.0,
    "fl_m":           2.0,
    "fl_n":           2.0,
    "fl_rw":          0.10,
    "fl_sw_cut":      0.60,
    # ── Core data ─────────────────────────────────────────
    "core_df":        None,
}

for _k, _v in _SS_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🛢️ Well Log Interpreter")
    st.caption("Formation Evaluation Suite · v3")
    st.divider()

    page = st.radio(
        "Navigate",
        [
            "📂 Data Loading & QC",
            "🪨 Lithology",
            "🕳️ Porosity",
            "💧 Fluid Analysis",
            "📊 Results & Export",
        ],
        label_visibility="collapsed",
        key="page_radio",
    )

    st.divider()
    _df = st.session_state.df
    if _df is not None:
        _d0, _d1 = _df["DEPTH"].min(), _df["DEPTH"].max()
        st.success(
            f"**Curves:** {len(_df.columns) - 1}  \n"
            f"**Samples:** {len(_df):,}  \n"
            f"**Depth:** {_d0:.1f} – {_d1:.1f}"
        )
        if st.session_state.depth_top:
            st.caption("🔍 Depth filter active")
        _computed = [
            c for c in ["VSH", "PHID", "PHIN", "PHIS", "PHIT", "PHIE",
                        "SW", "SH", "PAY_FLAG", "CLUSTER",
                        "M_LIT", "N_LIT", "RHOMAA", "DTMAA", "UMAA"]
            if c in _df.columns
        ]
        if _computed:
            st.caption("**Computed:** " + ", ".join(_computed))
    else:
        st.info("Upload a LAS file to begin.")


# ─────────────────────────────────────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────────────────────────────────────

def require_data():
    if st.session_state.df is None:
        st.warning("⚠️ Please load a LAS file first in **📂 Data Loading & QC**.")
        st.stop()


def _apply_depth_filter():
    """(Re-)apply the stored depth filter to df_full → df."""
    top  = st.session_state.depth_top
    base = st.session_state.depth_base
    full = st.session_state.df_full
    if full is None:
        return
    if top is not None and base is not None:
        st.session_state.df = utils.filter_depth(full, top, base)
    else:
        st.session_state.df = full.copy()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: DATA LOADING & QC
# ─────────────────────────────────────────────────────────────────────────────

if page == "📂 Data Loading & QC":

    st.title("📂 Data Loading & Quality Control")

    # ── 1. Upload ─────────────────────────────────────────────────────────────
    st.header("1. Upload LAS File")

    uploaded = st.file_uploader(
        "Drop a .las file here (LAS 1.2 / 2.0 / 3.0 supported)",
        type=["las", "LAS"],
        key="las_uploader",
    )

    if uploaded is not None:
        # Only re-parse if it's a genuinely new file (compare name + size)
        file_sig = f"{uploaded.name}_{uploaded.size}"
        if st.session_state.get("_loaded_file_sig") != file_sig:
            with st.spinner("Parsing LAS file …"):
                try:
                    las, raw_df = utils.load_las(uploaded)
                    st.session_state.las            = las
                    st.session_state.raw_df         = raw_df.copy()
                    st.session_state.df_full        = raw_df.copy()
                    st.session_state.df             = raw_df.copy()
                    st.session_state.df_qc          = raw_df.copy()
                    st.session_state.curve_info     = utils.get_curve_info(las)
                    st.session_state.well_info      = utils.get_well_info(las)
                    st.session_state.rename_map     = {}
                    st.session_state.selected_curves = [
                        c for c in raw_df.columns if c != "DEPTH"
                    ]
                    # reset filter & flags for fresh file
                    st.session_state.depth_top  = None
                    st.session_state.depth_base = None
                    st.session_state.vsh_done   = False
                    st.session_state.por_done   = False
                    st.session_state.sw_done    = False
                    st.session_state["_loaded_file_sig"] = file_sig
                    st.success(
                        f"✅ **{uploaded.name}** loaded — "
                        f"{len(raw_df):,} samples, "
                        f"{len(raw_df.columns) - 1} curves."
                    )
                except Exception as exc:
                    st.error(f"Could not read LAS file: {exc}")
                    st.stop()

    if st.session_state.df is None:
        st.markdown(
            '<div class="info-box">Upload a LAS file above to get started.</div>',
            unsafe_allow_html=True,
        )
        st.stop()

    # convenience references
    raw_df = st.session_state.raw_df

    # ── 2. Well Info ──────────────────────────────────────────────────────────
    with st.expander("2. Well Header Information", expanded=False):
        wi = st.session_state.well_info or {}
        rows = [
            {"Field": k, "Value": str(v.get("value", "")),
             "Unit": v.get("unit", ""), "Description": v.get("desc", "")}
            for k, v in wi.items()
            if str(v.get("value", "")) not in ("", "nan", "None")
        ]
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("No well header metadata found.")

    # ── 3. Curve Info ─────────────────────────────────────────────────────────
    with st.expander("3. Curve Information", expanded=False):
        st.dataframe(st.session_state.curve_info, use_container_width=True, hide_index=True)

    # ── 4. Curve Selection & Rename ───────────────────────────────────────────
    st.header("4. Curve Selection & Rename")
    all_curves = [c for c in raw_df.columns if c != "DEPTH"]

    # Persist selected_curves across reruns
    sel_default = st.session_state.selected_curves or all_curves
    selected = st.multiselect(
        "Curves to include in analysis",
        options=all_curves,
        default=[c for c in sel_default if c in all_curves],
        key="ms_curves",
    )
    st.session_state.selected_curves = selected  # save immediately

    st.caption("Optionally rename curves — names persist on all pages after clicking Apply.")
    rename_map = st.session_state.rename_map.copy()
    new_renames: dict = {}
    cols_per_row = 4
    for chunk_start in range(0, len(selected), cols_per_row):
        chunk = selected[chunk_start: chunk_start + cols_per_row]
        cols  = st.columns(len(chunk))
        for widget_col, orig in zip(cols, chunk):
            current = rename_map.get(orig, orig)
            entered = widget_col.text_input(
                label=orig, value=current, key=f"ren_{orig}", label_visibility="visible"
            )
            if entered and entered.strip() and entered.strip() != orig:
                new_renames[orig] = entered.strip()

    if st.button("✅ Apply Selection & Renames", type="primary", key="apply_renames"):
        # Build working frame: DEPTH + selected, then rename
        working = raw_df[["DEPTH"] + selected].copy()
        all_renames = {**rename_map, **new_renames}  # merge old + new
        if all_renames:
            working.rename(columns=all_renames, inplace=True)
        st.session_state.rename_map = all_renames

        # Carry computed columns that still fit the row count
        df_old = st.session_state.df_full
        computed_cols = [
            c for c in (df_old.columns if df_old is not None else [])
            if c not in working.columns and c != "DEPTH"
            and c in ["VSH","PHID","PHIN","PHIS","PHIT","PHIE",
                      "SW","SH","PAY_FLAG","CLUSTER",
                      "M_LIT","N_LIT","RHOMAA","DTMAA","UMAA"]
        ]
        if df_old is not None:
            for cc in computed_cols:
                if cc in df_old.columns and len(df_old) == len(working):
                    working[cc] = df_old[cc].values

        st.session_state.df_full = working.copy()
        _apply_depth_filter()
        st.session_state.df_qc = st.session_state.df.copy()
        st.success("✅ Curves & renames saved.  All pages will use the updated names.")

    # ── 5. Depth Range Filter ─────────────────────────────────────────────────
    st.header("5. Working Depth Range")
    df_full = st.session_state.df_full
    d_lo = float(df_full["DEPTH"].min())
    d_hi = float(df_full["DEPTH"].max())

    c_top, c_base = st.columns(2)
    saved_top  = st.session_state.depth_top  if st.session_state.depth_top  else d_lo
    saved_base = st.session_state.depth_base if st.session_state.depth_base else d_hi

    top_in  = c_top.number_input(
        "Top depth", value=float(saved_top),
        min_value=d_lo, max_value=d_hi, step=1.0, key="ni_top"
    )
    base_in = c_base.number_input(
        "Base depth", value=float(saved_base),
        min_value=d_lo, max_value=d_hi, step=1.0, key="ni_base"
    )

    btn_c1, btn_c2 = st.columns(2)
    if btn_c1.button("🔎 Apply Depth Filter", key="btn_depth_apply"):
        if top_in >= base_in:
            st.error("Top depth must be less than Base depth.")
        else:
            st.session_state.depth_top  = top_in
            st.session_state.depth_base = base_in
            _apply_depth_filter()
            n = len(st.session_state.df)
            st.success(f"Depth filter {top_in:.1f}–{base_in:.1f}: {n:,} samples retained.")

    if btn_c2.button("↩ Reset to Full Depth", key="btn_depth_reset"):
        st.session_state.depth_top  = None
        st.session_state.depth_base = None
        _apply_depth_filter()
        st.info("Depth filter cleared — using full well.")

    # ── 6. Raw Data Preview ───────────────────────────────────────────────────
    st.header("6. Raw Data Preview")
    df_show = st.session_state.df
    st.caption(f"{len(df_show):,} rows  ×  {len(df_show.columns)} columns")
    st.dataframe(df_show.head(400), use_container_width=True, height=280)

    # ── 7. Quick Log View ─────────────────────────────────────────────────────
    st.header("7. Quick Log View")
    num_curves = [c for c in df_show.columns if c != "DEPTH"]
    view_sel = st.multiselect(
        "Curves to display",
        num_curves,
        default=num_curves[: min(6, len(num_curves))],
        key="qlv_curves",
    )
    log_sc = st.multiselect(
        "Log-scale X-axis for",
        view_sel,
        key="qlv_logscale",
        help="Useful for resistivity curves",
    )
    if view_sel:
        st.plotly_chart(
            plots.plot_raw_logs(df_show, view_sel, log_scale_curves=log_sc),
            use_container_width=True,
        )

    # ── 8. Quality Control ────────────────────────────────────────────────────
    st.header("8. Quality Control")
    qc.render(st.session_state.df, st.session_state.raw_df)


# ─────────────────────────────────────────────────────────────────────────────
# OTHER PAGES — each module receives the current df from session_state
# ─────────────────────────────────────────────────────────────────────────────

elif page == "🪨 Lithology":
    require_data()
    lithology.render(st.session_state.df)

elif page == "🕳️ Porosity":
    require_data()
    porosity.render(st.session_state.df)

elif page == "💧 Fluid Analysis":
    require_data()
    fluids.render(st.session_state.df)

elif page == "📊 Results & Export":
    require_data()
    results.render(st.session_state.df)


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "Well Log Interpreter v3  ·  Streamlit + LASio + Plotly  ·  "
    "Archie (1942)  ·  Wyllie (1956)  ·  M-N / MID — Schlumberger (2005b)  ·  "
    "Crossplot theory — IIT ISM Formation Evaluation (Mandal 2026)"
)
