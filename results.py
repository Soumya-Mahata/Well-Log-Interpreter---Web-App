"""
results.py  —  Results & Export Module
========================================
All widget keys prefixed "res_".
Triple combo, cutoffs, composite plot, pay interval table, CSV export.
"""

import streamlit as st
import pandas as pd
import numpy as np
import utils
import plots


def _ss(key, default=None):
    return st.session_state.get(key, default)


def _render_kpi_cards(stats: dict):
    """
    Render a dict of {label: value} as a 2-column grid of KPI cards.
    Uses raw HTML so text is always dark-on-white regardless of Streamlit theme.
    Avoids st.metric() whose text inherits the app theme and becomes
    invisible (white-on-light) in Streamlit's dark mode.
    """
    items = list(stats.items())
    # Split into pairs for two-column layout
    for i in range(0, len(items), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j >= len(items):
                break
            label, value = items[i + j]
            col.markdown(
                f"""
                <div style="
                    background:#FFFFFF;
                    border-left:4px solid #1565C0;
                    border:1px solid #BBDEFB;
                    border-left:4px solid #1565C0;
                    border-radius:8px;
                    padding:12px 16px;
                    margin-bottom:8px;
                ">
                  <div style="color:#1565C0;font-size:0.78rem;font-weight:600;
                              letter-spacing:0.03em;margin-bottom:4px;">{label}</div>
                  <div style="color:#0D2B5E;font-size:1.6rem;font-weight:700;
                              line-height:1.2;">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render(df: pd.DataFrame):
    st.title("📊 Integration & Results")

    num_cols = [c for c in df.columns if c != "DEPTH"]
    _gr   = utils.find_col(df, ["GR",   "GRD"])
    _rt   = utils.find_col(df, ["RT",   "AT90", "ILD",  "RESD"])
    _nphi = utils.find_col(df, ["NPHI", "TNPH", "PHIN"])
    _rhob = utils.find_col(df, ["RHOB", "RHOZ", "DEN"])
    _phie = utils.find_col(df, ["PHIE", "PHIT"])
    _sw   = utils.find_col(df, ["SW"])

    st.markdown("""
<div class="info-box">
Apply porosity, Sw, and GR cutoffs to define net pay intervals.
The composite log highlights pay zones across all tracks with green shading.
</div>
""", unsafe_allow_html=True)

    def _sidx(col, opts):
        return opts.index(col) if col and col in opts else 0

    opts_n = ["None"] + num_cols

    tab_triple, tab_cuts, tab_composite, tab_export = st.tabs([
        "Triple Combo",
        "Cutoffs & Net Pay",
        "Composite Plot",
        "Export",
    ])

    # ── Tab 1 : Triple Combo ─────────────────────────────────────────────────
    with tab_triple:
        st.subheader("Triple Combo Log")
        st.caption(
            "Track 1 — GR (shaded, lithology indicator)  |  "
            "Track 2 — Resistivity log scale (fluid indicator)  |  "
            "Track 3 — NPHI + RHOB overlay (porosity tools)"
        )
        tc1, tc2, tc3, tc4 = st.columns(4)
        gr_tc   = tc1.selectbox("GR",   opts_n, key="res_tc_gr",   index=_sidx(_gr,   opts_n))
        rt_tc   = tc2.selectbox("RT",   opts_n, key="res_tc_rt",   index=_sidx(_rt,   opts_n))
        nphi_tc = tc3.selectbox("NPHI", opts_n, key="res_tc_nphi", index=_sidx(_nphi, opts_n))
        rhob_tc = tc4.selectbox("RHOB", opts_n, key="res_tc_rhob", index=_sidx(_rhob, opts_n))

        st.plotly_chart(
            plots.plot_triple_combo(
                df,
                gr_col=(None if gr_tc   == "None" else gr_tc),
                rt_col=(None if rt_tc   == "None" else rt_tc),
                nphi_col=(None if nphi_tc == "None" else nphi_tc),
                rhob_col=(None if rhob_tc == "None" else rhob_tc),
            ),
            use_container_width=True,
                key="results_pc1",
        )

    # ── Tab 2 : Cutoffs & Net Pay ─────────────────────────────────────────────
    with tab_cuts:
        st.subheader("Reservoir Cutoffs & Net Pay Calculation")
        st.markdown("""
A depth sample is **net pay** only when **all** applied cutoffs are satisfied:

| Criterion         | Cutoff applied | Petrophysical meaning            |
|-------------------|----------------|----------------------------------|
| PHIE > φ_cut      | Porosity       | Sufficient storage capacity      |
| Sw   < Sw_cut     | Saturation     | Hydrocarbon-bearing              |
| GR   < GR_cut API | Shaliness      | Clean formation (not shale)      |
        """)

        cc1, cc2, cc3 = st.columns(3)
        phie_col = cc1.selectbox("PHIE", opts_n, key="res_cut_phie", index=_sidx(_phie, opts_n))
        sw_col   = cc2.selectbox("Sw",   opts_n, key="res_cut_sw",   index=_sidx(_sw,   opts_n))
        gr_col   = cc3.selectbox("GR",   opts_n, key="res_cut_gr",   index=_sidx(_gr,   opts_n))

        phie_col = None if phie_col == "None" else phie_col
        sw_col   = None if sw_col   == "None" else sw_col
        gr_col   = None if gr_col   == "None" else gr_col

        cut1, cut2, cut3 = st.columns(3)
        phi_cut  = cut1.slider("PHIE cutoff (v/v)", 0.00, 0.40, 0.10, 0.01, key="res_phi_cut")
        sw_cut_r = cut2.slider("Sw cutoff",          0.10, 0.90, float(_ss("fl_sw_cut", 0.60)),
                                0.05, key="res_sw_cut")
        gr_cut   = cut3.slider("GR cutoff (API)",    20.0, 200.0, 75.0, 5.0, key="res_gr_cut")

        if st.button("🚦 Compute Net Pay & Flag Zones", type="primary", key="res_flag_btn"):
            with st.spinner("Flagging zones …"):
                pay_flag = utils.flag_reservoir(
                    df,
                    phie_col=phie_col or "",
                    sw_col=sw_col or "",
                    gr_col=gr_col or "",
                    phi_cut=phi_cut,
                    sw_cut=sw_cut_r,
                    gr_cut=gr_cut,
                )
                df_upd = st.session_state.df.copy()
                df_upd["PAY_FLAG"] = pay_flag.astype(int)
                st.session_state.df = df_upd

                stats = utils.compute_net_pay(df, pay_flag)

                # KPI row — use HTML cards to avoid theme-dependent invisible text
                st.markdown("#### Reservoir Summary")
                kpi_display = {
                    "Gross Interval":  f"{stats.get('Gross Interval', 0):.1f}",
                    "Net Pay":         f"{stats.get('Net Pay',        0):.1f}",
                    "NTG":             f"{stats.get('NTG',            0)*100:.1f}%",
                    "Pay Samples":     f"{stats.get('Pay Samples',    0):,}",
                    "Sample Interval": f"{stats.get('Sample Interval',0):.2f}",
                }
                _render_kpi_cards(kpi_display)

                # Pay interval table
                idf = utils.get_pay_intervals(df, pay_flag)
                if not idf.empty:
                    st.markdown("#### Pay Intervals")
                    st.dataframe(
                        idf.style.highlight_max(subset=["Thickness"], color="#C8F7C5"),
                        use_container_width=True, hide_index=True,
                    )
                else:
                    st.markdown(
                        '<div class="warn-box">No pay zones identified with the current cutoffs.  '
                        "Try relaxing the PHIE or Sw cutoff.</div>",
                        unsafe_allow_html=True,
                    )

    # ── Tab 3 : Composite Plot ────────────────────────────────────────────────
    with tab_composite:
        st.subheader("Final Composite Interpretation Log")
        st.caption(
            "Green horizontal bands = pay zones (computed in the Cutoffs tab).  "
            "Run the cutoffs computation first to enable pay shading."
        )

        cp1, cp2 = st.columns(2)
        gr_cp   = cp1.selectbox("GR",   opts_n, key="res_cp_gr",   index=_sidx(_gr,   opts_n))
        rt_cp   = cp2.selectbox("RT",   opts_n, key="res_cp_rt",   index=_sidx(_rt,   opts_n))
        cp3, cp4 = st.columns(2)
        nphi_cp = cp3.selectbox("NPHI", opts_n, key="res_cp_nphi", index=_sidx(_nphi, opts_n))
        rhob_cp = cp4.selectbox("RHOB", opts_n, key="res_cp_rhob", index=_sidx(_rhob, opts_n))
        cp5, cp6 = st.columns(2)
        phie_cp = cp5.selectbox("PHIE", opts_n, key="res_cp_phie", index=_sidx(_phie, opts_n))
        sw_cp   = cp6.selectbox("Sw",   opts_n, key="res_cp_sw",   index=_sidx(_sw,   opts_n))

        df_cur = st.session_state.df

        if "PAY_FLAG" in df_cur.columns:
            pay_flag_s = df_cur["PAY_FLAG"].astype(bool)
            fig_comp = plots.plot_final_interpretation(
                df_cur, pay_flag_s,
                gr_col=(None if gr_cp   == "None" else gr_cp),
                rt_col=(None if rt_cp   == "None" else rt_cp),
                nphi_col=(None if nphi_cp == "None" else nphi_cp),
                rhob_col=(None if rhob_cp == "None" else rhob_cp),
                phie_col=(None if phie_cp == "None" else phie_cp),
                sw_col=(None if sw_cp   == "None" else sw_cp),
            )
            st.plotly_chart(fig_comp, use_container_width=True,
                key="results_pc2")

            # Pay zone summary table beneath the composite log
            idf_comp = utils.get_pay_intervals(df_cur, pay_flag_s)
            if not idf_comp.empty:
                st.markdown("#### Pay Zone Summary")
                st.dataframe(
                    idf_comp.style.highlight_max(subset=["Thickness"], color="#C8F7C5"),
                    use_container_width=True, hide_index=True,
                )
        else:
            st.markdown(
                '<div class="warn-box">Run <b>Cutoffs & Net Pay</b> first to enable '
                "pay zone shading.  Showing triple combo instead.</div>",
                unsafe_allow_html=True,
            )
            st.plotly_chart(
                plots.plot_triple_combo(
                    df_cur,
                    gr_col=(None if gr_cp   == "None" else gr_cp),
                    rt_col=(None if rt_cp   == "None" else rt_cp),
                    nphi_col=(None if nphi_cp == "None" else nphi_cp),
                    rhob_col=(None if rhob_cp == "None" else rhob_cp),
                ),
                use_container_width=True,
                key="results_pc3",
            )

    # ── Tab 4 : Export ────────────────────────────────────────────────────────
    with tab_export:
        st.subheader("Download Results as CSV")
        df_cur = st.session_state.df

        all_col_opts = list(df_cur.columns)
        export_cols  = st.multiselect(
            "Columns to include",
            all_col_opts,
            default=all_col_opts,
            key="res_export_cols",
        )

        if export_cols:
            st.caption(f"Preview — first 100 rows  ({len(df_cur):,} total)")
            st.dataframe(df_cur[export_cols].head(100), use_container_width=True, height=260)

            csv_bytes = df_cur[export_cols].to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download CSV",
                data=csv_bytes,
                file_name="well_log_interpretation.csv",
                mime="text/csv",
                type="primary",
            )

        # ── Full Interpretation Summary ────────────────────────────────────
        st.divider()
        st.subheader("📋 Interpretation Summary")

        # KPI overview (if pay was computed)
        if "PAY_FLAG" in df_cur.columns:
            st.markdown("#### Reservoir Summary")
            pay_stats = utils.compute_net_pay(df_cur, df_cur["PAY_FLAG"].astype(bool))
            _render_kpi_cards(pay_stats)
            st.divider()

        # Petrophysical parameters used
        st.markdown("#### Parameters Used")
        param_rows = []

        # Porosity parameters
        for key, label, default in [
            ("por_rho_matrix", "Matrix density ρma (g/cc)", 2.65),
            ("por_rho_fluid",  "Fluid density ρf (g/cc)",   1.0),
            ("por_dt_matrix",  "Matrix travel time Δtma (µs/ft)", 55.5),
            ("por_dt_fluid",   "Fluid travel time Δtf (µs/ft)",   189.0),
            ("por_phid_sh",    "PHID at shale (PHID_sh)",   0.10),
            ("por_phin_sh",    "PHIN at shale (PHIN_sh)",   0.30),
        ]:
            val = st.session_state.get(key, default)
            param_rows.append({"Module": "Porosity", "Parameter": label, "Value": f"{val}"})

        # Archie / Fluid parameters
        for key, label, default in [
            ("fl_a",  "Tortuosity factor (a)", 1.0),
            ("fl_m",  "Cementation exponent (m)", 2.0),
            ("fl_n",  "Saturation exponent (n)", 2.0),
            ("fl_rw", "Formation water resistivity Rw (ohm·m)", 0.10),
            ("fl_sw_cut", "Sw hydrocarbon cutoff", 0.60),
        ]:
            val = st.session_state.get(key, default)
            param_rows.append({"Module": "Fluid Analysis", "Parameter": label, "Value": f"{val}"})

        if param_rows:
            param_df = pd.DataFrame(param_rows)
            st.dataframe(param_df, use_container_width=True, hide_index=True)

        # Computed curve statistics
        st.divider()
        st.markdown("#### Computed Curve Statistics")
        summary_rows = []
        for col in ["VSH", "PHID", "PHIN", "PHIS", "PHIT", "PHIE", "SW", "SH", "RQI"]:
            if col in df_cur.columns:
                s = df_cur[col].dropna()
                summary_rows.append({
                    "Curve":     col,
                    "Min":       f"{s.min():.4f}",
                    "P10":       f"{s.quantile(0.10):.4f}",
                    "Mean":      f"{s.mean():.4f}",
                    "Median":    f"{s.median():.4f}",
                    "P90":       f"{s.quantile(0.90):.4f}",
                    "Max":       f"{s.max():.4f}",
                    "Non-null %":f"{s.notna().mean()*100:.1f}%",
                })
        if summary_rows:
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No computed curves yet.  Run Porosity and Fluid Analysis first.")

        # Pay intervals detail
        if "PAY_FLAG" in df_cur.columns:
            idf_ex = utils.get_pay_intervals(df_cur, df_cur["PAY_FLAG"].astype(bool))
            if not idf_ex.empty:
                st.divider()
                st.markdown("#### Pay Interval Detail")
                st.dataframe(
                    idf_ex.style.highlight_max(subset=["Thickness"], color="#C8F7C5"),
                    use_container_width=True, hide_index=True,
                )
