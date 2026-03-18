"""
porosity.py  —  Porosity Estimation Module
============================================
All widget keys prefixed "por_".
Parameters are persisted in session_state so they survive page navigation.
"""

import streamlit as st
import pandas as pd
import numpy as np
import utils
import plots


# ── helper: read session_state param with fallback ────────────────────────────
def _ss(key, default=None):
    return st.session_state.get(key, default)


def render(df: pd.DataFrame):
    st.title("🕳️ Porosity Estimation")

    num_cols = [c for c in df.columns if c != "DEPTH"]
    _nphi = utils.find_col(df, ["NPHI", "TNPH", "PHIN"])
    _rhob = utils.find_col(df, ["RHOB", "RHOZ", "DEN"])
    _dt   = utils.find_col(df, ["DT",   "DTCO", "AC",   "DTC"])
    _gr   = utils.find_col(df, ["GR",   "GRD"])

    st.markdown("""
<div class="info-box">
Density–Neutron combination gives the best total porosity (PHIT) with gas correction.
Shale correction (using Vsh from GR) yields effective porosity (PHIE) —
the pore space available to hydrocarbons.
</div>
""", unsafe_allow_html=True)

    tab_assign, tab_shale, tab_compute, tab_core = st.tabs([
        "Curve Assignment",
        "Shale Point ID",
        "Compute & Plot",
        "Core Calibration",
    ])

    # ── Tab 1 : Curve Assignment ──────────────────────────────────────────────
    with tab_assign:
        st.subheader("Assign Log Curves")

        def _sidx(col, opts):
            return opts.index(col) if col and col in opts else 0

        opts_n = ["None"] + num_cols
        rhob_c = st.selectbox("RHOB (g/cc)",    opts_n, key="por_sel_rhob",
                               index=_sidx(_rhob, opts_n))
        nphi_c = st.selectbox("NPHI (v/v or %)", opts_n, key="por_sel_nphi",
                               index=_sidx(_nphi, opts_n))
        dt_c   = st.selectbox("DT (µs/ft)",     opts_n, key="por_sel_dt",
                               index=_sidx(_dt,   opts_n))
        gr_c   = st.selectbox("GR (API)",        opts_n, key="por_sel_gr",
                               index=_sidx(_gr,   opts_n))

        # Persist to session_state
        st.session_state["por_rhob_col"] = None if rhob_c == "None" else rhob_c
        st.session_state["por_nphi_col"] = None if nphi_c == "None" else nphi_c
        st.session_state["por_dt_col"]   = None if dt_c   == "None" else dt_c
        st.session_state["por_gr_col"]   = None if gr_c   == "None" else gr_c

        st.divider()
        st.subheader("Matrix & Fluid Parameters")

        with st.expander("Density Porosity Parameters", expanded=True):
            dc1, dc2 = st.columns(2)
            rho_ma = dc1.number_input(
                "Matrix density ρma (g/cc)",
                value=float(_ss("por_rho_matrix", 2.65)),
                step=0.01, format="%.2f", key="por_ni_rhoma",
                help="Sandstone 2.65  |  Limestone 2.71  |  Dolomite 2.87",
            )
            rho_fl = dc2.number_input(
                "Fluid density ρf (g/cc)",
                value=float(_ss("por_rho_fluid", 1.0)),
                step=0.01, format="%.2f", key="por_ni_rhofl",
                help="Freshwater 1.0  |  Saltwater 1.1  |  Gas ~0.7",
            )
            st.session_state["por_rho_matrix"] = rho_ma
            st.session_state["por_rho_fluid"]  = rho_fl

        with st.expander("Sonic Porosity Parameters (Wyllie Time-Average)"):
            sc1, sc2 = st.columns(2)
            dt_ma = sc1.number_input(
                "Matrix travel time Δtma (µs/ft)",
                value=float(_ss("por_dt_matrix", 55.5)),
                step=0.5, format="%.1f", key="por_ni_dtma",
                help="Sandstone 55.5  |  Limestone 47.5  |  Dolomite 43.5",
            )
            dt_fl = sc2.number_input(
                "Fluid travel time Δtf (µs/ft)",
                value=float(_ss("por_dt_fluid", 189.0)),
                step=1.0, format="%.1f", key="por_ni_dtfl",
            )
            st.session_state["por_dt_matrix"] = dt_ma
            st.session_state["por_dt_fluid"]  = dt_fl

    # ── Tab 2 : Shale Point ID ────────────────────────────────────────────────
    with tab_shale:
        st.subheader("Interactive Shale Point Identification")
        st.markdown(
            "Filter depth samples to the probable shale region.  "
            "The estimated PHIDsh / PHINsh are auto-filled in the Compute tab.  \n\n"
            "Typical shale zone: **30–40 p.u.** neutron and **2.35–2.50 g/cc** density "
            "(after Schlumberger / IIT ISM lecture notes)."
        )

        rhob_col = _ss("por_rhob_col")
        nphi_col = _ss("por_nphi_col")
        dt_col   = _ss("por_dt_col")

        shale_mask = pd.Series(True, index=df.index)

        if rhob_col and rhob_col in df.columns:
            r1, r2 = st.columns(2)
            rmin = r1.number_input("RHOB min (g/cc)", value=2.35, step=0.01, key="sh_rmin")
            rmax = r2.number_input("RHOB max (g/cc)", value=2.50, step=0.01, key="sh_rmax")
            shale_mask = shale_mask & (df[rhob_col] >= rmin) & (df[rhob_col] <= rmax)

        if nphi_col and nphi_col in df.columns:
            nphi_data = df[nphi_col].dropna()
            is_pct    = nphi_data.median() > 1.0
            n1, n2    = st.columns(2)
            if is_pct:
                nmin = n1.number_input("NPHI min (p.u.)", value=30.0, step=1.0, key="sh_nmin")
                nmax = n2.number_input("NPHI max (p.u.)", value=40.0, step=1.0, key="sh_nmax")
                shale_mask = shale_mask & (df[nphi_col] >= nmin) & (df[nphi_col] <= nmax)
            else:
                nmin = n1.number_input("NPHI min (v/v)", value=0.30, step=0.01, key="sh_nmin")
                nmax = n2.number_input("NPHI max (v/v)", value=0.40, step=0.01, key="sh_nmax")
                shale_mask = shale_mask & (df[nphi_col] >= nmin) & (df[nphi_col] <= nmax)

        if dt_col and dt_col in df.columns:
            use_dt = st.checkbox("Also filter by DT", value=False, key="sh_use_dt")
            if use_dt:
                d1, d2 = st.columns(2)
                dtmin = d1.number_input("DT min (µs/ft)", value=70.0, step=1.0, key="sh_dtmin")
                dtmax = d2.number_input("DT max (µs/ft)", value=100.0, step=1.0, key="sh_dtmax")
                shale_mask = shale_mask & (df[dt_col] >= dtmin) & (df[dt_col] <= dtmax)

        n_sh = int(shale_mask.sum())
        st.info(f"Shale filter: **{n_sh}** samples selected ({n_sh / max(len(df), 1) * 100:.1f}%)")

        if n_sh > 0:
            rma = _ss("por_rho_matrix", 2.65)
            rfl = _ss("por_rho_fluid",  1.0)
            ka, kb = st.columns(2)
            if rhob_col and rhob_col in df.columns:
                phid_sh = float(
                    utils.density_porosity(df.loc[shale_mask, rhob_col], rma, rfl).mean()
                )
                ka.metric("Estimated PHIDsh", f"{phid_sh:.4f}")
                st.session_state["por_phid_sh"] = phid_sh
            if nphi_col and nphi_col in df.columns:
                phin_sh = float(
                    utils.neutron_porosity(df.loc[shale_mask, nphi_col]).mean()
                )
                kb.metric("Estimated PHINsh", f"{phin_sh:.4f}")
                st.session_state["por_phin_sh"] = phin_sh

        # Visualise on N-D crossplot
        if rhob_col and nphi_col \
           and rhob_col in df.columns and nphi_col in df.columns:
            import plotly.graph_objects as go
            fig_sh = plots.plot_nphi_rhob(df, nphi_col, rhob_col, show_lines=True)
            nphi_sh = df.loc[shale_mask, nphi_col].copy()
            if not nphi_sh.empty and nphi_sh.dropna().median() > 1.0:
                nphi_sh = nphi_sh / 100.0
            fig_sh.add_trace(go.Scatter(
                x=nphi_sh, y=df.loc[shale_mask, rhob_col],
                mode="markers", name="Shale Points",
                marker=dict(color="#795548", size=6, symbol="circle",
                            line=dict(color="white", width=0.5)),
            ))
            st.plotly_chart(fig_sh, use_container_width=True,
                key="porosity_pc1")

    # ── Tab 3 : Compute & Plot ────────────────────────────────────────────────
    with tab_compute:
        st.subheader("Shale Correction & Porosity Computation")

        rhob_col = _ss("por_rhob_col")
        nphi_col = _ss("por_nphi_col")
        dt_col   = _ss("por_dt_col")
        gr_col   = _ss("por_gr_col")
        rho_ma   = _ss("por_rho_matrix", 2.65)
        rho_fl   = _ss("por_rho_fluid",  1.0)
        dt_ma    = _ss("por_dt_matrix",  55.5)
        dt_fl    = _ss("por_dt_fluid",   189.0)

        apply_vsh = st.checkbox(
            "Apply shale correction (requires GR)",
            value=bool(gr_col),
            key="por_apply_vsh",
        )

        gr_clean, gr_shale, phid_sh, phin_sh = 20.0, 120.0, 0.10, 0.30

        if apply_vsh and gr_col and gr_col in df.columns:
            gr_data   = df[gr_col].dropna()
            vc1, vc2  = st.columns(2)
            gr_clean  = vc1.number_input(
                "GR clean (API)", value=float(round(gr_data.quantile(0.05), 1)), key="por_gr_clean")
            gr_shale  = vc2.number_input(
                "GR shale (API)", value=float(round(gr_data.quantile(0.95), 1)), key="por_gr_shale")
            st.divider()
            sc1, sc2 = st.columns(2)
            phid_sh = sc1.number_input(
                "PHIDsh", value=float(round(_ss("por_phid_sh", 0.10), 4)),
                step=0.01, format="%.4f", key="por_phid_sh_ni")
            phin_sh = sc2.number_input(
                "PHINsh", value=float(round(_ss("por_phin_sh", 0.30), 4)),
                step=0.01, format="%.4f", key="por_phin_sh_ni")

        if st.button("🔄 Compute Porosity", type="primary", key="por_compute_btn"):
            with st.spinner("Calculating …"):
                df_upd = st.session_state.df.copy()
                computed = []

                if apply_vsh and gr_col and gr_col in df_upd.columns:
                    df_upd["VSH"] = utils.compute_vshale_gr(
                        df_upd[gr_col], gr_clean, gr_shale)
                    computed.append("VSH")

                if rhob_col and rhob_col in df_upd.columns:
                    df_upd["PHID"] = utils.density_porosity(
                        df_upd[rhob_col], rho_ma, rho_fl)
                    computed.append("PHID")

                if nphi_col and nphi_col in df_upd.columns:
                    df_upd["PHIN"] = utils.neutron_porosity(df_upd[nphi_col])
                    computed.append("PHIN")

                if dt_col and dt_col in df_upd.columns:
                    df_upd["PHIS"] = utils.sonic_porosity(df_upd[dt_col], dt_ma, dt_fl)
                    computed.append("PHIS")

                # Total porosity
                if "PHID" in df_upd.columns and "PHIN" in df_upd.columns:
                    df_upd["PHIT"] = utils.total_porosity(df_upd["PHID"], df_upd["PHIN"])
                elif "PHID" in df_upd.columns:
                    df_upd["PHIT"] = df_upd["PHID"].copy()
                elif "PHIN" in df_upd.columns:
                    df_upd["PHIT"] = df_upd["PHIN"].copy()
                if "PHIT" in df_upd.columns:
                    computed.append("PHIT")

                # Effective porosity
                if "PHIT" in df_upd.columns:
                    if apply_vsh and "VSH" in df_upd.columns:
                        df_upd["PHIE"] = utils.effective_porosity(
                            df_upd["PHIT"], df_upd["VSH"], phid_sh, phin_sh)
                    else:
                        df_upd["PHIE"] = df_upd["PHIT"].copy()
                    computed.append("PHIE")

                st.session_state.df      = df_upd
                st.session_state.por_done = True
                st.success(f"✅ Computed: {', '.join(computed)}")

        # Live plot — always show if por_done
        df_cur = st.session_state.df
        if _ss("por_done") and "PHIT" in df_cur.columns:
            st.subheader("Porosity Log Tracks")
            st.plotly_chart(
                plots.plot_porosity(
                    df_cur,
                    phid_col="PHID" if "PHID" in df_cur.columns else None,
                    phin_col="PHIN" if "PHIN" in df_cur.columns else None,
                    phis_col="PHIS" if "PHIS" in df_cur.columns else None,
                    phit_col="PHIT",
                    phie_col="PHIE" if "PHIE" in df_cur.columns else None,
                ),
                use_container_width=True,
                key="porosity_pc2",
            )
            por_cols = [c for c in ["VSH","PHID","PHIN","PHIS","PHIT","PHIE"]
                        if c in df_cur.columns]
            st.subheader("Statistics")
            st.dataframe(
                df_cur[por_cols].describe().T.style.format("{:.4f}"),
                use_container_width=True,
            )
        else:
            st.info("Press **Compute Porosity** above.")

    # ── Tab 4 : Core Calibration ──────────────────────────────────────────────
    with tab_core:
        st.subheader("Core Data Upload & Calibration")
        st.markdown(
            "Upload a CSV with at minimum two columns: **Depth** and **Core Porosity**.  \n"
            "The app interpolates core data to the log depth grid and fits a linear calibration."
        )

        core_file = st.file_uploader("Upload core CSV", type=["csv"], key="por_core_upload")
        if core_file is not None:
            try:
                core_df = pd.read_csv(core_file)
                st.session_state.core_df = core_df
                st.success(f"Loaded {len(core_df)} core samples.  Columns: {list(core_df.columns)}")
            except Exception as exc:
                st.error(f"Could not read file: {exc}")

        if st.session_state.get("core_df") is not None:
            core_df = st.session_state.core_df
            cc1, cc2 = st.columns(2)
            c_depth = cc1.selectbox("Depth column",   core_df.columns, key="por_core_depth")
            c_phi   = cc2.selectbox("Porosity column", core_df.columns,
                                     index=min(1, len(core_df.columns)-1), key="por_core_phi")

            df_cur = st.session_state.df
            log_phi_col = utils.find_col(df_cur, ["PHIE", "PHIT"])

            if log_phi_col:
                core_interp = utils.interpolate_core(
                    core_df, df_cur["DEPTH"], c_depth, c_phi)
                df_cur["CORE_PHI"] = core_interp.values
                n_match = int(df_cur[["CORE_PHI", log_phi_col]].dropna().__len__())
                st.info(f"Matched **{n_match}** depth points (log ↔ core).")

                if n_match >= 3:
                    slope, intercept, corrected = utils.linear_calibration(
                        df_cur[log_phi_col], df_cur["CORE_PHI"])
                    st.markdown(
                        f"**Calibration:** `Core_PHI = {slope:.4f} × {log_phi_col} + {intercept:.4f}`"
                    )
                    st.plotly_chart(
                        plots.plot_core_vs_log(
                            df_cur[log_phi_col], df_cur["CORE_PHI"], slope, intercept),
                        use_container_width=True,
                key="porosity_pc3",
                    )
                    if st.button("Apply calibration to PHIE", key="por_core_apply"):
                        df_upd = st.session_state.df.copy()
                        df_upd["PHIE"] = corrected.values
                        st.session_state.df = df_upd
                        st.success("✅ PHIE updated with core-calibrated values.")
                else:
                    st.warning("Need at least 3 overlapping depth points for calibration.")
            else:
                st.warning("Compute PHIE or PHIT first (Compute & Plot tab).")
        else:
            st.info("Upload a core CSV file above.")
