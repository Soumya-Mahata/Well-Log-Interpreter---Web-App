"""
porosity.py  —  Porosity Estimation Module
============================================
Follows formation_evaluation reference notebook (Practical 3).
All widget keys prefixed "por_".
Parameters persisted in session_state.

Key improvements:
- Density-derived porosity after matrix & fluid selection
- Sandstone calibration step (NPHI limestone → sandstone correction)
- Shale point via percentile-based selection (Vsh + bad-hole mask)
- Computes PHID_sh and PHI_sh from selected shale region
- Removes manual shale point input
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import utils
import plots


# ── helper: read session_state param with fallback ────────────────────────────
def _ss(key, default=None):
    return st.session_state.get(key, default)


def render(df: pd.DataFrame):
    st.title("🕳️ Porosity Estimation")

    # Always use the latest session state version
    df = st.session_state.df

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
        st.caption(
            "Select the expected matrix lithology to auto-fill parameters, "
            "or enter custom values below."
        )

        # Lithology presets
        _MATRIX_PRESETS = {
            "Sandstone":  (2.65, 55.5),
            "Limestone":  (2.71, 47.5),
            "Dolomite":   (2.87, 43.5),
            "Custom":     (None, None),
        }

        # Track last preset to detect change
        _prev_preset = _ss("_por_prev_preset", "Sandstone")
        preset_choice = st.radio(
            "Matrix lithology",
            list(_MATRIX_PRESETS.keys()),
            horizontal=True,
            key="por_matrix_preset",
        )
        preset_rho_ma, preset_dt_ma = _MATRIX_PRESETS[preset_choice]

        # If preset changed and is not Custom, push values into session_state
        # BEFORE the number_input widgets render so they pick up the new value
        if preset_choice != _prev_preset and preset_rho_ma is not None:
            st.session_state["por_rho_matrix"]   = preset_rho_ma
            st.session_state["por_dt_matrix"]     = preset_dt_ma
            st.session_state["por_ni_rhoma"]      = preset_rho_ma
            st.session_state["por_ni_dtma"]       = preset_dt_ma
            st.session_state["_por_prev_preset"]  = preset_choice
            st.rerun()
        st.session_state["_por_prev_preset"] = preset_choice

        with st.expander("Density Porosity Parameters", expanded=True):
            dc1, dc2 = st.columns(2)
            rho_ma = dc1.number_input(
                "Matrix density ρma (g/cc)",
                min_value=1.0, max_value=4.0,
                step=0.01, format="%.2f", key="por_ni_rhoma",
                help="Sandstone 2.65  |  Limestone 2.71  |  Dolomite 2.87",
            )
            rho_fl = dc2.number_input(
                "Fluid density ρf (g/cc)",
                value=float(_ss("por_rho_fluid", 1.0)),
                min_value=0.5, max_value=1.5,
                step=0.01, format="%.2f", key="por_ni_rhofl",
                help="Freshwater 1.0  |  Saltwater 1.1  |  Gas ~0.7",
            )
            st.session_state["por_rho_matrix"] = rho_ma
            st.session_state["por_rho_fluid"]  = rho_fl

        with st.expander("Sandstone Calibration (Neutron)"):
            st.markdown(
                "When NPHI is calibrated to **limestone**, apply a correction to "
                "convert to sandstone equivalence.  "
                "Standard correction: **+0.04 v/v** (Bateman & Konen, 1977)."
            )
            apply_ss_cal = st.checkbox(
                "Apply sandstone calibration to NPHI",
                value=False, key="por_ss_cal",
            )
            ss_corr = st.number_input(
                "Limestone→Sandstone NPHI correction (v/v)",
                value=0.04, step=0.005, format="%.3f",
                key="por_ss_corr",
                help="Add this value to NPHI to convert limestone-calibrated to sandstone-calibrated",
                disabled=not apply_ss_cal,
            )
            st.session_state["por_apply_ss_cal"] = apply_ss_cal
            st.session_state["por_ss_corr"]      = ss_corr

        with st.expander("Sonic Porosity Parameters (Wyllie Time-Average)"):
            sc1, sc2 = st.columns(2)
            dt_ma = sc1.number_input(
                "Matrix travel time Δtma (µs/ft)",
                min_value=30.0, max_value=100.0,
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
        st.subheader("Percentile-Based Shale Point Identification")
        st.markdown("""
Shale points are automatically identified using **Vsh percentile thresholding**
(instead of manual RHOB/NPHI range entry).

The shale region is defined as samples where `Vsh > Vsh threshold percentile`
and the hole quality is acceptable (CALI-based bad-hole mask excluded).

Reference: Formation Evaluation Practical (Mandal 2026, IIT ISM).
        """)

        rhob_col = _ss("por_rhob_col")
        nphi_col = _ss("por_nphi_col")
        gr_col   = _ss("por_gr_col")

        # Shale mask method selection
        sh_method = st.radio(
            "Shale detection method",
            ["Vsh percentile (recommended)", "RHOB + NPHI range (manual)"],
            horizontal=True, key="sh_method_radio",
        )

        shale_mask = pd.Series(True, index=df.index)

        if sh_method.startswith("Vsh"):
            # Percentile-based using VSH column if available, else compute on-the-fly
            if "VSH" in df.columns:
                vsh_series = df["VSH"]
                st.success("Using precomputed **VSH** column from Lithology module.")
            elif gr_col and gr_col in df.columns:
                gr_data = df[gr_col]
                p5  = float(gr_data.quantile(0.05))
                p95 = float(gr_data.quantile(0.95))
                denom = (p95 - p5) if abs(p95 - p5) > 1e-6 else 1.0
                vsh_series = ((gr_data - p5) / denom).clip(0, 1)
                st.info(f"VSH computed from GR on-the-fly (p5={p5:.1f}, p95={p95:.1f}).")
            else:
                vsh_series = None
                st.warning("No VSH column and no GR curve assigned. Select GR in Curve Assignment.")

            if vsh_series is not None:
                vsh_pct = st.slider(
                    "Vsh threshold percentile (shale region = Vsh ≥ this)",
                    min_value=50, max_value=95, value=80, step=5,
                    key="sh_vsh_pct",
                    help="80th percentile recommended (Mandal 2026)",
                )
                vsh_threshold = float(vsh_series.quantile(vsh_pct / 100))
                shale_mask = vsh_series >= vsh_threshold
                st.info(
                    f"Vsh threshold (p{vsh_pct}): **{vsh_threshold:.3f}** — "
                    f"**{int(shale_mask.sum())}** shale samples selected "
                    f"({shale_mask.mean()*100:.1f}%)"
                )
                # Fallback if too few samples
                if shale_mask.sum() < 5:
                    shale_mask = vsh_series >= 0.6
                    st.warning(f"Too few samples at p{vsh_pct}, falling back to Vsh ≥ 0.6 "
                               f"({int(shale_mask.sum())} samples).")

        else:
            # Manual RHOB + NPHI range
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

            n_sh = int(shale_mask.sum())
            st.info(f"Shale filter: **{n_sh}** samples selected ({n_sh / max(len(df), 1) * 100:.1f}%)")

        n_sh = int(shale_mask.sum())

        # Compute shale point values
        if n_sh > 0:
            rma = _ss("por_rho_matrix", 2.65)
            rfl = _ss("por_rho_fluid",  1.0)
            apply_ss = _ss("por_apply_ss_cal", False)
            ss_cor   = _ss("por_ss_corr", 0.04)

            ka, kb = st.columns(2)
            phid_sh = phin_sh = None

            if rhob_col and rhob_col in df.columns:
                phid_sh = float(
                    utils.density_porosity(df.loc[shale_mask, rhob_col], rma, rfl).mean()
                )
                ka.metric("PHID_sh (density porosity at shale)", f"{phid_sh:.4f}")
                st.session_state["por_phid_sh"] = phid_sh

            if nphi_col and nphi_col in df.columns:
                nphi_sh_raw = utils.neutron_porosity(df.loc[shale_mask, nphi_col])
                if apply_ss:
                    nphi_sh_raw = (nphi_sh_raw + ss_cor).clip(0, 1)
                phin_sh = float(nphi_sh_raw.mean())
                kb.metric("PHI_sh (neutron porosity at shale)", f"{phin_sh:.4f}")
                st.session_state["por_phin_sh"] = phin_sh

            # N-D crossplot with shale points highlighted
            if rhob_col and nphi_col \
               and rhob_col in df.columns and nphi_col in df.columns:
                fig_sh = plots.plot_nphi_rhob(df, nphi_col, rhob_col, show_lines=True)
                nphi_sh = df.loc[shale_mask, nphi_col].copy()
                if not nphi_sh.empty and nphi_sh.dropna().median() > 1.0:
                    nphi_sh = nphi_sh / 100.0
                if apply_ss:
                    nphi_sh = (nphi_sh + ss_cor).clip(0, 1)

                fig_sh.add_trace(go.Scatter(
                    x=nphi_sh, y=df.loc[shale_mask, rhob_col],
                    mode="markers", name="Shale Points",
                    marker=dict(color="#795548", size=7, symbol="square",
                                line=dict(color="black", width=1)),
                ))

                # Annotate shale point
                if phid_sh is not None and phin_sh is not None:
                    fig_sh.add_trace(go.Scatter(
                        x=[phin_sh], y=[phid_sh],
                        mode="markers", name="Shale Point (mean)",
                        marker=dict(color="#000000", size=12, symbol="square",
                                    line=dict(color="white", width=1.5)),
                    ))
                    fig_sh.add_annotation(
                        x=phin_sh + 0.02, y=phid_sh + 0.02,
                        text=f"<b>Shale<br>Φ_Nsh={phin_sh:.2f}<br>Φ_Dsh={phid_sh:.2f}</b>",
                        showarrow=True, arrowhead=2, arrowcolor="#333",
                        font=dict(size=10, color="#212121"),
                        bgcolor="rgba(255,255,255,0.90)",
                        bordercolor="#795548", borderwidth=1,
                    )

                st.plotly_chart(fig_sh, use_container_width=True, key="porosity_pc1")
        else:
            st.warning("No shale samples selected with current settings.")

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
        apply_ss = _ss("por_apply_ss_cal", False)
        ss_cor   = _ss("por_ss_corr", 0.04)

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
                "PHIDsh (from Shale Point ID tab)",
                value=float(round(_ss("por_phid_sh", 0.10), 4)),
                step=0.01, format="%.4f", key="por_phid_sh_ni",
                help="Auto-filled from Shale Point ID tab. Edit if needed.",
            )
            phin_sh = sc2.number_input(
                "PHINsh (from Shale Point ID tab)",
                value=float(round(_ss("por_phin_sh", 0.30), 4)),
                step=0.01, format="%.4f", key="por_phin_sh_ni",
                help="Auto-filled from Shale Point ID tab. Edit if needed.",
            )

        if apply_ss and nphi_col:
            st.info(f"Sandstone calibration active: NPHI + {ss_cor:.3f} v/v (limestone→sandstone)")

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
                    nphi_raw = utils.neutron_porosity(df_upd[nphi_col])
                    if apply_ss:
                        nphi_raw = (nphi_raw + ss_cor).clip(0, 1)
                        df_upd["NPSS"] = nphi_raw.values   # sandstone-calibrated NPHI
                        computed.append("NPSS")
                    df_upd["PHIN"] = nphi_raw.values
                    computed.append("PHIN")

                if dt_col and dt_col in df_upd.columns:
                    df_upd["PHIS"] = utils.sonic_porosity(df_upd[dt_col], dt_ma, dt_fl)
                    computed.append("PHIS")

                # Total porosity — gas correction (quadratic mean when PHID > PHIN)
                if "PHID" in df_upd.columns and "PHIN" in df_upd.columns:
                    phid = df_upd["PHID"]
                    phin = df_upd["PHIN"]
                    gas_flag = phid > phin
                    df_upd["PHIT"] = np.where(
                        gas_flag,
                        np.sqrt((phin ** 2 + phid ** 2) / 2),
                        (phin + phid) / 2,
                    ).clip(0, 0.5)
                elif "PHID" in df_upd.columns:
                    df_upd["PHIT"] = df_upd["PHID"].clip(0, 0.5)
                elif "PHIN" in df_upd.columns:
                    df_upd["PHIT"] = df_upd["PHIN"].clip(0, 0.5)
                if "PHIT" in df_upd.columns:
                    computed.append("PHIT")

                # Effective porosity (shale-corrected)
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

            # Use NPSS if sandstone calibration was applied, else PHIN
            phin_display = "NPSS" if ("NPSS" in df_cur.columns) else \
                           ("PHIN" if "PHIN" in df_cur.columns else None)

            st.plotly_chart(
                plots.plot_porosity(
                    df_cur,
                    phid_col="PHID"         if "PHID"  in df_cur.columns else None,
                    phin_col=phin_display,
                    phis_col="PHIS"         if "PHIS"  in df_cur.columns else None,
                    phit_col="PHIT",
                    phie_col="PHIE"         if "PHIE"  in df_cur.columns else None,
                ),
                use_container_width=True,
                key="porosity_pc2",
            )

            # ── Density–Neutron crossplot (following notebook) ────────────────
            if "PHID" in df_cur.columns and phin_display and phin_display in df_cur.columns:
                st.subheader("Density–Neutron Crossplot")
                phid_v = df_cur["PHID"]
                phin_v = df_cur[phin_display]
                gas_flag = phid_v > phin_v
                gas_pct  = float(gas_flag.mean() * 100)
                st.caption(
                    f"Orange points = gas crossover (DPHI > NPHI): **{gas_pct:.1f}%** of interval"
                )
                dn_fig = go.Figure()
                dn_fig.add_trace(go.Scatter(
                    x=phin_v[~gas_flag], y=phid_v[~gas_flag],
                    mode="markers", name="Liquid zone",
                    marker=dict(color="#1565C0", size=4, opacity=0.55),
                ))
                dn_fig.add_trace(go.Scatter(
                    x=phin_v[gas_flag], y=phid_v[gas_flag],
                    mode="markers", name="Gas crossover",
                    marker=dict(color="#E65100", size=5, opacity=0.70),
                ))
                # 1:1 clean formation line
                phi_line = np.linspace(0, 0.60, 100)
                dn_fig.add_trace(go.Scatter(
                    x=phi_line, y=phi_line,
                    mode="lines", name="Clean formation (Vsh=0)",
                    line=dict(color="#2E7D32", width=2, dash="dash"),
                ))
                # Shale point if available
                phid_sh = _ss("por_phid_sh")
                phin_sh = _ss("por_phin_sh")
                if phid_sh and phin_sh:
                    dn_fig.add_trace(go.Scatter(
                        x=[phin_sh], y=[phid_sh],
                        mode="markers", name="Shale point",
                        marker=dict(color="#795548", size=12, symbol="square",
                                    line=dict(color="black", width=1.5)),
                    ))
                    dn_fig.add_annotation(
                        x=phin_sh + 0.02, y=phid_sh + 0.02,
                        text=f"<b>Shale<br>NPHI={phin_sh:.2f}<br>DPHI={phid_sh:.2f}</b>",
                        showarrow=True, arrowhead=2,
                        font=dict(size=10, color="#212121"),
                        bgcolor="rgba(255,255,255,0.90)",
                        bordercolor="#795548", borderwidth=1,
                    )
                dn_fig.update_xaxes(title_text=f"{phin_display} (v/v)", range=[0, 0.65],
                                    showgrid=True, gridcolor="#d4d4d4",
                                    title_font=dict(color="#212121"),
                                    tickfont=dict(color="#212121"))
                dn_fig.update_yaxes(title_text="DPHI (v/v)", range=[0, 0.65],
                                    showgrid=True, gridcolor="#d4d4d4",
                                    title_font=dict(color="#212121"),
                                    tickfont=dict(color="#212121"))
                dn_fig.update_layout(
                    title=dict(text="<b>Density–Neutron Crossplot</b>",
                               font=dict(size=14, color="#0D2B5E")),
                    height=480, plot_bgcolor="white", paper_bgcolor="white",
                    legend=dict(font=dict(color="#212121")),
                )
                st.plotly_chart(dn_fig, use_container_width=True, key="porosity_dn_xplot")

            por_cols = [c for c in ["VSH", "PHID", "PHIN", "NPSS", "PHIS", "PHIT", "PHIE"]
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
            "Upload a CSV with at minimum two columns: **Depth** and **Core Porosity**.  \\n"
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
