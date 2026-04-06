"""
lithology.py  —  Lithology Identification Module
==================================================
Six standard crossplots + custom plot + K-means clustering.
All widget keys are unique and prefixed with "lit_" to avoid collisions.
Computed M, N, RHOMAA, DTMAA, UMAA columns are stored in session_state.df.
"""

import streamlit as st
import pandas as pd
import numpy as np
import utils
import plots


def render(df: pd.DataFrame):
    st.title("🪨 Lithology Identification")

    num_cols = [c for c in df.columns if c != "DEPTH"]

    # Auto-detect standard curves
    _nphi = utils.find_col(df, ["NPHI", "TNPH", "PHIN", "NPH"])
    _rhob = utils.find_col(df, ["RHOB", "RHOZ", "DEN"])
    _dt   = utils.find_col(df, ["DT",   "DTCO", "AC",   "DTC"])
    _gr   = utils.find_col(df, ["GR",   "GRD"])
    _pe   = utils.find_col(df, ["PE",   "PEF",  "PEFZ"])

    st.markdown("""
<div class="info-box">
Crossplots discriminate lithology using the different responses of neutron, density,
sonic, and photoelectric tools.  Reference lines and mineral points follow
<b>Schlumberger Log Interpretation Charts (2005b)</b> and IIT ISM lecture notes (Mandal 2026).
</div>
""", unsafe_allow_html=True)

    color_opts = ["None"] + num_cols

    def _idx(col, opts):
        """Safe index lookup for selectbox defaults."""
        try:
            return opts.index(col) if col in opts else 0
        except ValueError:
            return 0

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "NPHI–RHOB",
        "NPHI–Sonic",
        "Density–Sonic",
        "M-N Plot",
        "MID (Δtmaa / ρmaa)",
        "MID (Umaa / ρmaa)",
        "Custom + K-Means",
    ])

    # ── Tab 1 : Neutron–Density ───────────────────────────────────────────────
    with tab1:
        st.subheader("Neutron–Density Crossplot")
        st.caption(
            "NPHI (x) vs RHOB (y).  Gas effect shifts points left and down "
            "(below the sandstone line).  Shale zone: 30–40 p.u. / 2.35–2.50 g/cc."
        )
        c1, c2, c3 = st.columns(3)
        nphi_nd  = c1.selectbox("NPHI", ["None"] + num_cols, key="lit_nd_nphi",
                                 index=_idx(_nphi, ["None"] + num_cols))
        rhob_nd  = c2.selectbox("RHOB", ["None"] + num_cols, key="lit_nd_rhob",
                                 index=_idx(_rhob, ["None"] + num_cols))
        col_nd   = c3.selectbox("Colour by", color_opts, key="lit_nd_color",
                                 index=_idx(_gr, color_opts))
        lines_nd = st.checkbox("Show lithology lines", value=True, key="lit_nd_lines")

        if nphi_nd != "None" and rhob_nd != "None" \
           and nphi_nd in df.columns and rhob_nd in df.columns:
            st.plotly_chart(
                plots.plot_nphi_rhob(
                    df, nphi_nd, rhob_nd,
                    color_col=(None if col_nd == "None" else col_nd),
                    show_lines=lines_nd,
                ),
                use_container_width=True,
                key="lithology_pc1",
            )
        else:
            st.info("Select NPHI and RHOB curves above.")

    # ── Tab 2 : Neutron–Sonic ────────────────────────────────────────────────
    with tab2:
        st.subheader("Neutron–Sonic Crossplot")
        st.caption(
            "NPHI (x) vs DT (y).  Preferred when density quality is degraded (washout).  "
            "Shale zone: 30–40 p.u. / 70–100 µs/ft."
        )
        c1, c2, c3 = st.columns(3)
        nphi_ns  = c1.selectbox("NPHI", ["None"] + num_cols, key="lit_ns_nphi",
                                 index=_idx(_nphi, ["None"] + num_cols))
        dt_ns    = c2.selectbox("DT",   ["None"] + num_cols, key="lit_ns_dt",
                                 index=_idx(_dt,   ["None"] + num_cols))
        col_ns   = c3.selectbox("Colour by", color_opts, key="lit_ns_color",
                                 index=_idx(_gr, color_opts))
        lines_ns = st.checkbox("Show lithology lines", value=True, key="lit_ns_lines")

        if nphi_ns != "None" and dt_ns != "None" \
           and nphi_ns in df.columns and dt_ns in df.columns:
            st.plotly_chart(
                plots.plot_nphi_dt(
                    df, nphi_ns, dt_ns,
                    color_col=(None if col_ns == "None" else col_ns),
                    show_lines=lines_ns,
                ),
                use_container_width=True,
                key="lithology_pc2",
            )
        else:
            st.info("Select NPHI and DT curves above.")

    # ── Tab 3 : Density–Sonic ────────────────────────────────────────────────
    with tab3:
        st.subheader("Density–Sonic Crossplot")
        st.caption(
            "DT (x) vs RHOB (y).  Useful for mineral identification (evaporites, carbonates)."
        )
        c1, c2, c3 = st.columns(3)
        rhob_ds  = c1.selectbox("RHOB", ["None"] + num_cols, key="lit_ds_rhob",
                                 index=_idx(_rhob, ["None"] + num_cols))
        dt_ds    = c2.selectbox("DT",   ["None"] + num_cols, key="lit_ds_dt",
                                 index=_idx(_dt,   ["None"] + num_cols))
        col_ds   = c3.selectbox("Colour by", color_opts, key="lit_ds_color",
                                 index=_idx(_gr, color_opts))
        lines_ds = st.checkbox("Show lithology lines", value=True, key="lit_ds_lines")

        if rhob_ds != "None" and dt_ds != "None" \
           and rhob_ds in df.columns and dt_ds in df.columns:
            st.plotly_chart(
                plots.plot_rhob_dt(
                    df, rhob_ds, dt_ds,
                    color_col=(None if col_ds == "None" else col_ds),
                    show_lines=lines_ds,
                ),
                use_container_width=True,
                key="lithology_pc3",
            )
        else:
            st.info("Select RHOB and DT curves above.")

    # ── Tab 4 : M-N Plot ─────────────────────────────────────────────────────
    with tab4:
        st.subheader("M-N Plot")
        st.markdown(r"""
Requires **DT + RHOB + NPHI**.  Computed from:

$$
M = \frac{\Delta t_f - \Delta t}{(\rho_b - \rho_f) \times 100}               
\qquad
N = \frac{\phi_{Nf} - \phi_N}{\rho_b - \rho_f}
$$

Reference: Schlumberger 2005b Table 24.1.
        """)
        m1, m2 = st.columns(2)
        dt_mn   = m1.selectbox("DT",   ["None"] + num_cols, key="lit_mn_dt",
                                index=_idx(_dt,   ["None"] + num_cols))
        rhob_mn = m1.selectbox("RHOB", ["None"] + num_cols, key="lit_mn_rhob",
                                index=_idx(_rhob, ["None"] + num_cols))
        nphi_mn = m2.selectbox("NPHI", ["None"] + num_cols, key="lit_mn_nphi",
                                index=_idx(_nphi, ["None"] + num_cols))
        col_mn  = m2.selectbox("Colour by", color_opts, key="lit_mn_color",
                                index=_idx(_gr, color_opts))

        with st.expander("⚙️ Fluid constants"):
            f1, f2, f3 = st.columns(3)
            dt_f_mn   = f1.number_input("Δtf (µs/ft)", value=189.0, key="lit_mn_dtf")
            rho_f_mn  = f2.number_input("ρf (g/cc)",   value=1.0,   key="lit_mn_rhof")
            phin_f_mn = f3.number_input("φNf (v/v)",   value=1.0,   key="lit_mn_phinf")

        need = [dt_mn, rhob_mn, nphi_mn]
        if all(v != "None" and v in df.columns for v in need):
            phin_s = utils.neutron_porosity(df[nphi_mn])
            M_s    = utils.compute_M(df[dt_mn],   df[rhob_mn], dt_f_mn,   rho_f_mn)
            N_s    = utils.compute_N(phin_s,       df[rhob_mn], phin_f_mn, rho_f_mn)

            df_upd = st.session_state.df.copy()
            df_upd["M_LIT"] = M_s.values
            df_upd["N_LIT"] = N_s.values
            st.session_state.df = df_upd

            st.plotly_chart(
                plots.plot_mn(df_upd, "M_LIT", "N_LIT",
                              color_col=(None if col_mn == "None" else col_mn)),
                use_container_width=True,
                key="lithology_pc4",
            )
        else:
            st.info("Select DT, RHOB, and NPHI above.")

    # ── Tab 5 : MID Plot — Δtmaa vs ρmaa ────────────────────────────────────
    with tab5:
        st.subheader("MID Plot — Δtmaa vs ρmaa")
        st.markdown(r"""
Matrix Identification Diagram — porosity- and mud-type-independent.
Requires **NPHI + RHOB + DT**.

$$
\rho_{maa} = \frac{\rho_b - \phi_{ND}\,\rho_f}{1 - \phi_{ND}}
\qquad
\Delta t_{maa} = \frac{\Delta t - \phi_{SN}\,\Delta t_f}{1 - \phi_{SN}}
$$
        """)
        m5a, m5b = st.columns(2)
        nphi_m5 = m5a.selectbox("NPHI", ["None"] + num_cols, key="lit_m5_nphi",
                                 index=_idx(_nphi, ["None"] + num_cols))
        rhob_m5 = m5a.selectbox("RHOB", ["None"] + num_cols, key="lit_m5_rhob",
                                 index=_idx(_rhob, ["None"] + num_cols))
        dt_m5   = m5b.selectbox("DT",   ["None"] + num_cols, key="lit_m5_dt",
                                 index=_idx(_dt,   ["None"] + num_cols))
        col_m5  = m5b.selectbox("Colour by", color_opts, key="lit_m5_color",
                                 index=_idx(_gr, color_opts))
        with st.expander("⚙️ Fluid constants"):
            fm1, fm2 = st.columns(2)
            rho_f_m5 = fm1.number_input("ρf (g/cc)",   value=1.0,   key="lit_m5_rhof")
            dt_f_m5  = fm2.number_input("Δtf (µs/ft)", value=189.0, key="lit_m5_dtf")

        need5 = [nphi_m5, rhob_m5, dt_m5]
        if all(v != "None" and v in df.columns for v in need5):
            phin5  = utils.neutron_porosity(df[nphi_m5])
            phid5  = utils.density_porosity(df[rhob_m5])
            phis5  = utils.sonic_porosity(df[dt_m5])
            phi_nd = utils.nd_porosity(phid5, phin5)
            phi_sn = utils.sn_porosity(phis5, phin5)
            rho_maa = utils.compute_rho_maa(df[rhob_m5], phi_nd, rho_f=rho_f_m5)
            dt_maa  = utils.compute_dt_maa(df[dt_m5],   phi_sn, dt_f=dt_f_m5)

            df_upd = st.session_state.df.copy()
            df_upd["RHOMAA"] = rho_maa.values
            df_upd["DTMAA"]  = dt_maa.values
            st.session_state.df = df_upd

            st.plotly_chart(
                plots.plot_mid_dt_rho(df_upd, "DTMAA", "RHOMAA",
                                      color_col=(None if col_m5 == "None" else col_m5)),
                use_container_width=True,
                key="lithology_pc5",
            )
        else:
            st.info("Select NPHI, RHOB, and DT above.")

    # ── Tab 6 : MID Plot — Umaa vs ρmaa ─────────────────────────────────────
    with tab6:
        st.subheader("MID Plot — Umaa vs ρmaa")
        st.markdown(r"""
Requires litho-density tool output: **RHOB + Pe + NPHI**.

$$U_{maa} = \frac{\rho_b \cdot P_e - \phi_{ND} \cdot U_f}{1 - \phi_{ND}}$$

Uf (freshwater) = 0.398 barns/cm³.  
Gas shifts ρmaa upward; barite shifts Umaa to the right.
        """)
        m6a, m6b = st.columns(2)
        rhob_m6 = m6a.selectbox("RHOB", ["None"] + num_cols, key="lit_m6_rhob",
                                 index=_idx(_rhob, ["None"] + num_cols))
        pe_m6   = m6a.selectbox("Pe (photoelectric)", ["None"] + num_cols, key="lit_m6_pe",
                                 index=_idx(_pe, ["None"] + num_cols))
        nphi_m6 = m6b.selectbox("NPHI", ["None"] + num_cols, key="lit_m6_nphi",
                                 index=_idx(_nphi, ["None"] + num_cols))
        col_m6  = m6b.selectbox("Colour by", color_opts, key="lit_m6_color",
                                 index=_idx(_gr, color_opts))
        u_f     = st.number_input("Uf — fluid photoelectric (barns/cm³)",
                                   value=0.398, step=0.001, format="%.3f", key="lit_m6_uf")

        need6 = [rhob_m6, pe_m6, nphi_m6]
        if all(v != "None" and v in df.columns for v in need6):
            phin6  = utils.neutron_porosity(df[nphi_m6])
            phid6  = utils.density_porosity(df[rhob_m6])
            phi_nd6 = utils.nd_porosity(phid6, phin6)
            rho_maa6 = utils.compute_rho_maa(df[rhob_m6], phi_nd6)
            u_maa6   = utils.compute_U_maa(df[rhob_m6], df[pe_m6], phi_nd6, U_f=u_f)

            df_upd = st.session_state.df.copy()
            df_upd["RHOMAA"] = rho_maa6.values
            df_upd["UMAA"]   = u_maa6.values
            st.session_state.df = df_upd

            st.plotly_chart(
                plots.plot_mid_u_rho(df_upd, "UMAA", "RHOMAA",
                                     color_col=(None if col_m6 == "None" else col_m6)),
                use_container_width=True,
                key="lithology_pc6",
            )
        else:
            st.info(
                "Pe (photoelectric) log not found or not selected.  "
                "This plot requires a litho-density tool output."
            )

    # ── Tab 7 : Custom + K-Means ─────────────────────────────────────────────
    with tab7:
        st.subheader("Custom Crossplot")
        c1, c2, c3 = st.columns(3)
        x_cx  = c1.selectbox("X axis", num_cols, key="lit_cx_x",
                            index=_idx(_nphi, num_cols) if _nphi else 0)
        y_cx  = c2.selectbox("Y axis", num_cols, key="lit_cx_y",
                            index=_idx(_rhob, num_cols) if _rhob else 1)
        col_cx = c3.selectbox("Colour by", color_opts, key="lit_cx_color",
                            index=_idx(_gr, color_opts))
        invert_y_main = st.checkbox("Invert Y-axis (Main Crossplot)", key="lit_inv_y_main")

        # Use cluster column if it already exists
        df_cur     = st.session_state.df
        has_cluster = "CLUSTER" in df_cur.columns

        fig_cx = plots.plot_crossplot(
            df_cur,
            x_cx,
            y_cx,
            color_col=(None if col_cx == "None" else col_cx),
            cluster_col=None,   # ✅ FIX: removed automatic CLUSTER coupling
            title=f"{x_cx}  vs  {y_cx}",
        )

        if invert_y_main:
            fig_cx.update_yaxes(autorange="reversed")

        st.plotly_chart(fig_cx, use_container_width=True,
                        key="lithology_pc7")

        st.divider()
        st.subheader("K-Means Lithology Clustering")
        st.caption(
            "Groups depth samples with similar log signatures into N lithofacies classes.  "
            "Results are stored as column **CLUSTER** and used to colour the crossplot above."
        )

        feats = st.multiselect(
            "Features",
            num_cols,
            default=[c for c in [_nphi, _rhob, _gr, _dt] if c],
            key="lit_km_feats",
        )
        n_cl = st.slider("Number of clusters", 2, 8, 4, key="lit_km_n")

        if st.button("Run K-Means", key="lit_km_run", type="primary"):
            if len(feats) < 2:
                st.warning("Select at least 2 features.")
            else:
                with st.spinner("Clustering …"):
                    labels = utils.kmeans_lithology(df_cur, feats, n_cl)
                    df_upd = st.session_state.df.copy()
                    df_upd["CLUSTER"] = labels
                    st.session_state.df = df_upd
                    st.success(f"✅ Clustered into {n_cl} groups.  Column **CLUSTER** saved.")

        if "CLUSTER" in st.session_state.df.columns:
            df_cl = st.session_state.df
            ca, cb = st.columns([1, 2])

            with ca:
                st.plotly_chart(
                    plots.plot_cluster_strip(df_cl, "CLUSTER"),
                    use_container_width=True,
                    key="lithology_pc8"
                )

            with cb:
                invert_y_cluster = st.checkbox("Invert Y-axis (Cluster Plot)", key="lit_inv_y_cluster")

                fig_cluster = plots.plot_crossplot(
                    df_cl,
                    x_cx,
                    y_cx,
                    cluster_col="CLUSTER",
                    title=f"Clusters — {x_cx} vs {y_cx}"
                )

                if invert_y_cluster:
                    fig_cluster.update_yaxes(autorange="reversed")

                st.plotly_chart(
                    fig_cluster,
                    use_container_width=True,
                    key="lithology_pc9",
                )
