"""
fluids.py  —  Fluid Analysis Module
=====================================
All widget keys prefixed "fl_".
Archie parameters and Rw are stored in session_state and survive navigation.
Three Rw estimation methods: direct input, Pickett plot, SP log method.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import utils
import plots


def _ss(key, default=None):
    return st.session_state.get(key, default)


def render(df: pd.DataFrame):
    st.title("💧 Fluid Analysis — Water Saturation")

    num_cols = [c for c in df.columns if c != "DEPTH"]
    _rt   = utils.find_col(df, ["RT",   "AT90", "ILD",  "RESD", "RD"])
    _phi  = utils.find_col(df, ["PHIE", "PHIT"])
    _gr   = utils.find_col(df, ["GR",   "GRD"])
    _sp   = utils.find_col(df, ["SP"])

    st.markdown("""
<div class="info-box">
<b>Archie's Law (1942)</b> — water saturation from porosity and resistivity:<br>
<code>Sw = [(a × Rw) / (Rt × ϕ<sup>m</sup>)]<sup>1/n</sup></code><br>
Zones with <b>low Sw</b> (below the cutoff) are hydrocarbon-bearing.
</div>
""", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "1. Curve & Archie Setup",
        "2. Pickett Plot (Rw)",
        "3. SP Method (Rw)",
        "4. Compute Sw",
    ])

    # ── Tab 1 : Curve & Archie Setup ─────────────────────────────────────────
    with tab1:
        st.subheader("Assign Curves")

        def _sidx(col, opts):
            return opts.index(col) if col and col in opts else 0

        opts_n = ["None"] + num_cols
        rt_sel  = st.selectbox("True Resistivity (RT) ohm·m", opts_n, key="fl_sel_rt",
                                index=_sidx(_rt,  opts_n))
        phi_sel = st.selectbox("Porosity (PHIE or PHIT)",     opts_n, key="fl_sel_phi",
                                index=_sidx(_phi, opts_n))

        st.session_state["fl_rt_col"]  = None if rt_sel  == "None" else rt_sel
        st.session_state["fl_phi_col"] = None if phi_sel == "None" else phi_sel

        st.divider()
        st.subheader("Archie Parameters")
        st.markdown("""
| Symbol | Typical range | Meaning |
|--------|--------------|---------|
| **a**  | 0.62–1.0     | Tortuosity factor |
| **m**  | 1.8–2.2      | Cementation exponent |
| **n**  | 2.0          | Saturation exponent |
| **Rw** | site-specific| Formation water resistivity (ohm·m) |
        """)

        ac1, ac2, ac3, ac4 = st.columns(4)
        a_v  = ac1.number_input("a",  value=float(_ss("fl_a",  1.0)),
                                 min_value=0.1, max_value=5.0, step=0.05,
                                 format="%.2f", key="fl_ni_a")
        m_v  = ac2.number_input("m",  value=float(_ss("fl_m",  2.0)),
                                 min_value=1.0, max_value=4.0, step=0.05,
                                 format="%.2f", key="fl_ni_m")
        n_v  = ac3.number_input("n",  value=float(_ss("fl_n",  2.0)),
                                 min_value=1.0, max_value=4.0, step=0.05,
                                 format="%.2f", key="fl_ni_n")
        rw_v = ac4.number_input("Rw", value=float(_ss("fl_rw", 0.10)),
                                 min_value=0.001, max_value=10.0,
                                 step=0.005, format="%.4f", key="fl_ni_rw")

        st.session_state["fl_a"]  = a_v
        st.session_state["fl_m"]  = m_v
        st.session_state["fl_n"]  = n_v
        st.session_state["fl_rw"] = rw_v

    # ── Tab 2 : Pickett Plot ──────────────────────────────────────────────────
    with tab2:
        st.subheader("Pickett Plot — Rw Estimation")
        st.markdown(r"""
Log-log plot of ϕ (x) vs Rt (y).  In water-bearing zones (Sw = 1):

$$\log(R_t) = -m \cdot \log(\phi) + \log(a \cdot R_w)$$

Adjust **Rw** until the blue Sw = 1 line passes through the clean water-bearing
scatter points, then click **Use this Rw**.
        """)

        rt_col  = _ss("fl_rt_col")
        phi_col = _ss("fl_phi_col")
        a_v     = _ss("fl_a",  1.0)
        m_v     = _ss("fl_m",  2.0)
        n_v     = _ss("fl_n",  2.0)

        color_opts = ["None"] + num_cols
        col_pk = st.selectbox("Colour by", color_opts, key="fl_pk_color",
                               index=(color_opts.index(_gr) if _gr and _gr in color_opts else 0))

        if rt_col and phi_col and rt_col in df.columns and phi_col in df.columns:
            rw_auto = utils.estimate_rw_pickett(df[rt_col], df[phi_col], a_v, m_v)
            st.info(f"Auto-estimated Rw from Pickett regression: **{rw_auto:.4f} ohm·m**")

            rw_pk = st.number_input(
                "Rw for Pickett overlay",
                value=float(round(rw_auto, 4)),
                min_value=0.001, max_value=10.0,
                step=0.005, format="%.4f", key="fl_pk_rw_ni",
            )

            st.plotly_chart(
                plots.plot_pickett(df, rt_col, phi_col,
                                   rw=rw_pk, a=a_v, m=m_v, n=n_v,
                                   color_col=(None if col_pk == "None" else col_pk)),
                use_container_width=True,
                key="fluids_pc1",
            )

            if st.button("✅ Use this Rw in computation", key="fl_pk_use"):
                st.session_state["fl_rw"] = rw_pk
                st.success(f"Rw set to **{rw_pk:.4f}** ohm·m.")
        else:
            st.info("Assign RT and porosity curves in **Tab 1** first.")

    # ── Tab 3 : SP Method ────────────────────────────────────────────────────
    with tab3:
        st.subheader("SP Log Method — Rw Estimation")
        st.markdown(r"""
Simplified Schlumberger formula:

$$R_{mf,eq} = R_{mf} \cdot \frac{T + 6.77}{75 + 6.77}
\qquad
R_{w,eq}   = \frac{R_{mf,eq}}{10^{SSP/61}}
\qquad
R_w        = R_{w,eq} \cdot \frac{75 + 6.77}{T + 6.77}$$

SSP = median SP amplitude in the permeable zone (mV).
K ≈ 61 mV at 25 °C; this is an approximation — use the Pickett plot for precision.
        """)

        sp_opts = ["None"] + num_cols
        sp_col  = st.selectbox("SP curve", sp_opts, key="fl_sp_col",
                                index=(sp_opts.index(_sp) if _sp and _sp in sp_opts else 0))

        sp1, sp2 = st.columns(2)
        rmf_v  = sp1.number_input("Rmf (ohm·m)", value=1.0,
                                   min_value=0.001, max_value=50.0,
                                   step=0.1, format="%.3f", key="fl_sp_rmf")
        temp_v = sp2.number_input("Formation temperature (°F)", value=150.0,
                                   min_value=60.0, max_value=400.0,
                                   step=5.0, key="fl_sp_temp")

        if sp_col != "None" and sp_col in df.columns:
            fig_sp = px.line(
                x=df[sp_col], y=df["DEPTH"],
                title=f"SP Log ({sp_col})",
                labels={"x": "SP (mV)", "y": "Depth"},
                color_discrete_sequence=["#1565C0"],
            )
            fig_sp.update_yaxes(autorange="reversed")
            fig_sp.update_layout(height=380, plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig_sp, use_container_width=True,
                key="fluids_pc2")

            ssp_default = float(df[sp_col].dropna().median())
            ssp_v = st.number_input(
                "SSP amplitude (mV) — override if needed",
                value=round(ssp_default, 1), key="fl_sp_ssp",
            )
            rw_sp = utils.estimate_rw_sp(df[sp_col], rmf=rmf_v, temp_f=temp_v)
            # Recalculate with user override
            rmf_eq  = rmf_v * (temp_v + 6.77) / (75.0 + 6.77)
            rw_eq   = rmf_eq / (10.0 ** (ssp_v / 61.0))
            rw_sp   = float(np.clip(rw_eq * (75.0 + 6.77) / (temp_v + 6.77), 0.001, 10.0))

            st.success(f"Estimated Rw from SP: **{rw_sp:.4f} ohm·m**")
            if st.button("✅ Use this Rw in computation", key="fl_sp_use"):
                st.session_state["fl_rw"] = rw_sp
                st.success(f"Rw set to **{rw_sp:.4f}** ohm·m.")
        else:
            st.info("No SP curve assigned.  Select one above or use the Pickett method.")

    # ── Tab 4 : Compute Sw ────────────────────────────────────────────────────
    with tab4:
        st.subheader("Compute Water Saturation (Archie's Equation)")

        rt_col  = _ss("fl_rt_col")
        phi_col = _ss("fl_phi_col")
        a_v     = float(_ss("fl_a",   1.0))
        m_v     = float(_ss("fl_m",   2.0))
        n_v     = float(_ss("fl_n",   2.0))
        rw_v    = float(_ss("fl_rw",  0.10))

        # Show current parameter summary
        st.markdown(f"""
| Parameter | Value |
|-----------|-------|
| RT curve       | `{rt_col  or 'not assigned'}` |
| Porosity curve | `{phi_col or 'not assigned'}` |
| a | {a_v:.2f} |  m | {m_v:.2f} |  n | {n_v:.2f} |
| **Rw** | **{rw_v:.4f} ohm·m** |
        """)

        sw_cut = st.slider(
            "Hydrocarbon zone Sw cutoff  (Sw below this = HC-bearing)",
            0.10, 0.90,
            float(_ss("fl_sw_cut", 0.60)),
            0.05, key="fl_sw_cut_sl",
        )
        st.session_state["fl_sw_cut"] = sw_cut

        if st.button("🔄 Compute Sw", type="primary", key="fl_compute_btn"):
            if not rt_col or not phi_col:
                st.error("Assign RT and porosity curves in Tab 1.")
            elif rt_col not in df.columns or phi_col not in df.columns:
                st.error("One or more assigned curves not found in the current dataset.")
            else:
                with st.spinner("Applying Archie's equation …"):
                    df_upd = st.session_state.df.copy()
                    df_upd["SW"] = utils.water_saturation_archie(
                        df_upd[rt_col], df_upd[phi_col], rw_v, a_v, m_v, n_v)
                    df_upd["SH"] = utils.hydrocarbon_saturation(df_upd["SW"])
                    st.session_state.df    = df_upd
                    st.session_state.sw_done = True

                    hc_pct = float((df_upd["SW"] < sw_cut).mean() * 100)
                    st.success(
                        f"✅ Sw computed.  **{hc_pct:.1f}%** of samples below "
                        f"Sw cutoff ({sw_cut})."
                    )

        # ── Results display (always shown if sw_done) ─────────────────────────
        df_cur = st.session_state.df
        if _ss("sw_done") and "SW" in df_cur.columns:
            phie_col = utils.find_col(df_cur, ["PHIE", "PHIT"])
            rt_plot  = rt_col if (rt_col and rt_col in df_cur.columns) else None

            st.subheader("Sw vs Depth")
            st.plotly_chart(
                plots.plot_sw(df_cur, "SW", phie_col, rt_plot, sw_cut),
                use_container_width=True,
                key="fluids_pc3",
            )

            col_a, col_b = st.columns(2)
            with col_a:
                fig_hist = px.histogram(
                    df_cur["SW"].dropna(), x="SW", nbins=60,
                    title="Water Saturation Distribution",
                    color_discrete_sequence=["#1565C0"],
                )
                fig_hist.add_vline(
                    x=sw_cut,
                    line=dict(color="#E53935", dash="dash", width=1.8),
                    annotation_text=f"Sw cut = {sw_cut}",
                    annotation_font=dict(color="#E53935"),
                )
                fig_hist.update_layout(
                    height=360, plot_bgcolor="white", paper_bgcolor="white")
                st.plotly_chart(fig_hist, use_container_width=True,
                key="fluids_pc4")

            with col_b:
                st.subheader("Statistics")
                sw_stats = df_cur[["SW", "SH"]].describe().T
                st.dataframe(sw_stats.style.format("{:.4f}"), use_container_width=True)

                pay_n = int((df_cur["SW"] < sw_cut).sum())
                c1, c2 = st.columns(2)
                c1.metric("HC-bearing samples", f"{pay_n:,}")
                c2.metric("HC %", f"{pay_n / len(df_cur) * 100:.1f}%")
        else:
            st.info("Press **Compute Sw** above after assigning curves and Rw.")
