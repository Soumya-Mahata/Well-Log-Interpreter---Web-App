"""
lithology.py  —  Lithology Identification Module
==================================================
Six standard crossplots + custom plot + K-means clustering.
All widget keys are unique and prefixed with "lit_" to avoid collisions.
Computed M, N, RHOMAA, DTMAA, UMAA columns are stored in session_state.df.
VSH computation (with age correction) is included as a pre-processing step.
"""

import streamlit as st
import pandas as pd
import numpy as np
import utils
import plots


def render(df: pd.DataFrame):
    st.title("🪨 Lithology Identification")

    # Always work from the latest session_state version (picks up computed VSH etc.)
    df = st.session_state.df

    num_cols = [c for c in df.columns if c != "DEPTH"]

    # Auto-detect standard curves
    _nphi = utils.find_col(df, ["NPHI", "TNPH", "PHIN", "NPH"])
    _rhob = utils.find_col(df, ["RHOB", "RHOZ", "DEN"])
    _dt   = utils.find_col(df, ["DT",   "DTCO", "AC",   "DTC"])
    _gr   = utils.find_col(df, ["GR",   "GRD"])
    _pe   = utils.find_col(df, ["PE",   "PEF",  "PEFZ"])
    _rt   = utils.find_col(df, ["RT",   "LLD",  "ILD",  "RDEEP"])

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

    # ── Volume of Shale (Vsh) Calculation ────────────────────────────────────
    with st.expander("🪨 Volume of Shale (Vsh) Calculation", expanded=False):
        st.markdown("""
Compute **Vsh** (Volume of Shale) from the Gamma Ray log using multiple methods.

| Method | Formula | Best For |
|---|---|---|
| Linear (IGR) | IGR = (GR − GRmin) / (GRmax − GRmin) | All ages |
| Larionov Young | 0.083 × (2^(3.7 × IGR) − 1) | Tertiary rocks |
| Larionov Old | 0.33 × (2^(2 × IGR) − 1) | Older/consolidated rocks |
| Clavier | 1.7 − (3.38 − (IGR + 0.7)²)^0.5 | General use |
| Steiber | 0.5 × IGR / (1.5 − IGR) | Tertiary rocks |
        """)

        gr_vsh = st.selectbox(
            "GR curve for Vsh",
            ["None"] + num_cols,
            index=_idx(_gr, ["None"] + num_cols),
            key="lit_vsh_gr",
        )

        if gr_vsh != "None" and gr_vsh in df.columns:
            gr_data = df[gr_vsh].dropna()
            v1, v2 = st.columns(2)
            gr_clean_p = v1.number_input(
                "GR clean percentile", value=5, min_value=1, max_value=20, step=1,
                key="lit_vsh_gr_clean_pct",
                help="Percentile used for clean sand GR (default p05)",
            )
            gr_shale_p = v2.number_input(
                "GR shale percentile", value=95, min_value=80, max_value=99, step=1,
                key="lit_vsh_gr_shale_pct",
                help="Percentile used for shale GR (default p95)",
            )
            gr_clean_val = float(gr_data.quantile(gr_clean_p / 100))
            gr_shale_val = float(gr_data.quantile(gr_shale_p / 100))

            c1, c2 = st.columns(2)
            c1.metric(f"GR clean (p{gr_clean_p})", f"{gr_clean_val:.1f} API")
            c2.metric(f"GR shale (p{gr_shale_p})", f"{gr_shale_val:.1f} API")

            # Age correction selection
            st.markdown("**Age Correction:**")
            age_correction = st.radio(
                "Correction method",
                ["None (Linear IGR)", "Larionov — Tertiary (young)", "Larionov — Older rocks",
                 "Clavier (1971)", "Steiber (1969) — Tertiary"],
                horizontal=True,
                key="lit_vsh_correction",
            )

            if st.button("Compute Vsh", type="primary", key="lit_vsh_compute"):
                gr_s = df[gr_vsh]
                denom = gr_shale_val - gr_clean_val
                denom = denom if abs(denom) > 1e-6 else 1.0
                igr = ((gr_s - gr_clean_val) / denom).clip(0.0, 1.0)

                if "Tertiary (young)" in age_correction:
                    vsh = (0.083 * (2 ** (3.7 * igr) - 1)).clip(0, 1)
                    method_label = "Larionov Young"
                elif "Older rocks" in age_correction:
                    vsh = (0.33 * (2 ** (2 * igr) - 1)).clip(0, 1)
                    method_label = "Larionov Old"
                elif "Clavier" in age_correction:
                    vsh = (1.7 - (3.38 - (igr + 0.7) ** 2) ** 0.5).clip(0, 1)
                    method_label = "Clavier"
                elif "Steiber" in age_correction:
                    vsh = (0.5 * igr / (1.5 - igr + 1e-9)).clip(0, 1)
                    method_label = "Steiber"
                else:
                    vsh = igr
                    method_label = "Linear IGR"

                df_upd = st.session_state.df.copy()
                df_upd["VSH"] = vsh.values
                st.session_state.df = df_upd
                st.session_state["vsh_done"] = True
                st.success(f"✅ VSH computed using **{method_label}** and saved to column **VSH**.")

            # Show GR + VSH dual-track if computed
            if "VSH" in st.session_state.df.columns:
                from plotly.subplots import make_subplots as _msp
                import plotly.graph_objects as _go
                import numpy as _np
                df_cur = st.session_state.df
                _dep   = df_cur["DEPTH"].values
                _gr_v  = df_cur[gr_vsh].values
                _vsh_v = df_cur["VSH"].values
                _gr_max = max(150.0, float(df_cur[gr_vsh].quantile(0.99)) * 1.05)
                _gr_cut_default = gr_clean_val + (gr_shale_val - gr_clean_val) * 0.50
                _gr_cut = st.slider(
                    "GR cutoff (API)",
                    min_value=float(gr_clean_val),
                    max_value=float(gr_shale_val),
                    value=float(_gr_cut_default),
                    step=1.0,
                    key="lit_vsh_gr_cutoff",
                    help="Threshold separating Sand (below) from Shale (above) on the GR track",
                )

                fig_vsh = _msp(rows=1, cols=2, shared_yaxes=True,
                                subplot_titles=["GR (API)", "VSHALE"],
                                horizontal_spacing=0.04)

                # ══════════════════════════════════════════════════════════════
                # Track 1 — GR
                # Exact same logic as Triple Combo GR track:
                #   fill_betweenx(depth, 0, gr, where=(gr < cutoff))  → yellow
                #   fill_betweenx(depth, 0, gr, where=(gr >= cutoff)) → grey
                # Implemented as contiguous-run closed polygons (x=0 baseline)
                # ══════════════════════════════════════════════════════════════
                _gr_clipped = _np.clip(_gr_v, 0, _gr_max)

                def _vsh_fill_betweenx_zero(mask, fill_color, legend_name):
                    """Exact fill_betweenx(depth, 0, gr, where=mask) equivalent."""
                    idx = _np.where(mask & ~_np.isnan(_gr_clipped))[0]
                    if not len(idx):
                        return
                    runs, cur = [], [idx[0]]
                    for i in idx[1:]:
                        if i == cur[-1] + 1:
                            cur.append(i)
                        else:
                            runs.append(cur); cur = [i]
                    runs.append(cur)
                    for ri, r in enumerate(runs):
                        d_r  = _dep[r]
                        gr_r = _gr_clipped[r]
                        # Closed polygon: zeros forward + gr reversed
                        x_p = _np.concatenate([_np.zeros(len(r)), gr_r[::-1]])
                        y_p = _np.concatenate([d_r,               d_r[::-1]])
                        fig_vsh.add_trace(_go.Scatter(
                            x=x_p, y=y_p,
                            fill="toself", fillcolor=fill_color,
                            line=dict(width=0), mode="none",
                            name=legend_name,
                            showlegend=(ri == 0),
                            legendgroup=legend_name,
                            hoverinfo="skip",
                        ), row=1, col=1)

                _vsh_fill_betweenx_zero(
                    _gr_v < _gr_cut,
                    "rgba(255,255,0,0.50)", "Sand Zone"
                )
                _vsh_fill_betweenx_zero(
                    _gr_v >= _gr_cut,
                    "rgba(128,128,128,0.50)", "Shale Zone"
                )

                # GR line on top (limegreen, identical to triple combo)
                fig_vsh.add_trace(_go.Scatter(
                    x=_gr_clipped, y=_dep,
                    mode="lines", name="GR log",
                    line=dict(color="limegreen", width=1.5),
                    hovertemplate="GR: %{x:.1f} API<br>Depth: %{y:.1f}<extra></extra>",
                ), row=1, col=1)

                # GRclean / GRshale reference lines
                fig_vsh.add_vline(x=gr_clean_val, row=1, col=1,
                    line=dict(color="#1565C0", dash="dash", width=1.2),
                    annotation_text=f"|GRclean={gr_clean_val:.0f}",
                    annotation_font=dict(color="#1565C0", size=9),
                    annotation_position="top right")
                fig_vsh.add_vline(x=gr_shale_val, row=1, col=1,
                    line=dict(color="#C62828", dash="dash", width=1.2),
                    annotation_text=f"|GRshale={gr_shale_val:.0f}",
                    annotation_font=dict(color="#C62828", size=9),
                    annotation_position="top right")
                fig_vsh.update_xaxes(
                    title_text="GR (API)", range=[0, _gr_max],
                    showgrid=True, gridcolor="#d4d4d4",
                    title_font=dict(color="green", size=11),
                    tickfont=dict(color="#212121", size=10),
                    row=1, col=1,
                )

                # ══════════════════════════════════════════════════════════════
                # Track 2 — Vsh with gradient colormap fill (summer_r)
                # Notebook: per-segment polygon fill, colour = cmap(avg_vsh)
                # We implement as contiguous thin rectangles coloured by avg_vsh.
                # summer_r: 0 = yellow-green (#ffff00) → 1 = teal (#008066)
                # ══════════════════════════════════════════════════════════════
                def _summer_r_rgba(v, alpha=0.85):
                    """Interpolate summer_r: v=0→yellow, v=1→teal."""
                    v = float(_np.clip(v, 0.0, 1.0))
                    # summer_r  0→ (0.5, 1, 0.4) teal  reversed: 0→yellow, 1→teal
                    # Simple two-stop linear: yellow (255,255,0) → teal (0,128,102)
                    r = int(255 + v * (0   - 255))
                    g = int(255 + v * (128 - 255))
                    b = int(0   + v * (102 -   0))
                    return f"rgba({r},{g},{b},{alpha})"

                _n = len(_dep)
                for i in range(_n - 1):
                    avg_v = float((_vsh_v[i] + _vsh_v[i + 1]) / 2.0)
                    if _np.isnan(avg_v):
                        continue
                    col   = _summer_r_rgba(avg_v)
                    x_seg = [0, 0, float(_vsh_v[i + 1]), float(_vsh_v[i]), 0]
                    y_seg = [float(_dep[i]), float(_dep[i + 1]),
                             float(_dep[i + 1]), float(_dep[i]), float(_dep[i])]
                    fig_vsh.add_trace(_go.Scatter(
                        x=x_seg, y=y_seg,
                        fill="toself", fillcolor=col,
                        line=dict(width=0), mode="none",
                        showlegend=False, hoverinfo="skip",
                    ), row=1, col=2)

                # VSH curve on top (black, lw=2, notebook style)
                fig_vsh.add_trace(_go.Scatter(
                    x=_vsh_v, y=_dep, mode="lines", name="VSHALE",
                    line=dict(color="black", width=2.0),
                    hovertemplate="Vsh: %{x:.3f}<br>Depth: %{y:.1f}<extra></extra>",
                ), row=1, col=2)
                # Colorbar dummy trace
                fig_vsh.add_trace(_go.Scatter(
                    x=[None], y=[None], mode="markers",
                    marker=dict(
                        colorscale=[[0,"#ffff00"],[0.5,"#80c060"],[1,"#008066"]],
                        cmin=0, cmax=1, color=[0], showscale=True,
                        colorbar=dict(
                            title=dict(text="Shale Volume", side="right"),
                            x=1.02, thickness=14, len=0.9,
                            tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                            ticktext=["0.0","0.25","0.5","0.75","1.0"],
                        ),
                    ),
                    showlegend=False,
                ), row=1, col=2)
                fig_vsh.update_xaxes(
                    title_text="VSHALE", range=[0, 1],
                    showgrid=True, gridcolor="#d4d4d4",
                    title_font=dict(color="#212121", size=11),
                    tickfont=dict(color="#212121", size=10),
                    row=1, col=2,
                )

                # Shared depth axis
                fig_vsh.update_yaxes(
                    autorange="reversed", title_text="Depth (m)",
                    showgrid=True, gridcolor="#d4d4d4",
                    title_font=dict(color="#212121", size=12),
                    tickfont=dict(color="#212121", size=10),
                    row=1, col=1,
                )
                for ann in fig_vsh.layout.annotations:
                    ann.font = dict(color="#212121", size=11)
                fig_vsh.update_layout(
                    title=dict(
                        text="<b>Volume of Shale from Gamma Ray Log</b>",
                        font=dict(size=15, color="#0D2B5E"),
                        x=0.5, xanchor="center",
                    ),
                    height=720, margin=dict(l=65, r=100, t=70, b=80),
                    plot_bgcolor="white", paper_bgcolor="white",
                    legend=dict(
                        orientation="h", yanchor="top", y=-0.08,
                        xanchor="center", x=0.25,
                        bgcolor="rgba(255,255,255,0.95)",
                        bordercolor="#aaaaaa", borderwidth=1,
                        font=dict(color="#212121", size=10),
                    ),
                    showlegend=True,
                )
                st.plotly_chart(fig_vsh, use_container_width=True, key="lit_vsh_track")
        else:
            st.info("Select a GR curve above to compute Vsh.")

    st.divider()

    # ── Global Colormap Selector ──────────────────────────────────────────────
    _CMAPS = [
        "Viridis_r", "Viridis", "Plasma", "Inferno", "Magma", "Cividis",
        "Turbo", "Jet", "Rainbow", "RdYlGn", "RdBu", "Spectral",
        "Blues", "Greens", "Oranges", "Reds", "YlOrRd", "YlGnBu",
    ]
    st.markdown("**🎨 Colormap for scatter plots:**")
    _cmap_col1, _cmap_col2 = st.columns([2, 4])
    selected_cmap = _cmap_col1.selectbox(
        "Color scale",
        _CMAPS,
        index=0,
        key="lit_global_cmap",
        help="Applied to all crossplots when a 'Colour by' column is selected.",
    )
    _cmap_col2.caption(
        "Colormaps: **Viridis_r** (default, perceptually uniform) · "
        "**Plasma/Inferno** (high contrast) · **Jet/Rainbow** (classic) · "
        "**RdYlGn/RdBu** (diverging)"
    )
    st.divider()

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab_tc = st.tabs([
        "NPHI–RHOB",
        "NPHI–Sonic",
        "Density–Sonic",
        "M-N Plot",
        "MID (Δtmaa / ρmaa)",
        "MID (Umaa / ρmaa)",
        "Custom + K-Means",
        "Triple Combo",
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
                    colorscale=selected_cmap,
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
                    colorscale=selected_cmap,
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
                    colorscale=selected_cmap,
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
                              color_col=(None if col_mn == "None" else col_mn),
                              colorscale=selected_cmap),
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
                                      color_col=(None if col_m5 == "None" else col_m5),
                                      colorscale=selected_cmap),
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
                                     color_col=(None if col_m6 == "None" else col_m6),
                                     colorscale=selected_cmap),
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
            colorscale=selected_cmap,
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
                    title=f"Clusters — {x_cx} vs {y_cx}",
                    colorscale=selected_cmap,
                )

                if invert_y_cluster:
                    fig_cluster.update_yaxes(autorange="reversed")

                st.plotly_chart(
                    fig_cluster,
                    use_container_width=True,
                    key="lithology_pc9",
                )

    # ── Tab 8 : Triple Combo Plot ─────────────────────────────────────────────
    with tab_tc:
        st.subheader("Triple Combo Plot")
        st.caption(
            "Standard 3-track composite log: GR + Caliper | Resistivity (log scale) "
            "| RHOB + NPHI.  "
            "Shaded fills indicate sand/shale zones (Track 1) and HC/shale crossover (Track 3)."
        )

        # Auto-detect additional curves
        _cali = utils.find_col(df, ["CALI", "HCAL", "CAL"])
        _lls  = utils.find_col(df, ["LLS", "AT10", "AHT10", "RLA2"])
        _llm  = utils.find_col(df, ["LLM", "AT30", "AHT30", "RLA3"])

        # ── Curve selectors ───────────────────────────────────────────────────
        tc1, tc2, tc3, tc4, tc5 = st.columns(5)
        tc_gr   = tc1.selectbox("GR",   ["None"] + num_cols, key="lit_tc_gr",
                                 index=_idx(_gr,   ["None"] + num_cols))
        tc_rt   = tc2.selectbox("LLD / RT (deep)",  ["None"] + num_cols, key="lit_tc_rt",
                                 index=_idx(_rt,   ["None"] + num_cols))
        tc_rhob = tc3.selectbox("RHOB", ["None"] + num_cols, key="lit_tc_rhob",
                                 index=_idx(_rhob, ["None"] + num_cols))
        tc_nphi = tc4.selectbox("NPHI", ["None"] + num_cols, key="lit_tc_nphi",
                                 index=_idx(_nphi, ["None"] + num_cols))
        tc_gr_cutoff = tc5.number_input(
            "GR cutoff (API)", value=75.0, min_value=0.0, max_value=300.0,
            step=5.0, key="lit_tc_gr_cutoff",
            help="GR values below cutoff = sand (yellow), above = shale (grey)",
        )

        # ── Additional curves ─────────────────────────────────────────────────
        with st.expander("⚙️ Additional curves (CALI, LLM, LLS)", expanded=False):
            oc1, oc2, oc3, oc4 = st.columns(4)
            tc_cali = oc1.selectbox("CALI", ["None"] + num_cols, key="lit_tc_cali",
                                     index=_idx(_cali, ["None"] + num_cols))
            tc_lls  = oc2.selectbox("LLS (shallow)", ["None"] + num_cols, key="lit_tc_lls",
                                     index=_idx(_lls, ["None"] + num_cols))
            tc_llm  = oc3.selectbox("LLM (medium)",  ["None"] + num_cols, key="lit_tc_llm",
                                     index=_idx(_llm, ["None"] + num_cols))
            tc_bit  = oc4.number_input("Bit size (in)", value=8.5, min_value=4.0,
                                        max_value=26.0, step=0.5, key="lit_tc_bit")

        if any(
            c != "None" and c in df.columns
            for c in [tc_gr, tc_rt, tc_rhob, tc_nphi]
        ):
            st.plotly_chart(
                plots.plot_triple_combo_lit(
                    df=st.session_state.df,
                    gr_col=None if tc_gr == "None" else tc_gr,
                    rt_col=None if tc_rt == "None" else tc_rt,
                    rhob_col=None if tc_rhob == "None" else tc_rhob,
                    nphi_col=None if tc_nphi == "None" else tc_nphi,
                    gr_cutoff=tc_gr_cutoff,
                    cali_col=None if tc_cali == "None" else tc_cali,
                    bit_size=tc_bit,
                    lls_col=None if tc_lls == "None" else tc_lls,
                    llm_col=None if tc_llm == "None" else tc_llm,
                ),
                use_container_width=True,
                key="lithology_tc_plot",
            )
        else:
            st.info("Select at least one curve above to display the Triple Combo plot.")

