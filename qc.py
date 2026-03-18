"""
qc.py  -  Quality Control Module
All st.plotly_chart() calls carry a unique key= to prevent StreamlitDuplicateElementId.
"""

import streamlit as st
import pandas as pd
import utils
import plots


def render(df: pd.DataFrame, raw_df: pd.DataFrame):
    tab_null, tab_out, tab_smooth, tab_hole = st.tabs([
        "📉 Null Values", "⚡ Outlier Detection", "〰️ Smoothing", "🕳️ Hole Quality",
    ])

    # ── Null Values ───────────────────────────────────────────────────────────
    with tab_null:
        st.markdown(
            "Null values arise from tool failure, washout zones, or LAS null-fill values.  \n"
            "Choose a strategy then press **Apply**."
        )
        ndf = utils.count_nulls(df)
        if int(ndf["Null Count"].sum()) > 0:
            st.dataframe(ndf.style.background_gradient(subset=["Null %"], cmap="YlOrRd"),
                         use_container_width=True, hide_index=True)
        else:
            st.success("✅ No null values found in the current depth range.")

        method = st.selectbox(
            "Fill method", ["interpolate", "ffill", "bfill", "mean", "drop"],
            index=0, key="qc_null_method",
            help=("**interpolate** — linear between valid neighbours (recommended)\n"
                  "**ffill/bfill** — forward/backward fill\n"
                  "**mean** — column mean\n"
                  "**drop** — remove rows with any null"))
        if st.button("Apply Null Filling", key="qc_null_btn"):
            filled = utils.fill_nulls(st.session_state.df, method=method)
            st.session_state.df    = filled
            st.session_state.df_qc = filled.copy()
            st.success(f"✅ Null filling applied.  Remaining NaNs: {int(filled.isna().sum().sum())}")

    # ── Outlier Detection ─────────────────────────────────────────────────────
    with tab_out:
        st.markdown("**Z-score** — global mean-based.\n\n"
                    "**Median MAD** — rolling median — more robust for spiky log data.")
        df_w     = st.session_state.df
        num_cols = [c for c in df_w.columns if c != "DEPTH"]

        oc1, oc2, oc3 = st.columns([2, 2, 1])
        crv  = oc1.selectbox("Curve", num_cols, key="qc_out_crv")
        meth = oc2.radio("Method", ["Z-score", "Median MAD"], horizontal=True, key="qc_out_meth")
        logx = oc3.checkbox("Log X", key="qc_out_log")
        thr  = st.slider("Detection threshold", 1.5, 6.0, 3.0, 0.25, key="qc_out_thr")
        win_mad = 7
        if meth == "Median MAD":
            win_mad = st.slider("MAD window (samples)", 3, 31, 7, 2, key="qc_out_win")

        if crv in df_w.columns:
            mask  = (utils.detect_outliers_zscore(df_w[crv], thr) if meth == "Z-score"
                     else utils.detect_outliers_median(df_w[crv], win_mad, thr))
            n_out = int(mask.sum())
            col_a, col_b = st.columns(2)
            col_a.metric("Outliers detected", f"{n_out:,}")
            col_b.metric("As % of samples",   f"{mask.mean()*100:.2f}%")

            # Key encodes every parameter that changes the figure content
            st.plotly_chart(plots.plot_outlier_flags(df_w, crv, mask, log_scale=logx),
                            use_container_width=True,
                            key=f"out_{crv}_{meth}_{thr}_{win_mad}_{logx}")

            if st.button("Replace outliers with interpolation", key="qc_out_replace"):
                cleaned = utils.replace_outliers(df_w, crv, mask)
                st.session_state.df    = cleaned
                st.session_state.df_qc = cleaned.copy()
                st.plotly_chart(plots.plot_before_after(df_w, cleaned, crv, log_scale=logx),
                                use_container_width=True,
                                key=f"ba_{crv}_{meth}_{thr}_{win_mad}_{logx}_after")
                st.success(f"✅ {n_out} outliers replaced in **{crv}**.")

    # ── Smoothing ─────────────────────────────────────────────────────────────
    with tab_smooth:
        st.markdown("**Moving Average** — uniform noise reduction.\n\n"
                    "**Savitzky-Golay** — preserves peak heights (recommended for GR, RT).")
        df_w     = st.session_state.df
        num_cols = [c for c in df_w.columns if c != "DEPTH"]

        sc1, sc2, sc3 = st.columns([2, 2, 1])
        sm_crv  = sc1.selectbox("Curve", num_cols, key="qc_sm_crv")
        sm_meth = sc2.radio("Method", ["moving_average", "savgol"], horizontal=True, key="qc_sm_meth")
        sm_log  = sc3.checkbox("Log X", key="qc_sm_log")
        sm_win  = st.slider("Window (samples)", 3, 51, 9, 2, key="qc_sm_win")

        if sm_crv in df_w.columns:
            smoothed   = utils.smooth_log(df_w[sm_crv], sm_meth, sm_win)
            df_preview = df_w.copy()
            df_preview[sm_crv] = smoothed
            st.plotly_chart(plots.plot_before_after(df_w, df_preview, sm_crv, log_scale=sm_log),
                            use_container_width=True,
                            key=f"sm_{sm_crv}_{sm_meth}_{sm_win}_{sm_log}")
            if st.button("Apply Smoothing", key="qc_sm_apply"):
                st.session_state.df[sm_crv]    = smoothed.values
                st.session_state.df_qc[sm_crv] = smoothed.values
                st.success(f"✅ Smoothing applied to **{sm_crv}**.")

    # ── Hole Quality ──────────────────────────────────────────────────────────
    with tab_hole:
        st.markdown("""
| Condition    | Criterion             | Effect on logs               |
|-------------|----------------------|------------------------------|
| **Washout**  | CALI > 110 % bit     | Density / neutron unreliable |
| **In-Gauge** | 95 % ≤ CALI ≤ 110 % | Good quality                 |
| **Mudcake**  | CALI < 95 % bit      | Minor effect                 |
        """)
        df_w     = st.session_state.df
        num_cols = [c for c in df_w.columns if c != "DEPTH"]
        cali_candidates = [c for c in num_cols if "CALI" in c.upper()]

        hc1, hc2 = st.columns(2)
        cali_col = hc1.selectbox("Caliper curve",
                                  cali_candidates if cali_candidates else num_cols,
                                  key="qc_cali_col")
        bit_size = hc2.number_input("Bit size (same units as CALI)",
                                     value=8.5, min_value=1.0, max_value=36.0,
                                     step=0.5, key="qc_bit_size")

        if cali_col and cali_col in df_w.columns:
            hq     = utils.hole_quality_check(df_w, cali_col, bit_size)
            counts = hq.value_counts()
            n_tot  = len(hq)
            kc1, kc2, kc3 = st.columns(3)
            kc1.metric("🔴 Washout",  f"{counts.get('Washout',  0)/n_tot*100:.1f}%")
            kc2.metric("🟢 In-Gauge", f"{counts.get('In-Gauge', 0)/n_tot*100:.1f}%")
            kc3.metric("🔵 Mudcake",  f"{counts.get('Mudcake',  0)/n_tot*100:.1f}%")
            st.plotly_chart(plots.plot_hole_quality(df_w, cali_col, bit_size, hq),
                            use_container_width=True,
                            key=f"hole_{cali_col}_{bit_size}")
            with st.expander("Washout interval list"):
                wo_df = df_w[hq == "Washout"][["DEPTH", cali_col]].copy()
                if len(wo_df):
                    st.dataframe(wo_df.head(300), use_container_width=True, hide_index=True)
                else:
                    st.success("No washout intervals detected.")
        else:
            st.info("No CALI curve detected.  Select a caliper curve above.")
