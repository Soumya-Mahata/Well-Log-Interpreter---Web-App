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
        has_nulls = int(ndf["Null Count"].sum()) > 0
        if has_nulls:
            st.dataframe(ndf.style.background_gradient(subset=["Null %"], cmap="YlOrRd"),
                         use_container_width=True, hide_index=True)
        else:
            st.success("✅ No null values found in the current depth range.")

        st.divider()

        _FILL_METHODS = ["interpolate", "ffill", "bfill", "mean", "median", "drop"]
        _HELP = (
            "**interpolate** — linear between valid neighbours (recommended)\n"
            "**ffill/bfill** — forward/backward fill\n"
            "**mean** — column mean\n"
            "**median** — column median\n"
            "**drop** — remove rows with any null"
        )

        fill_mode = st.radio(
            "Filling Mode",
            ["🌐 Global — same method for all curves",
             "🎯 Column-wise — choose per curve"],
            horizontal=True,
            key="qc_null_fill_mode",
        )

        # Identify curves that actually have null values
        null_curves = [c for c in df.columns if c != "DEPTH" and df[c].isna().any()]

        if fill_mode.startswith("🌐"):
            if null_curves:
                st.info(f"**{len(null_curves)}** curve(s) with null values: "
                        f"{', '.join(f'`{c}`' for c in null_curves)}")
            method = st.selectbox(
                "Fill method (applied to all curves with nulls)", _FILL_METHODS,
                index=0, key="qc_null_method", help=_HELP,
            )
            col_methods: dict = {}

        else:
            if not null_curves:
                st.success("✅ No curves with null values — nothing to fill column-wise.")
                col_methods = {}
                method = "interpolate"
            else:
                st.markdown(f"**Select a fill method for each of the "
                            f"{len(null_curves)} curve(s) with null values:**")
                col_methods = {}
                cols_per_row = 3
                for chunk_start in range(0, len(null_curves), cols_per_row):
                    chunk = null_curves[chunk_start: chunk_start + cols_per_row]
                    row_cols = st.columns(len(chunk))
                    for widget_col, curve in zip(row_cols, chunk):
                        null_count = int(df[curve].isna().sum())
                        label = f"`{curve}`  ⚠️ {null_count} nulls"
                        col_methods[curve] = widget_col.selectbox(
                            label, _FILL_METHODS, index=0,
                            key=f"qc_null_col_{curve}",
                        )
                method = "interpolate"

        st.divider()

        if st.button("Apply Null Filling", type="primary", key="qc_null_btn"):
            working = st.session_state.df.copy()
            if fill_mode.startswith("🌐"):
                # Only fill columns that actually have nulls
                if null_curves:
                    for curve in null_curves:
                        tmp = utils.fill_nulls(working[[curve]], method=method)
                        working[curve] = tmp[curve].values
                filled = working
            else:
                filled = working.copy()
                for curve, m in col_methods.items():
                    if curve not in filled.columns:
                        continue
                    tmp = utils.fill_nulls(filled[[curve]], method=m)
                    filled[curve] = tmp[curve].values

            st.session_state.df    = filled
            st.session_state.df_qc = filled.copy()
            remaining = int(filled.isna().sum().sum())
            st.success(f"✅ Null filling applied.  Remaining NaNs: {remaining}")

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
**Hole condition assessment** based on caliper vs bit size ratio:

| Condition | Criterion | Effect on Logs |
|---|---|---|
| 🔴 **Bad Hole / Washout** | CALI > 110 % bit size | Density / neutron unreliable |
| 🟡 **Moderate** | 100 % < CALI ≤ 110 % bit size | Use with caution |
| 🟢 **Good (In-Gauge)** | CALI ≤ 100 % bit size | High confidence |
        """)

        df_w     = st.session_state.df
        num_cols = [c for c in df_w.columns if c != "DEPTH"]
        cali_candidates = [c for c in num_cols if "CALI" in c.upper()]

        hc1, hc2 = st.columns(2)
        cali_col = hc1.selectbox(
            "Caliper curve",
            cali_candidates if cali_candidates else num_cols,
            key="qc_cali_col",
        )
        bit_size = hc2.number_input(
            "Bit size (same units as CALI)",
            value=8.5, min_value=1.0, max_value=36.0,
            step=0.5, key="qc_bit_size",
        )

        if cali_col and cali_col in df_w.columns:
            hq     = utils.hole_quality_check(df_w, cali_col, bit_size)
            counts = hq.value_counts()
            n_tot  = len(hq)

            bad_pct  = counts.get("Bad Hole",      0) / n_tot * 100
            mod_pct  = counts.get("Moderate Hole", 0) / n_tot * 100
            good_pct = counts.get("Good Hole",     0) / n_tot * 100

            kc1, kc2, kc3 = st.columns(3)
            kc1.metric("🔴 Bad / Washout",   f"{bad_pct:.1f}%")
            kc2.metric("🟡 Moderate",          f"{mod_pct:.1f}%")
            kc3.metric("🟢 Good (In-Gauge)",   f"{good_pct:.1f}%")

            st.plotly_chart(
                plots.plot_hole_quality(df_w, cali_col, bit_size, hq),
                use_container_width=True,
                key=f"hole_{cali_col}_{bit_size}",
            )

            with st.expander("Washout / Bad-Hole interval list"):
                bad_mask = hq == "Bad Hole"
                wo_df = df_w[bad_mask][["DEPTH", cali_col]].copy()
                if len(wo_df):
                    st.dataframe(wo_df.head(300), use_container_width=True, hide_index=True)
                else:
                    st.success("No washout / bad-hole intervals detected.")
        else:
            st.info("No CALI curve detected.  Select a caliper curve above.")
