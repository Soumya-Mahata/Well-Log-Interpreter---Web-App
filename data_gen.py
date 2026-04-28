"""
data_gen.py  —  Missing P- & S-Sonic Data Generation  (entry point)
====================================================================
This file is the ONLY module imported by main.py:

    import data_gen
    data_gen.render(df)

It contains NO business logic — it simply:
  1. Shows a title + info banner
  2. Renders a radio selector (single source of truth via session_state)
  3. Delegates to the appropriate sub-module

Sub-module layout
─────────────────
  dg_utils.py          Shared helpers, Plotly builders, show_results()
  dg_conventional.py   Physics-based empirical methods  (Gardner, Castagna …)
  dg_unconventional.py ML models (RF, XGBoost …) + DL models (CNN-Bi-LSTM …)
  dg_comparison.py     Metrics table, bar charts, depth tracks, UQ summary
"""

from __future__ import annotations

import streamlit as st
import pandas as pd

from dg_conventional import render as _render_conv
from dg_unconventional import render_dl
from dg_comparison import render as _render_cmp


# ── Approach registry ─────────────────────────────────────────────────────────

_APPROACHES = [
    "🔵  Conventional Methods",
    "🔴  Unconventional Methods",
    "📊  Comparison",
]

_DISPATCH = {
    "🔵  Conventional Methods": _render_conv,
    "🔴  Unconventional Methods":        render_dl,
    "📊  Comparison":           _render_cmp,
}


# ════════════════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def render(df: pd.DataFrame) -> None:
    st.title("🤖 Missing P- & S-Sonic Data Generation")

    st.info(
        "**Objective:** Predict missing **Vp (P-wave sonic logs)** using:\n\n"
        "• **Conventional methods** (empirical equations)\n"
        "• **Unconventional Methods (CNN-BiLSTM)** using a pre-trained model\n\n"
        "Then compare all predictions using statistical metrics.\n\n"
        "**Outputs:**\n"
        "- `Vp_pred`\n"
        "- `Vp_uncertainty`\n"
    )

    # Single source of truth — radio key IS the session_state value.
    # Changing the radio immediately reruns the page and calls the correct
    # sub-module; there are no tabs to get out of sync.
    approach = st.radio(
        "Select Approach",
        _APPROACHES,
        horizontal=True,
        key="dg_approach",
    )

    st.divider()

    fn = _DISPATCH.get(approach, _render_conv)
    fn(st.session_state.df)
