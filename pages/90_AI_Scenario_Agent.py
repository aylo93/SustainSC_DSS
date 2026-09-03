"""SustainSCM multipage entry point for the bounded ASCA prototype."""

from pathlib import Path

import streamlit as st

from asca.streamlit_page import render_asca_page
from sustainsc.ui import apply_design_system, render_page_header


ROOT = Path(__file__).resolve().parents[1]

st.set_page_config(
    page_title="ASCA · SustainSCM DSS",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)
apply_design_system()

# The shared design system intentionally keeps a compact global content offset.
# This page begins with navigation, so add page-local clearance in normal flow.
st.markdown(
    """
    <style>
    .asca-toolbar-clearance { height: 2.5rem; }
    </style>
    <div class="asca-toolbar-clearance" aria-hidden="true"></div>
    """,
    unsafe_allow_html=True,
)

st.link_button(
    "Back to SustainSCM home",
    "/",
    type="primary",
    icon=":material/arrow_back:",
    width="content",
)
render_page_header(
    "AI Scenario Agent (ASCA)",
    "Configure and screen synthetic scenarios using the bounded metamodels from "
    "the Romanian experiments.",
    metadata="Romanian experiments · Domain-gated metamodel routing",
)
render_asca_page(assets_dir=ROOT / "asca_assets", show_title=False)
