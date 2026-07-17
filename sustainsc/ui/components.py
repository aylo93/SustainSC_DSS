"""Small, typed Streamlit presentation components."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Mapping

import streamlit as st

from .theme import GLOBAL_CSS


def apply_design_system() -> None:
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


def render_page_header(
    title: str,
    description: str,
    *,
    eyebrow: str = "Scientific sustainability command center",
    metadata: str | None = None,
) -> None:
    meta = f"<div class='sc-eyebrow'>{escape(metadata)}</div>" if metadata else ""
    st.markdown(
        "<header class='sc-page-header'>"
        f"<div class='sc-eyebrow'>{escape(eyebrow)}</div>"
        f"<h1 class='sc-page-title'>{escape(title)}</h1>"
        f"<div class='sc-page-copy'>{escape(description)}</div>{meta}</header>",
        unsafe_allow_html=True,
    )


def render_section_header(title: str, description: str = "") -> None:
    st.markdown(
        "<div class='sc-section'><div class='sc-section-mark'></div><div>"
        f"<h2>{escape(title)}</h2><p>{escape(description)}</p></div></div>",
        unsafe_allow_html=True,
    )


def render_data_status_panel(values: Mapping[str, object]) -> None:
    cards = "".join(
        "<div class='sc-status-card'>"
        f"<div class='sc-status-label'>{escape(str(label))}</div>"
        f"<div class='sc-status-value'>{escape(str(value))}</div></div>"
        for label, value in values.items()
    )
    st.markdown(f"<div class='sc-status-grid'>{cards}</div>", unsafe_allow_html=True)


def render_workflow_progress(stages: Mapping[str, str]) -> None:
    items = "".join(
        f"<div class='sc-stage {escape(status)}'><strong>{escape(label)}</strong>"
        f"<small>{escape(status.replace('_', ' ').title())}</small></div>"
        for label, status in stages.items()
    )
    st.markdown(f"<div class='sc-workflow'>{items}</div>", unsafe_allow_html=True)


def render_empty_state(title: str, description: str) -> None:
    asset = Path(__file__).parent / "assets" / "supply_chain.svg"
    svg = asset.read_text(encoding="utf-8") if asset.exists() else ""
    st.markdown(
        "<section class='sc-empty' role='status'>"
        f"<div aria-hidden='true'>{svg}</div><h3>{escape(title)}</h3>"
        f"<p>{escape(description)}</p></section>",
        unsafe_allow_html=True,
    )


def render_filter_summary(filters: Mapping[str, str]) -> None:
    active = [f"{label}: {value}" for label, value in filters.items() if value != "All"]
    message = "Active table filters — " + " · ".join(active) if active else "No restrictive table filters."
    st.markdown(f"<div class='sc-filter-summary'>{escape(message)}</div>", unsafe_allow_html=True)
