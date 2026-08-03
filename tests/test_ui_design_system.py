from __future__ import annotations

from pathlib import Path

import plotly.io as pio

from sustainsc.ui.chart_theme import DIMENSION_COLOR_MAP
from sustainsc.ui.theme import (
    BACKGROUND,
    BORDER,
    PRIMARY,
    SURFACE,
    TEXT_PRIMARY,
)


def test_semantic_theme_tokens_and_dimension_colors_are_registered():
    assert all(value.startswith("#") and len(value) == 7 for value in (
        BACKGROUND,
        SURFACE,
        TEXT_PRIMARY,
        BORDER,
        PRIMARY,
    ))
    assert set(DIMENSION_COLOR_MAP) == {
        "environmental",
        "economic",
        "social",
        "technological",
    }
    assert "sustainscm" in pio.templates


def test_local_svg_is_small_accessible_and_not_hotlinked():
    asset = Path("sustainsc/ui/assets/supply_chain.svg")
    content = asset.read_text(encoding="utf-8")
    assert asset.stat().st_size < 10_000
    assert "aria-label=" in content
    assert "http://" not in content.replace("http://www.w3.org/2000/svg", "")
    assert "https://" not in content


def test_streamlit_theme_and_plotly_usage_are_centralized():
    config = Path(".streamlit/config.toml").read_text(encoding="utf-8")
    dashboard = Path("kpi_dashboard.py").read_text(encoding="utf-8")
    assert 'primaryColor = "#087F78"' in config
    assert "apply_design_system()" in dashboard
    assert "template=\"sustainscm\"" in dashboard
    assert "st.bar_chart(" not in dashboard
    assert "st.line_chart(" not in dashboard


def test_every_application_table_uses_downloadable_component():
    dashboard = Path("kpi_dashboard.py").read_text(encoding="utf-8")
    completion = Path("scenario_completion_page.py").read_text(encoding="utf-8")
    assert "st.dataframe(" not in dashboard
    assert "st.dataframe(" not in completion
    assert dashboard.count("render_downloadable_table(") == 17
    assert completion.count("render_downloadable_table(") == 6
