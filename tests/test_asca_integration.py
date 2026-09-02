from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from asca import ASCAEngine


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "asca_assets"
ROMANIAN_CABLE_DESCRIPTION = (
    "Medium-sized cable-assembly manufacturer with limited digital maturity, "
    "traceability gaps and stable demand variability."
)


def test_asca_romanian_cable_poc_routes_only_validated_outputs() -> None:
    engine = ASCAEngine(ASSETS)
    suggestion = engine.suggest(ROMANIAN_CABLE_DESCRIPTION)
    evaluation = engine.evaluate(
        archetype=suggestion.archetype,
        size_class=suggestion.size_class,
        strategy=suggestion.strategy,
        lambda_intensity=suggestion.lambda_intensity,
        parameters=suggestion.parameters,
        suggestion=suggestion,
    )

    assert suggestion.archetype == "RO-A2"
    assert suggestion.strategy == "DIGITAL"
    assert suggestion.intensity_label == "MODERATE"
    assert evaluation.domain.status == "INSIDE_VALIDATED_DOMAIN"
    assert evaluation.predictions["route"].value_counts().to_dict() == {
        "SURROGATE_SCREENING": 10,
        "SURROGATE_EXPLORATORY": 5,
        "PARENT_MODEL_REQUIRED": 2,
    }
    parent_only = evaluation.predictions[
        evaluation.predictions["validation_status"].eq("FULL_MODEL_REQUIRED")
    ]
    assert parent_only["prediction"].isna().all()


def test_asca_boundary_control_blocks_every_surrogate() -> None:
    engine = ASCAEngine(ASSETS)
    suggestion = engine.suggest(
        "Medium-sized Romanian automotive component supplier with high logistics "
        "complexity, low renewable-energy use and moderate digital maturity."
    )
    parameters = dict(suggestion.parameters)
    parameters["oee"] = 0.90
    evaluation = engine.evaluate(
        archetype=suggestion.archetype,
        size_class=suggestion.size_class,
        strategy=suggestion.strategy,
        lambda_intensity=suggestion.lambda_intensity,
        parameters=parameters,
        suggestion=suggestion,
    )

    assert evaluation.domain.status == "OUTSIDE_FINITE_TRAINING_ENVELOPE"
    assert evaluation.predictions["prediction"].isna().all()
    assert set(evaluation.predictions["route"]) == {"PARENT_MODEL_REQUIRED"}


def test_asca_assets_and_home_navigation_are_complete() -> None:
    registry = json.loads(
        (ASSETS / "03_metamodels" / "model_registry.json").read_text(encoding="utf-8")
    )
    validation = pd.read_csv(
        ASSETS / "03_metamodels" / "holdout_validation_metrics.csv"
    )
    dashboard = (ROOT / "kpi_dashboard.py").read_text(encoding="utf-8")
    page = (ROOT / "pages" / "90_AI_Scenario_Agent.py").read_text(encoding="utf-8")
    config = (ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8")

    assert len(registry) == 17
    assert validation["study_acceptance"].value_counts().to_dict() == {
        "PASS": 10,
        "CONDITIONAL": 5,
        "FULL_MODEL_REQUIRED": 2,
    }
    assert "render_starting_options()" in dashboard
    assert "Open AI Scenario Agent" in dashboard
    assert (ROOT / "pages" / "90_AI_Scenario_Agent.py").is_file()
    assert "render_asca_page" in page
    assert "Back to SustainSCM home" in page
    assert 'type="primary"' in page
    assert 'icon=":material/arrow_back:"' in page
    assert "showSidebarNavigation = false" in config


def test_asca_page_runs_the_romanian_cable_example_end_to_end() -> None:
    app = AppTest.from_file(
        str(ROOT / "pages" / "90_AI_Scenario_Agent.py"),
        default_timeout=60,
    ).run(timeout=60)

    assert not app.exception
    assert app.selectbox[0].value == "Cable assembly / DIGITAL"

    next(
        button
        for button in app.button
        if button.label == "Interpret description with ASCA"
    ).click()
    app.run(timeout=60)
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["Suggested archetype"] == "RO-A2"
    assert metrics["Suggested strategy"] == "DIGITAL"

    next(
        button
        for button in app.button
        if button.label == "3 · Validate domain and run eligible metamodels"
    ).click()
    app.run(timeout=60)
    metrics = {metric.label: metric.value for metric in app.metric}
    assert not app.exception
    assert metrics["Validated screening"] == "10"
    assert metrics["Exploratory"] == "5"
    assert metrics["Parent model required"] == "2"
    assert [item.label for item in app.get("download_button")] == [
        "Download ASCA trace (JSON)",
        "Download metamodel screening (CSV)",
        "Download BASE comparison (CSV)",
        "Download interpretation (JSON)",
    ]
