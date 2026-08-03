import pandas as pd
import pytest

from sustainsc.ui.chart_theme import (
    build_horizontal_ranking_chart,
    ranking_chart_height,
    ranking_left_margin,
)


def test_horizontal_ranking_keeps_every_label_visible_including_winner():
    frame = pd.DataFrame(
        {
            "scenario_code": ["SCENARIO_A", "SCENARIO_B", "SCENARIO_C"],
            "score": [72.0, 50.0, 40.0],
        }
    )
    figure = build_horizontal_ranking_chart(
        frame,
        scenario_col="scenario_code",
        score_col="score",
        title="Ranking",
        x_title="Score",
        color="#2F6B9A",
    )

    categories = list(figure.layout.yaxis.categoryarray)
    plotted = list(figure.data[0].y)
    assert categories == ["SCENARIO_C", "SCENARIO_B", "SCENARIO_A"]
    assert categories[-1] == "SCENARIO_A"  # last category renders at the top
    assert plotted == categories
    assert figure.layout.yaxis.showticklabels is True
    assert figure.layout.yaxis.automargin is True
    assert figure.layout.yaxis.tickmode == "array"
    assert list(figure.layout.yaxis.tickvals) == categories
    assert list(figure.layout.yaxis.ticktext) == categories
    assert figure.layout.yaxis.ticklabeloverflow == "allow"
    assert len(plotted) == len(categories) == frame["scenario_code"].nunique()
    assert all(label for label in categories)
    assert figure.layout.margin.l >= 120
    assert figure.layout.margin.t >= 70
    assert figure.layout.height >= 440


@pytest.mark.parametrize("score_col", ["WSM_score", "TOPSIS_score"])
def test_cuba_winner_is_present_as_top_category_with_dynamic_ticks(score_col):
    frame = pd.DataFrame(
        {
            "scenario_code": ["BASE", "VSMC_KAIZEN", "MILP_MIN_CO2"],
            score_col: [40.0, 72.0, 50.0],
        }
    )
    figure = build_horizontal_ranking_chart(
        frame,
        scenario_col="scenario_code",
        score_col=score_col,
        title="Ranking",
        x_title="Score",
        color="#2F6B9A",
    )

    categories = list(figure.layout.yaxis.categoryarray)
    plotted = list(figure.data[0].y)
    assert categories == plotted
    assert categories[-1] == "VSMC_KAIZEN"
    assert len(categories) == len(set(categories)) == len(frame)
    assert list(figure.layout.yaxis.ticktext)[-1] == "VSMC_KAIZEN"


def test_all_24_ranking_labels_are_explicitly_rendered():
    labels = [f"SCENARIO_{index:02d}" for index in range(24)]
    frame = pd.DataFrame({"scenario": labels, "score": range(24)})
    figure = build_horizontal_ranking_chart(
        frame,
        scenario_col="scenario",
        score_col="score",
        title="Ranking",
        x_title="Score",
        color="#087F78",
    )

    assert len(figure.layout.yaxis.tickvals) == 24
    assert len(figure.layout.yaxis.ticktext) == 24
    assert set(figure.layout.yaxis.ticktext) == set(labels)


def test_horizontal_ranking_rejects_duplicate_or_missing_categories():
    duplicate = pd.DataFrame({"scenario": ["A", "A"], "score": [2.0, 1.0]})
    with pytest.raises(ValueError, match="unique"):
        build_horizontal_ranking_chart(
            duplicate,
            scenario_col="scenario",
            score_col="score",
            title="Ranking",
            x_title="Score",
            color="#2F6B9A",
        )


def test_ranking_layout_scales_for_many_and_long_scenarios():
    assert ranking_chart_height(24) == 956
    assert ranking_chart_height(100) == 1100
    assert ranking_left_margin(["A", "A_VERY_LONG_SCENARIO_IDENTIFIER"]) > 120
