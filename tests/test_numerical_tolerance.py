import pandas as pd

from sustainsc.normalization import normalize_value
from sustainsc.numerical import comparison_effect, snap_to_threshold


def test_threshold_noise_is_snapped_and_effect_is_same():
    assert snap_to_threshold(49.999997, [50.0, 80.0]) == 50.0
    assert comparison_effect(49.999997 - 50.0) == "Same"
    rule = pd.Series({
        "direction": "higher_better", "norm_method": "absolute_continuous",
        "lower_ref": 0.0, "upper_ref": 100.0,
        "amber_threshold": 50.0, "green_threshold": 80.0,
    })
    score, light = normalize_value(49.999997, rule)
    assert score == 50.0
    assert light == "Amber"


def test_genuine_difference_is_not_hidden():
    assert comparison_effect(-0.001) == "Worse"
