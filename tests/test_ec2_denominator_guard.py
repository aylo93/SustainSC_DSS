import pytest

from sustainsc.normalization import guard_denominator_only_improvement


def test_lower_share_without_lower_energy_cost_per_fu_is_neutralized():
    guarded, flagged = guard_denominator_only_improvement(65.0, 50.0, 2.0, 2.0)
    assert guarded == pytest.approx(50.0)
    assert flagged


def test_lower_share_with_lower_energy_cost_per_fu_remains_an_improvement():
    guarded, flagged = guard_denominator_only_improvement(65.0, 50.0, 1.8, 2.0)
    assert guarded == pytest.approx(65.0)
    assert not flagged


def test_energy_cost_deterioration_is_not_hidden_when_share_is_already_worse():
    guarded, flagged = guard_denominator_only_improvement(40.0, 50.0, 2.2, 2.0)
    assert guarded == pytest.approx(40.0)
    assert not flagged


def test_missing_corroboration_does_not_invent_a_guard_result():
    guarded, flagged = guard_denominator_only_improvement(65.0, 50.0, None, 2.0)
    assert guarded == pytest.approx(65.0)
    assert not flagged
