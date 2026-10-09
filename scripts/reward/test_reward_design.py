"""Tests for the geometric-mean reward design.

The claims that justify switching away from a weighted sum are testable, so they
are tested here rather than asserted in a docstring.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import reward_design as R


# --------------------------------------------------------------------------
# the ramp
# --------------------------------------------------------------------------


def test_ramp_spans_zero_to_one():
    c = R.Component("t", "m", lo=0.2, hi=0.8, higher_is_better=True,
                    depends_on_oracle=True, hackability=0.1)
    assert c.desirability(0.2) == pytest.approx(R.DESIRABILITY_FLOOR)
    assert c.desirability(0.8) == pytest.approx(1.0)
    assert c.desirability(0.0) == pytest.approx(R.DESIRABILITY_FLOOR)
    assert c.desirability(1.0) == pytest.approx(1.0)


def test_ramp_midpoint_is_half():
    c = R.Component("t", "m", lo=0.0, hi=1.0, higher_is_better=True,
                    depends_on_oracle=True, hackability=0.1)
    assert c.desirability(0.5) == pytest.approx(0.5)


def test_ramp_steepest_at_midpoint():
    """The user asked for maximum slope at the middle of the ramp."""
    c = R.Component("t", "m", lo=0.0, hi=1.0, higher_is_better=True,
                    depends_on_oracle=True, hackability=0.1)
    h = 1e-4

    def slope(x):
        return (c.desirability(x + h) - c.desirability(x - h)) / (2 * h)

    assert slope(0.5) > slope(0.35) > slope(0.2)
    assert slope(0.5) > slope(0.65) > slope(0.8)
    assert slope(0.5) == pytest.approx(math.pi / 2, rel=1e-3)


def test_ramp_handles_inverted_direction():
    """RMSD improves downward, so lo > hi."""
    c = R.Component("rmsd", "rmsd", lo=12.0, hi=2.0, higher_is_better=False,
                    depends_on_oracle=True, hackability=0.1)
    assert c.desirability(12.0) == pytest.approx(R.DESIRABILITY_FLOOR)
    assert c.desirability(2.0) == pytest.approx(1.0)
    assert c.desirability(7.0) == pytest.approx(0.5)
    assert c.desirability(20.0) == pytest.approx(R.DESIRABILITY_FLOOR)
    assert c.desirability(1.0) == pytest.approx(1.0)


def test_threshold_sits_above_midpoint():
    c = R.Component("t", "m", lo=0.0, hi=1.0, higher_is_better=True,
                    depends_on_oracle=True, hackability=0.1)
    assert c.threshold == pytest.approx(0.6)
    assert c.threshold > 0.5


def test_ramp_rejects_non_finite():
    c = R.Component("t", "m", lo=0.0, hi=1.0, higher_is_better=True,
                    depends_on_oracle=True, hackability=0.1)
    assert c.desirability(None) is None
    assert c.desirability(float("nan")) is None
    assert c.desirability(float("inf")) is None


# --------------------------------------------------------------------------
# the composite: this is the whole point of the rewrite
# --------------------------------------------------------------------------


def test_geometric_mean_of_equal_terms_is_that_value():
    assert R.composite([0.5, 0.5, 0.5], [1, 1, 1]) == pytest.approx(0.5)


def test_a_dead_term_collapses_the_composite():
    """The property a weighted sum does not have."""
    healthy = R.composite([0.9, 0.9, 0.9], [1, 1, 1])
    one_dead = R.composite([R.DESIRABILITY_FLOOR, 0.9, 0.9], [1, 1, 1])
    assert healthy > 0.85
    # a sum would give (0 + 0.9 + 0.9)/3 = 0.60; the geometric mean must be far lower
    assert one_dead < 0.10
    assert one_dead < healthy / 8


def test_no_compensation_between_terms():
    """Perfecting one term cannot rescue a failing one."""
    balanced = R.composite([0.5, 0.5], [1, 1])
    lopsided = R.composite([1.0, 0.05], [1, 1])
    assert balanced > lopsided


def test_composite_is_bounded():
    for values in ([0.1, 0.2, 0.3], [1.0, 1.0], [0.5, 0.9, 0.01]):
        result = R.composite(values, [1] * len(values))
        assert 0.0 <= result <= 1.0


def test_missing_terms_are_dropped_not_imputed():
    """A missing oracle score is missing data, not evidence of badness."""
    with_missing = R.composite([0.8, None, 0.8], [1, 1, 1])
    without = R.composite([0.8, 0.8], [1, 1])
    assert with_missing == pytest.approx(without)


def test_composite_returns_none_when_nothing_is_available():
    assert R.composite([None, None], [1, 1]) is None
    assert R.composite([0.5, 0.5], [0, 0]) is None


def test_weights_shift_the_composite_toward_the_heavier_term():
    low_weight_on_bad = R.composite([0.9, 0.2], [0.9, 0.1])
    high_weight_on_bad = R.composite([0.9, 0.2], [0.1, 0.9])
    assert low_weight_on_bad > high_weight_on_bad


# --------------------------------------------------------------------------
# vectorised path must agree with the scalar one
# --------------------------------------------------------------------------


def _toy_dataset(seed: int = 0, groups: int = 12, per_group: int = 8) -> R.Dataset:
    rng = np.random.default_rng(seed)
    n = groups * per_group
    truth = rng.uniform(0, 1, size=n)
    # three terms of decreasing fidelity to the truth
    d = np.stack([
        np.clip(truth + rng.normal(0, 0.05, n), 0.01, 1.0),
        np.clip(truth + rng.normal(0, 0.15, n), 0.01, 1.0),
        np.clip(rng.uniform(0, 1, n), 0.01, 1.0),
    ])
    return R.Dataset(
        desirabilities=d,
        objective=truth,
        groups=np.repeat(np.arange(groups), per_group),
        names=["good", "noisy", "junk"],
    )


def test_vectorised_rewards_match_scalar_composite():
    data = _toy_dataset()
    weights = np.array([0.5, 0.3, 0.2])
    vector = data.rewards(weights)
    for i in range(0, len(vector), 7):
        scalar = R.composite(list(data.desirabilities[:, i]), list(weights))
        assert vector[i] == pytest.approx(scalar, rel=1e-9)


def test_vectorised_rewards_handle_nan_columns():
    data = _toy_dataset()
    data.desirabilities[1, 3] = np.nan
    weights = np.array([0.5, 0.3, 0.2])
    vector = data.rewards(weights)
    expected = R.composite([data.desirabilities[0, 3], None, data.desirabilities[2, 3]],
                           [0.5, 0.3, 0.2])
    assert vector[3] == pytest.approx(expected, rel=1e-9)


def test_reward_is_nan_when_a_candidate_has_no_terms():
    data = _toy_dataset()
    data.desirabilities[:, 5] = np.nan
    assert math.isnan(data.rewards(np.array([0.4, 0.3, 0.3]))[5])


# --------------------------------------------------------------------------
# the measured properties
# --------------------------------------------------------------------------


def test_alignment_prefers_the_faithful_term():
    data = _toy_dataset()
    faithful = R.alignment(np.array([1.0, 0.0, 0.0]), data)
    junk = R.alignment(np.array([0.0, 0.0, 1.0]), data)
    assert faithful > 0.9
    assert faithful > junk


def test_discrimination_is_positive_for_a_faithful_reward():
    data = _toy_dataset()
    assert R.discrimination(np.array([1.0, 0.0, 0.0]), data) > 0.5


def test_discrimination_is_near_zero_for_noise():
    data = _toy_dataset()
    assert abs(R.discrimination(np.array([0.0, 0.0, 1.0]), data)) < 0.4


def test_exposure_tracks_the_worst_single_term():
    components = [
        R.Component("safe", "a", 0, 1, True, True, hackability=0.1),
        R.Component("risky", "b", 0, 1, True, False, hackability=0.9),
    ]
    assert R.exposure(np.array([0.9, 0.1]), components) == pytest.approx(0.09 * 1.0, abs=1e-9)
    assert R.exposure(np.array([0.5, 0.5]), components) == pytest.approx(0.45)
    # robustness is just the negation, so more mass on the risky term is worse
    assert R.robustness(np.array([0.9, 0.1]), components) > R.robustness(np.array([0.1, 0.9]), components)


def test_spearman_is_invariant_to_monotone_rescaling():
    a = [1, 2, 3, 4, 5]
    b = [10, 20, 30, 40, 50]
    c = [math.exp(x) for x in a]
    assert R.spearman(a, b) == pytest.approx(1.0)
    assert R.spearman(a, c) == pytest.approx(1.0)
    assert R.spearman(a, list(reversed(a))) == pytest.approx(-1.0)


def test_spearman_averages_ties():
    assert abs(R.spearman([1, 1, 1, 1], [1, 2, 3, 4])) < 1e-9


# --------------------------------------------------------------------------
# the constrained programme
# --------------------------------------------------------------------------


def test_simplex_grid_sums_to_one():
    grid = R.grid_simplex(3, 0.25)
    assert grid.shape[1] == 3
    assert np.allclose(grid.sum(axis=1), 1.0)
    assert len(grid) == 15  # C(4+2,2)


def test_floors_are_constraints_not_preferences():
    """An unreachable alignment floor must report infeasible, not degrade."""
    data = _toy_dataset()
    components = [
        R.Component("good", "a", 0, 1, True, True, hackability=0.1, is_primary=True),
        R.Component("noisy", "b", 0, 1, True, True, hackability=0.2, is_primary=True),
        R.Component("junk", "c", 0, 1, True, False, hackability=0.9),
    ]
    impossible = R.Constraints(alignment_min=0.999, discrimination_min=0.0,
                               primary_floor=0.0, per_term_cap=1.0)
    result = R.optimise(components, data, impossible, step=0.25)
    assert result["feasible"] is False
    assert result["solution"] is None
    assert result["blocked_by"]  # and it says why


def test_solution_satisfies_every_declared_floor():
    data = _toy_dataset()
    components = [
        R.Component("good", "a", 0, 1, True, True, hackability=0.1, is_primary=True),
        R.Component("noisy", "b", 0, 1, True, True, hackability=0.2, is_primary=True),
        R.Component("junk", "c", 0, 1, True, False, hackability=0.9),
    ]
    cons = R.Constraints(alignment_min=0.5, discrimination_min=0.3,
                         primary_floor=0.6, per_term_cap=0.8)
    result = R.optimise(components, data, cons, step=0.1)
    assert result["feasible"]
    solution = result["solution"]
    assert solution["alignment"] >= cons.alignment_min - 1e-9
    assert solution["discrimination"] >= cons.discrimination_min - 1e-9
    primary = solution["weights"]["good"] + solution["weights"]["noisy"]
    assert primary >= cons.primary_floor - 1e-9
    assert max(solution["weights"].values()) <= cons.per_term_cap + 1e-9


def test_solver_minimises_exposure_among_feasible_points():
    """The objective is robustness, so the answer must be the safest feasible one."""
    data = _toy_dataset()
    components = [
        R.Component("good", "a", 0, 1, True, True, hackability=0.1, is_primary=True),
        R.Component("noisy", "b", 0, 1, True, True, hackability=0.2, is_primary=True),
        R.Component("junk", "c", 0, 1, True, False, hackability=0.9),
    ]
    cons = R.Constraints(alignment_min=0.4, discrimination_min=0.2,
                         primary_floor=0.5, per_term_cap=1.0)
    result = R.optimise(components, data, cons, step=0.1)
    assert result["feasible"]
    best = np.array(result["solution"]["weight_vector"])
    best_exposure = R.exposure(best, components)
    for weights in R.grid_simplex(3, 0.1):
        ok, _ = R.feasible(weights, components, data, cons)
        if ok:
            assert R.exposure(weights, components) >= best_exposure - 1e-9


def test_feasible_reports_the_violated_constraint():
    data = _toy_dataset()
    components = [
        R.Component("good", "a", 0, 1, True, True, hackability=0.1, is_primary=True),
        R.Component("noisy", "b", 0, 1, True, True, hackability=0.2, is_primary=True),
        R.Component("junk", "c", 0, 1, True, False, hackability=0.9),
    ]
    cons = R.Constraints(alignment_min=0.0, discrimination_min=0.0,
                         primary_floor=0.9, per_term_cap=1.0)
    ok, reason = R.feasible(np.array([0.1, 0.1, 0.8]), components, data, cons)
    assert not ok
    assert "primary mass" in reason


def test_negative_and_unnormalised_weights_are_rejected():
    data = _toy_dataset()
    components = [
        R.Component("a", "a", 0, 1, True, True, hackability=0.1),
        R.Component("b", "b", 0, 1, True, True, hackability=0.1),
        R.Component("c", "c", 0, 1, True, True, hackability=0.1),
    ]
    cons = R.Constraints(alignment_min=-1, discrimination_min=-1,
                         primary_floor=0.0, per_term_cap=1.0)
    assert not R.feasible(np.array([-0.1, 0.6, 0.5]), components, data, cons)[0]
    assert not R.feasible(np.array([0.5, 0.3, 0.9]), components, data, cons)[0]


def test_pareto_sweep_shows_infeasibility_boundary():
    """Raising the floors must eventually make the problem infeasible."""
    data = _toy_dataset()
    components = [
        R.Component("good", "a", 0, 1, True, True, hackability=0.1, is_primary=True),
        R.Component("noisy", "b", 0, 1, True, True, hackability=0.2, is_primary=True),
        R.Component("junk", "c", 0, 1, True, False, hackability=0.9),
    ]
    frontier = R.pareto_frontier(
        components, data,
        alignment_levels=[0.3, 0.9, 0.999],
        discrimination_levels=[0.2],
        base=R.Constraints(primary_floor=0.0, per_term_cap=1.0),
        step=0.2,
    )
    assert frontier[0]["feasible"]
    assert frontier[-1]["feasible"] is False


def test_default_components_are_internally_consistent():
    components = R.default_components()
    names = [c.name for c in components]
    assert len(names) == len(set(names))
    assert sum(1 for c in components if c.is_primary) == 3
    for c in components:
        assert 0.0 <= c.hackability <= 1.0
        # every primary term routes through the oracle, and the exploitable
        # thermodynamic proxies do not
        if c.is_primary:
            assert c.depends_on_oracle
        assert c.desirability(c.lo) == pytest.approx(R.DESIRABILITY_FLOOR)
        assert c.desirability(c.hi) == pytest.approx(1.0)


def test_oracle_terms_are_less_hackable_than_proxies():
    components = R.default_components()
    oracle = [c.hackability for c in components if c.depends_on_oracle]
    proxy = [c.hackability for c in components if not c.depends_on_oracle]
    assert max(oracle) < min(proxy)


def test_report_renders_both_outcomes():
    data = _toy_dataset()
    # the dataset carries three terms, so the component list has to match it
    components = [
        R.Component("good", "a", 0, 1, True, True, hackability=0.1, is_primary=True),
        R.Component("noisy", "b", 0, 1, True, True, hackability=0.2, is_primary=True),
        R.Component("junk", "c", 0, 1, True, False, hackability=0.9),
    ]
    ok = R.optimise(components, data,
                    R.Constraints(alignment_min=0.3, discrimination_min=0.1,
                                  primary_floor=0.5, per_term_cap=1.0), step=0.25)
    text = R.report(ok)
    assert "weights:" in text and "alignment" in text

    bad = R.optimise(components, data,
                     R.Constraints(alignment_min=0.9999, discrimination_min=0.1,
                                   primary_floor=0.5, per_term_cap=1.0), step=0.25)
    assert "INFEASIBLE" in R.report(bad)
