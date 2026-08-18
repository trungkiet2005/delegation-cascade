"""The interaction layer is evaluated exactly; this checks it against sampling."""

from __future__ import annotations

import numpy as np
import pytest

from dcascade.race import (
    SAFE,
    STRATEGIES,
    UNSAFE,
    RaceParams,
    action_paths,
    build_race_tables,
    evaluate_matchup,
)


def _simulate(row: str, col: str, params: RaceParams, n: int, seed: int = 7):
    """Monte Carlo counterpart of :func:`evaluate_matchup`."""
    rng = np.random.default_rng(seed)
    horizons = params.min_rounds + rng.geometric(params.stop_prob, size=n) - 1
    horizons = np.minimum(horizons, params.max_rounds)
    a_row, a_col = action_paths(row, col, int(horizons.max()))

    stage = np.cumsum(params.stage_payoffs[a_row, a_col])
    steps = np.cumsum(np.where(a_row == UNSAFE, params.step_unsafe, params.step_safe))
    steps_opp = np.cumsum(np.where(a_col == UNSAFE, params.step_unsafe, params.step_safe))
    unsafe = np.cumsum(a_row == UNSAFE)

    idx = horizons - 1
    freq = unsafe[idx] / horizons
    q = params.p_max * freq
    wins = steps[idx] > steps_opp[idx]
    ties = steps[idx] == steps_opp[idx]
    at_risk = wins | ties
    prize = np.where(wins, params.prize, 0.0) + np.where(ties, params.prize / 2, 0.0)
    kept = stage[idx] + prize
    payoff = np.where(at_risk, (1.0 - q) * kept, stage[idx])
    return payoff.mean(), unsafe[idx].mean(), freq.mean()


@pytest.mark.parametrize("row", STRATEGIES)
@pytest.mark.parametrize("col", STRATEGIES)
def test_exact_matches_simulation(row: str, col: str) -> None:
    params = RaceParams()
    exact = evaluate_matchup(row, col, params)
    payoff, count, freq = _simulate(row, col, params, n=200_000)
    assert exact.payoff == pytest.approx(payoff, rel=0.02, abs=0.5)
    assert exact.unsafe_count == pytest.approx(count, rel=0.02, abs=0.05)
    assert exact.unsafe_frequency == pytest.approx(freq, rel=0.02, abs=0.01)


def test_horizon_law_is_a_probability_distribution() -> None:
    params = RaceParams()
    values, probs = params.horizon_distribution()
    assert probs.sum() == pytest.approx(1.0, abs=1e-12)
    assert (probs >= 0).all()
    assert values[0] == params.min_rounds
    # mean of a shifted geometric: min_rounds - 1 + 1 / stop_prob
    assert params.expected_horizon == pytest.approx(
        params.min_rounds - 1 + 1 / params.stop_prob, rel=1e-9
    )


def test_action_paths_follow_the_reduced_definitions() -> None:
    row, col = action_paths("AS", "AU", 4)
    assert list(row) == [SAFE] * 4
    assert list(col) == [UNSAFE] * 4

    row, col = action_paths("CS", "CAS", 4)
    # CS opens Safe and copies; CAS opens Unsafe and copies
    assert list(row) == [SAFE, UNSAFE, SAFE, UNSAFE]
    assert list(col) == [UNSAFE, SAFE, UNSAFE, SAFE]


def test_designs_are_listed_in_the_erosion_order() -> None:
    assert STRATEGIES == ("AS", "CS", "CAS", "AU")


def test_harm_never_falls_along_the_erosion_order() -> None:
    """The structural fact the whole depth argument rests on."""
    for p_max in (0.1, 0.3, 0.6, 0.9):
        for prize in (50.0, 100.0, 200.0):
            tables = build_race_tables(RaceParams(p_max=p_max, prize=prize))
            assert (np.diff(tables.unsafe_count, axis=0) >= -1e-12).all()
            assert (np.diff(tables.unsafe_frequency, axis=0) >= -1e-12).all()


def test_always_safe_never_acts_unsafely() -> None:
    tables = build_race_tables(RaceParams())
    assert np.allclose(tables.unsafe_count[0], 0.0)
    assert np.allclose(tables.unsafe_frequency[0], 0.0)


def test_always_unsafe_always_acts_unsafely() -> None:
    tables = build_race_tables(RaceParams())
    assert np.allclose(tables.unsafe_frequency[3], 1.0)
    assert np.allclose(tables.unsafe_count[3], RaceParams().expected_horizon)


def test_setback_scope_only_changes_payoffs() -> None:
    total = build_race_tables(RaceParams(setback_scope="total"))
    prize_only = build_race_tables(RaceParams(setback_scope="prize"))
    assert np.allclose(total.unsafe_count, prize_only.unsafe_count)
    assert not np.allclose(total.payoff, prize_only.payoff)
