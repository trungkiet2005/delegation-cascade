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


def test_the_erosion_order_dominates_round_by_round() -> None:
    """Lemma 1, in the pathwise form the proof establishes.

    Against a fixed opponent the focal action path never falls anywhere along
    the erosion order.  This is what makes the harm order independent of the
    horizon law and of every payoff parameter, rather than a numerical
    coincidence at the baseline setting.
    """
    for col in STRATEGIES:
        for lo, hi in zip(STRATEGIES, STRATEGIES[1:]):
            for n in (1, 2, 3, 7, 40, 401):
                a_lo, _ = action_paths(lo, col, n)
                a_hi, _ = action_paths(hi, col, n)
                assert (a_hi >= a_lo).all(), (lo, hi, col, n)


def test_harm_never_falls_along_the_erosion_order() -> None:
    """The structural fact the whole depth argument rests on."""
    for p_max in (0.0, 0.1, 0.3, 0.6, 0.9, 1.0):
        for prize in (0.0, 50.0, 100.0, 200.0):
            for stop_prob in (0.05, 0.2, 0.5):
                for min_rounds in (1, 2, 5):
                    for scope in ("total", "prize"):
                        tables = build_race_tables(
                            RaceParams(
                                p_max=p_max,
                                prize=prize,
                                stop_prob=stop_prob,
                                min_rounds=min_rounds,
                                setback_scope=scope,
                            )
                        )
                        assert (np.diff(tables.unsafe_count, axis=0) >= -1e-12).all()
                        assert (np.diff(tables.unsafe_frequency, axis=0) >= -1e-12).all()


def test_the_harm_matrix_matches_its_closed_form() -> None:
    """``m`` depends on the horizon law only through E[T] and P(T odd)."""
    for stop_prob in (0.05, 0.2, 0.5, 0.8):
        for min_rounds in (1, 2, 5, 6):
            params = RaceParams(
                stop_prob=stop_prob, min_rounds=min_rounds, max_rounds=2000
            )
            t_values, probs = params.horizon_distribution()
            mu = float(t_values @ probs)
            odd = float(probs[t_values % 2 == 1].sum())
            expected = np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, (mu - odd) / 2.0, mu - 1.0],
                    [1.0, (mu + odd) / 2.0, mu, mu],
                    [mu, mu, mu, mu],
                ]
            )
            got = build_race_tables(params).unsafe_count
            assert np.allclose(got, expected, atol=1e-12), (stop_prob, min_rounds)


def test_harm_is_free_of_the_payoff_parameters() -> None:
    """``m`` and ``u`` are functionals of the action path alone."""
    base = build_race_tables(RaceParams())
    alt = build_race_tables(
        RaceParams(
            prize=7.0,
            p_max=0.05,
            stage_payoffs=np.array([[9.0, -3.0], [0.5, 2.0]]),
            setback_scope="prize",
        )
    )
    assert np.allclose(base.unsafe_count, alt.unsafe_count)
    assert np.allclose(base.unsafe_frequency, alt.unsafe_frequency)


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
