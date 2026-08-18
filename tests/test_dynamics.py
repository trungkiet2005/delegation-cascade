"""Evolutionary dynamics, including the closed form used in place of EGTtools."""

from __future__ import annotations

import numpy as np
import pytest

import egttools
from egttools.analytical import PairwiseComparison
from egttools.games import Matrix2PlayerGameHolder

from dcascade.dynamics import (
    average_replicator_attractor,
    fixation_probability,
    integrate_replicator,
    invades,
    neutrally_stable_strategies,
    replicator_field,
    sml_transition_matrix,
    stationary_analysis,
    stationary_analysis_sml,
    strict_nash_strategies,
)


def _random_game(seed: int, n: int = 4) -> np.ndarray:
    return np.ascontiguousarray(np.random.default_rng(seed).normal(size=(n, n)) * 3.0)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("beta", [0.05, 0.2])
def test_fixation_probability_matches_egttools(seed: int, beta: float) -> None:
    """The closed form replaces the generic routine, so it must agree with it."""
    payoff = _random_game(seed)
    size = 60
    # the evolver holds a raw pointer to the game, so the game must outlive it
    game = Matrix2PlayerGameHolder(4, payoff)
    evolver = PairwiseComparison(size, game)
    for i in range(4):
        for j in range(4):
            if i == j:
                continue
            mine = fixation_probability(payoff, i, j, size, beta)
            theirs = float(evolver.calculate_fixation_probability(i, j, beta))
            # EGTtools truncates strongly disadvantaged mutants to exactly zero;
            # see test_closed_form_resolves_probabilities_egttools_truncates
            assert mine == pytest.approx(theirs, abs=1e-6)


def test_closed_form_resolves_probabilities_egttools_truncates() -> None:
    """A rare mutant always has a positive fixation probability at finite beta.

    The generic EGTtools routine returns exactly zero for the case below; the
    closed form used here returns 3.287091609e-8, which agrees with a 60-digit
    evaluation of the same sum to ten significant figures.
    """
    payoff = _random_game(0)
    size, beta = 60, 0.05
    game = Matrix2PlayerGameHolder(4, payoff)
    evolver = PairwiseComparison(size, game)
    assert float(evolver.calculate_fixation_probability(2, 1, beta)) == 0.0
    assert fixation_probability(payoff, 2, 1, size, beta) == pytest.approx(
        3.287091609e-8, rel=1e-6
    )


def test_neutral_game_gives_the_neutral_fixation_probability() -> None:
    payoff = np.zeros((3, 3))
    assert fixation_probability(payoff, 0, 1, 50, 0.1) == pytest.approx(1 / 50)


def test_fixation_survives_a_large_selection_pressure() -> None:
    """The exponentials are shifted, so a strongly dominated mutant is not a NaN."""
    payoff = np.array([[0.0, 0.0], [500.0, 500.0]])
    value = fixation_probability(payoff, 0, 1, 200, 1.0)
    assert np.isfinite(value)
    assert 0.0 <= value < 1e-6


def test_sml_transition_matrix_is_row_stochastic() -> None:
    p = sml_transition_matrix(_random_game(3, 6), 80, 0.05)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert (p >= -1e-15).all()


def test_sml_stationary_distribution_is_a_distribution() -> None:
    payoff = _random_game(4, 8)
    result = stationary_analysis_sml(payoff, np.zeros((8, 8)), 100, 0.05)
    assert result.strategy_frequencies.sum() == pytest.approx(1.0)
    assert (result.strategy_frequencies >= -1e-12).all()


def test_sml_is_uniform_on_a_neutral_game() -> None:
    payoff = np.zeros((5, 5))
    result = stationary_analysis_sml(payoff, np.zeros((5, 5)), 60, 0.1)
    assert result.strategy_frequencies == pytest.approx(np.full(5, 0.2), abs=1e-9)


def test_sml_scales_to_the_full_design_space() -> None:
    """28 designs are out of reach for the state-space route but fine here."""
    payoff = _random_game(5, 28)
    result = stationary_analysis_sml(payoff, np.zeros((28, 28)), 100, 0.05)
    assert result.strategy_frequencies.sum() == pytest.approx(1.0)


def test_full_chain_and_small_mutation_limit_agree_as_mutation_vanishes() -> None:
    payoff = _random_game(6, 3)
    unsafe = np.abs(_random_game(7, 3)) / 10.0
    limit = stationary_analysis_sml(payoff, unsafe, 40, 0.05)
    coarse = stationary_analysis(payoff, unsafe, 40, 0.05, mu=0.05)
    fine = stationary_analysis(payoff, unsafe, 40, 0.05, mu=0.002)
    gap_coarse = np.abs(coarse.strategy_frequencies - limit.strategy_frequencies).max()
    gap_fine = np.abs(fine.strategy_frequencies - limit.strategy_frequencies).max()
    assert gap_fine < gap_coarse


def test_replicator_stays_on_the_simplex() -> None:
    payoff = _random_game(8, 5)
    x0 = np.full(5, 0.2)
    _, traj = integrate_replicator(payoff, x0, t_end=50.0, n_points=25)
    assert np.allclose(traj.sum(axis=1), 1.0)
    assert (traj >= -1e-9).all()


def test_replicator_field_sums_to_zero() -> None:
    payoff = _random_game(9, 4)
    x = np.array([0.1, 0.2, 0.3, 0.4])
    assert replicator_field(x, payoff).sum() == pytest.approx(0.0, abs=1e-12)


def test_dominant_strategy_takes_over() -> None:
    payoff = np.array([[5.0, 5.0], [0.0, 0.0]])
    end = average_replicator_attractor(payoff, n_starts=8, t_end=500.0)
    assert end[0] == pytest.approx(1.0, abs=1e-3)


def test_equilibrium_notions_on_a_known_game() -> None:
    # a prisoner's dilemma: defection is the only strict equilibrium
    payoff = np.array([[3.0, 0.0], [5.0, 1.0]])
    assert strict_nash_strategies(payoff) == [1]
    assert neutrally_stable_strategies(payoff) == [1]
    assert invades(payoff, 1, 0)
    assert not invades(payoff, 0, 1)
