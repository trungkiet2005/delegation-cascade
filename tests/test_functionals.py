"""The payoff functionals over the (depth, intent) design space."""

from __future__ import annotations

import numpy as np
import pytest

from dcascade.chain import ChainParams
from dcascade.functionals import (
    aggregate_unsafe_frequency,
    build_functionals,
    declaration_gap,
    depth_distribution,
    executed_distribution,
    intent_distribution,
    mean_depth,
)
from dcascade.race import STRATEGIES, RaceParams, build_race_tables

RACE = build_race_tables(RaceParams())
CHAIN = ChainParams(max_depth=4, eps=0.2, phi=0.75, gain=1.5)


def test_the_design_space_has_the_expected_size() -> None:
    fun = build_functionals(RACE, CHAIN, 1.0, 20.0)
    assert fun.n == (CHAIN.max_depth + 1) * len(STRATEGIES)
    assert fun.pi_P.shape == (fun.n, fun.n)


def test_depth_zero_reproduces_the_interaction_layer() -> None:
    """With no hand-offs and no benefit the model collapses to the source game."""
    chain = ChainParams(max_depth=0, eps=0.3, phi=0.5, gain=0.0)
    fun = build_functionals(RACE, chain, 1.0, 0.0)
    assert np.allclose(fun.task, RACE.payoff)
    assert np.allclose(fun.harm, RACE.unsafe_count)
    assert np.allclose(fun.pi_P, fun.pi_S)


def test_payoffs_are_bilinear_in_the_transmission_laws() -> None:
    fun = build_functionals(RACE, CHAIN, 1.0, 20.0)
    for i in (0, 5, 11, fun.n - 1):
        for j in (2, 7, fun.n - 2):
            expected = fun.transmission[i] @ RACE.payoff @ fun.transmission[j]
            expected += CHAIN.organisational_benefit(int(fun.depth[i]))
            assert fun.task[i, j] == pytest.approx(expected)


def test_the_wedge_is_the_unattributed_harm() -> None:
    lam, harm = 1.0, 20.0
    fun = build_functionals(RACE, CHAIN, lam, harm)
    attribution = np.array([CHAIN.attribution(int(d)) for d in fun.depth])[:, None]
    assert np.allclose(fun.wedge, harm * (1.0 - lam * attribution) * fun.harm)


def test_the_wedge_widens_with_depth() -> None:
    fun = build_functionals(RACE, CHAIN, 1.0, 20.0)
    j = fun.index(0, "AU")
    values = [fun.wedge[fun.index(d, "AS"), j] for d in CHAIN.depths]
    assert all(b >= a - 1e-12 for a, b in zip(values, values[1:]))


def test_full_pass_through_closes_the_wedge() -> None:
    chain = ChainParams(max_depth=3, eps=0.2, phi=1.0, gain=1.0)
    fun = build_functionals(RACE, chain, 1.0, 20.0)
    assert np.allclose(fun.wedge, 0.0)
    assert np.allclose(fun.pi_P, fun.pi_S)


def test_harm_is_non_decreasing_in_depth_for_every_intent() -> None:
    fun = build_functionals(RACE, CHAIN, 1.0, 20.0)
    for intent in STRATEGIES:
        for j in range(fun.n):
            values = [fun.harm[fun.index(d, intent), j] for d in CHAIN.depths]
            assert all(b >= a - 1e-12 for a, b in zip(values, values[1:]))


def test_only_the_product_of_rate_and_harm_enters_the_private_functional() -> None:
    a = build_functionals(RACE, CHAIN, 1.0, 20.0)
    b = build_functionals(RACE, CHAIN, 2.0, 10.0)
    assert np.allclose(a.pi_P, b.pi_P)
    assert not np.allclose(a.pi_S, b.pi_S)


def test_population_summaries_agree_with_their_definitions() -> None:
    fun = build_functionals(RACE, CHAIN, 1.0, 20.0)
    rng = np.random.default_rng(1)
    x = rng.dirichlet(np.ones(fun.n))
    assert aggregate_unsafe_frequency(x, fun.unsafe_frequency) == pytest.approx(
        x @ fun.unsafe_frequency @ x
    )
    assert mean_depth(x, fun.depth) == pytest.approx(x @ fun.depth)
    assert sum(intent_distribution(x, fun.intent).values()) == pytest.approx(1.0)
    assert sum(depth_distribution(x, fun.depth).values()) == pytest.approx(1.0)
    assert executed_distribution(x, fun.transmission).sum() == pytest.approx(1.0)


def test_the_declaration_gap_vanishes_without_erosion() -> None:
    chain = ChainParams(max_depth=4, eps=0.0, phi=0.75, gain=1.5)
    fun = build_functionals(RACE, chain, 1.0, 20.0)
    rng = np.random.default_rng(2)
    x = rng.dirichlet(np.ones(fun.n))
    assert declaration_gap(x, fun.intent, fun.transmission) == pytest.approx(0.0, abs=1e-12)


def test_the_declaration_gap_is_positive_under_erosion() -> None:
    fun = build_functionals(RACE, CHAIN, 1.0, 20.0)
    x = np.zeros(fun.n)
    x[fun.index(CHAIN.max_depth, "AS")] = 1.0
    assert declaration_gap(x, fun.intent, fun.transmission) > 0.3


def test_invalid_parameters_are_rejected() -> None:
    with pytest.raises(ValueError):
        build_functionals(RACE, CHAIN, -1.0, 20.0)
    with pytest.raises(ValueError):
        build_functionals(RACE, CHAIN, 1.0, -5.0)
