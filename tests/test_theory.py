"""The closed-form results and the structural claims of the manuscript."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from dcascade import config as cfg
from dcascade import theory as th
from dcascade.chain import ChainParams
from dcascade.functionals import build_functionals
from dcascade.race import STRATEGIES, RaceParams, build_race_tables

RACE = build_race_tables(cfg.RACE)
CHAIN = cfg.CHAIN
SML = dict(population_size=cfg.POPULATION, beta=cfg.BETA)
FUN = build_functionals(RACE, CHAIN, cfg.LAM, cfg.HARM)


def test_the_erosion_order_is_a_harm_order() -> None:
    assert th.erosion_monotone(RACE)
    assert th.erosion_monotone_frequency(RACE)


def test_depth_profiles_are_non_decreasing_in_harm() -> None:
    for intent in STRATEGIES:
        profile = th.self_profile(FUN, intent)
        assert all(b >= a - 1e-12 for a, b in zip(profile.harm, profile.harm[1:]))


def test_attributed_harm_is_the_discounted_harm() -> None:
    profile = th.self_profile(FUN, "AS")
    discount = np.array([CHAIN.attribution(int(d)) for d in profile.depths])
    assert np.allclose(profile.attributed, cfg.LAM * discount * profile.harm)


def test_the_deniability_threshold_tends_to_one() -> None:
    """Realised harm saturates, so m(d)/m(d+1) rises to one: the shelter theorem."""
    probe = rb_profile = th.self_profile(
        build_functionals(RACE, replace(CHAIN, max_depth=40), cfg.LAM, cfg.HARM), "AS"
    )
    thresholds = [th.deniability_threshold(probe, d) for d in range(39)]
    thresholds = [t for t in thresholds if t is not None]
    assert thresholds[-1] > 0.999
    assert thresholds[-1] > thresholds[len(thresholds) // 2]
    assert rb_profile.harm[-1] > rb_profile.harm[5]


def test_any_attribution_below_one_eventually_opens_a_shelter() -> None:
    deep = build_functionals(RACE, replace(CHAIN, max_depth=40), cfg.LAM, cfg.HARM)
    profile = th.self_profile(deep, "AS")
    for phi in (0.5, 0.75, 0.9, 0.99):
        onset = th.shelter_depth(profile, phi)
        assert onset is not None, phi
        # beyond the onset every further hand-off lowers the attributed harm
        attributed = phi ** np.arange(41) * profile.harm
        assert all(
            b <= a + 1e-12 for a, b in zip(attributed[onset:], attributed[onset + 1 :])
        )


def test_full_attribution_never_opens_a_shelter() -> None:
    deep = build_functionals(RACE, replace(CHAIN, max_depth=40), cfg.LAM, cfg.HARM)
    profile = th.self_profile(deep, "AS")
    assert th.shelter_depth(profile, 1.0) is None


def test_depth_invasion_is_affine_in_the_liability() -> None:
    """Every invasion condition is a line in L, which is what gives L*(d)."""
    for intent in ("AS", "CS"):
        for depth in range(CHAIN.max_depth):
            values = []
            for liability in (0.0, 10.0, 20.0):
                fun = build_functionals(RACE, CHAIN, liability / cfg.HARM, cfg.HARM)
                i, j = fun.index(depth + 1, intent), fun.index(depth, intent)
                values.append(fun.pi_P[i, j] - fun.pi_P[j, j])
            slope = (values[1] - values[0]) / 10.0
            assert values[2] == pytest.approx(values[0] + 20.0 * slope, abs=1e-9)


def test_the_critical_liability_matches_the_invasion_formula() -> None:
    for intent in ("AS", "CS", "CAS"):
        for depth in range(CHAIN.max_depth):
            inv = th.depth_invasion(FUN, intent, depth)
            if inv.critical_liability is None:
                continue
            # the advantage is benefit - L * attributed_gain, so the root is exact
            assert inv.benefit == pytest.approx(
                inv.critical_liability * inv.attributed_gain, abs=1e-9
            )
            if inv.critical_liability <= 0.0:
                continue  # a negative root is outside the admissible range of L
            fun = build_functionals(
                RACE, CHAIN, inv.critical_liability / cfg.HARM, cfg.HARM
            )
            i, j = fun.index(depth + 1, intent), fun.index(depth, intent)
            assert fun.pi_P[i, j] - fun.pi_P[j, j] == pytest.approx(0.0, abs=1e-8)


def test_a_negative_attributed_gain_means_liability_cannot_stop_depth() -> None:
    inv = [th.depth_invasion(FUN, "CAS", d) for d in range(CHAIN.max_depth)]
    negative = [i for i in inv if i.attributed_gain < 0]
    assert negative, "the baseline should contain at least one sheltering step"
    for step in negative:
        assert step.direction == "deeper_above"
        # raising the liability only widens the deeper design's advantage
        assert step.deeper_invades_at(1e6)


def test_depth_zero_invasion_threshold_reproduces_the_source_game() -> None:
    """CAS invading AS at depth zero is the threshold of the single-layer study."""
    value = th.invasion_threshold_depth_zero(RACE, "CAS", "AS")
    assert value == pytest.approx(42.77, abs=0.02)


def test_the_depth_zero_race_is_safe_at_the_baseline_liability() -> None:
    critical = th.critical_liability(RACE, "sml", **SML)
    assert critical < cfg.effective_liability()
    chain = ChainParams(max_depth=0, eps=0.0, phi=1.0, gain=0.0)
    fun = build_functionals(RACE, chain, cfg.LAM, cfg.HARM)
    assert th.equilibrium(fun, method="sml", **SML).unsafe_frequency < 1e-3


def test_attribution_failure_depth_follows_its_formula() -> None:
    assert th.attribution_failure_depth(20.0, 5.0, 0.5) == pytest.approx(
        np.log(5.0 / 20.0) / np.log(0.5)
    )
    assert th.attribution_failure_depth(20.0, 5.0, 1.0) == float("inf")
    assert th.attribution_failure_depth(3.0, 5.0, 0.5) == 0.0


def test_the_two_mechanisms_are_super_additive_at_the_baseline() -> None:
    dec = th.mechanism_decomposition(RACE, CHAIN, cfg.LAM, cfg.HARM, method="sml", **SML)
    assert dec.baseline.unsafe_frequency < 1e-3
    assert dec.both.unsafe_frequency > 0.25
    assert dec.interaction > 0.2
    assert dec.interaction_share > 0.8


def test_per_layer_attribution_removes_the_depth_response() -> None:
    """Under pass-through the chain shortens; per-layer attribution stops that."""
    dec = th.mechanism_decomposition(RACE, CHAIN, cfg.LAM, cfg.HARM, method="sml", **SML)
    assert dec.drift_only.mean_depth < dec.baseline.mean_depth - 1.0
    assert dec.both.mean_depth > dec.drift_only.mean_depth + 1.0


def test_the_increase_is_composition_not_behaviour() -> None:
    frozen = th.frozen_depth_counterfactual(
        RACE, CHAIN, cfg.LAM, cfg.HARM, method="sml", **SML
    )
    assert frozen["unsafe_frozen_composition"] == pytest.approx(
        frozen["unsafe_passthrough"], abs=1e-9
    )
    assert frozen["composition_channel"] > 0.25


def test_the_population_declares_safety_it_does_not_execute() -> None:
    eq = th.equilibrium(FUN, method="sml", **SML)
    assert eq.intent_distribution["AS"] > 0.99
    assert eq.executed_distribution[0] < 0.4
    assert eq.declaration_gap > 0.5


def test_the_selected_depth_beats_neither_optimum() -> None:
    profile = th.self_profile(FUN, "AS")
    outcome = th.equilibrium(FUN, method="sml", **SML).mean_depth
    assert outcome > CHAIN.max_depth - 0.1
    assert profile.private[-1] < profile.private.max() - 1.0
    assert profile.social[-1] < profile.social.max() - 1.0
    assert th.socially_optimal_depth(FUN, "AS") < CHAIN.max_depth


def test_the_instruction_loses_force_with_depth() -> None:
    separation = [
        th.behavioural_leverage(FUN)[d] for d in range(CHAIN.max_depth + 1)
    ]
    assert separation[0] >= separation[-1]


def test_equilibrium_rejects_an_unknown_method() -> None:
    with pytest.raises(ValueError):
        th.equilibrium(FUN, method="nonsense")
