"""The four instruments, and the claims made about how they compare."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from dcascade import config as cfg
from dcascade import interventions as iv
from dcascade import theory as th
from dcascade.chain import AuditPlan, ChainParams
from dcascade.functionals import build_functionals
from dcascade.race import build_race_tables

RACE = build_race_tables(cfg.RACE)
CHAIN = cfg.CHAIN
SML = dict(population_size=cfg.POPULATION, beta=cfg.BETA)


def test_a_null_intervention_reproduces_the_baseline() -> None:
    baseline = th.equilibrium(
        build_functionals(RACE, CHAIN, cfg.LAM, cfg.HARM), method="sml", **SML
    )
    for outcomes in (
        iv.pass_through_sweep(RACE, CHAIN, np.array([CHAIN.phi]), cfg.LAM, cfg.HARM, "sml", **SML),
        iv.attestation_sweep(RACE, CHAIN, np.array([1.0]), cfg.LAM, cfg.HARM, "sml", **SML),
        iv.depth_cap_sweep(
            RACE, CHAIN, np.array([CHAIN.max_depth]), cfg.LAM, cfg.HARM, "sml", **SML
        ),
    ):
        assert outcomes[0].unsafe_frequency == pytest.approx(
            baseline.unsafe_frequency, abs=1e-9
        )


def test_perfect_attestation_removes_the_erosion_channel() -> None:
    outcome = iv.attestation_sweep(
        RACE, CHAIN, np.array([0.0]), cfg.LAM, cfg.HARM, "sml", **SML
    )[0]
    no_erosion = th.equilibrium(
        build_functionals(RACE, replace(CHAIN, eps=0.0), cfg.LAM, cfg.HARM),
        method="sml",
        **SML,
    )
    assert outcome.unsafe_frequency == pytest.approx(no_erosion.unsafe_frequency, abs=1e-9)


def test_a_depth_ceiling_of_zero_removes_delegation() -> None:
    outcome = iv.depth_cap_sweep(
        RACE, CHAIN, np.array([0]), cfg.LAM, cfg.HARM, "sml", **SML
    )[0]
    assert outcome.mean_depth == pytest.approx(0.0)
    assert outcome.declaration_gap == pytest.approx(0.0, abs=1e-9)


def test_a_ceiling_at_the_social_optimum_beats_no_ceiling_on_both_counts() -> None:
    sweep = {
        int(o.setting): o
        for o in iv.depth_cap_sweep(
            RACE, CHAIN, np.arange(CHAIN.max_depth, -1, -1), cfg.LAM, cfg.HARM, "sml", **SML
        )
    }
    best = max(sweep.values(), key=lambda o: o.social_payoff)
    uncapped = sweep[CHAIN.max_depth]
    assert best.social_payoff > uncapped.social_payoff
    assert best.unsafe_frequency < uncapped.unsafe_frequency
    assert int(best.setting) < CHAIN.max_depth


def test_attribution_has_a_threshold_rather_than_a_gradient() -> None:
    sweep = iv.pass_through_sweep(
        RACE, CHAIN, np.linspace(CHAIN.phi, 1.0, 26), cfg.LAM, cfg.HARM, "sml", **SML
    )
    depths = np.array([o.mean_depth for o in sweep])
    unsafe = np.array([o.unsafe_frequency for o in sweep])
    drop = np.diff(depths)
    # one step accounts for most of the fall in chain length
    assert -drop.min() > 0.5 * (depths[0] - depths[-1])
    assert unsafe[0] > 10 * unsafe[-1]


def test_the_liability_instrument_leaves_the_true_harm_alone() -> None:
    """Raising the penalty must not make an Unsafe action less damaging."""
    sweep = iv.liability_sweep(
        RACE, CHAIN, np.array([cfg.HARM, 4 * cfg.HARM]), cfg.LAM, cfg.HARM, "sml", **SML
    )
    assert sweep[1].unsafe_frequency <= sweep[0].unsafe_frequency + 1e-9
    assert sweep[1].social_payoff > sweep[0].social_payoff


def test_a_check_is_worth_more_the_later_it_is_placed_for_a_fixed_design() -> None:
    """The mechanical claim, before the population is allowed to respond."""
    for strength in (0.4, 1.0):
        harms = []
        for layer in range(CHAIN.max_depth + 1):
            fun = build_functionals(
                RACE,
                replace(CHAIN, audit=AuditPlan(layer=layer, strength=strength)),
                cfg.LAM,
                cfg.HARM,
            )
            i = fun.index(CHAIN.max_depth, "AS")
            harms.append(fun.harm[i, i])
        assert all(b <= a + 1e-12 for a, b in zip(harms, harms[1:])), strength


def test_a_partial_check_late_in_the_chain_beats_an_early_one() -> None:
    sweep = iv.audit_placement_sweep(RACE, CHAIN, 0.5, cfg.LAM, cfg.HARM, "sml", **SML)
    assert sweep[-1].unsafe_frequency < sweep[0].unsafe_frequency


def test_fidelity_instruments_have_a_perverse_range() -> None:
    """Reported as a result: attestation is not monotone in its own strength."""
    sweep = iv.attestation_sweep(
        RACE, CHAIN, np.linspace(1.0, 0.0, 21), cfg.LAM, cfg.HARM, "sml", **SML
    )
    values = np.array([o.unsafe_frequency for o in sweep])
    assert (np.diff(values) > 1e-6).any()


def test_the_frontier_returns_paired_coordinates() -> None:
    outcomes = iv.depth_cap_sweep(
        RACE, CHAIN, np.arange(CHAIN.max_depth, -1, -1), cfg.LAM, cfg.HARM, "sml", **SML
    )
    social, unsafe = iv.instrument_frontier(outcomes)
    assert social.shape == unsafe.shape == (len(outcomes),)


def test_every_instrument_can_reach_a_moderate_target() -> None:
    matched = iv.matched_effort_comparison(
        RACE, CHAIN, 0.05, cfg.LAM, cfg.HARM, "sml", **SML
    )
    assert set(matched) == {"pass_through", "depth_cap", "attestation", "liability"}
    for name, row in matched.items():
        assert row["reaches_target"] == 1.0, name
        assert row["best_unsafe"] <= 0.05 + 1e-9
