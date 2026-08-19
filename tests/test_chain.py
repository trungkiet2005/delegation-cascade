"""The delegation layer: transmission laws, the erosion law and audits."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import binom

from dcascade.chain import (
    AuditPlan,
    ChainParams,
    design_labels,
    design_space,
    drift_direction,
    drift_matrix,
    fidelity,
    handoff_kernel,
    intent_separation,
    second_eigenvalue,
    specification_half_life,
    transmission,
    transmission_stack,
)
from dcascade.race import STRATEGIES

KERNELS = ("ladder", "collapse", "uniform")


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("eps", [0.0, 0.05, 0.2, 0.5, 1.0])
def test_kernels_are_row_stochastic(kernel: str, eps: float) -> None:
    m = handoff_kernel(eps, kernel)
    assert np.allclose(m.sum(axis=1), 1.0)
    assert (m >= -1e-15).all()


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("depth", [0, 1, 3, 7])
def test_transmission_laws_are_distributions(kernel: str, depth: int) -> None:
    p = transmission(depth, 0.2, kernel)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert (p >= -1e-15).all()


def test_zero_depth_transmits_the_intent_exactly() -> None:
    assert np.allclose(transmission(0, 0.3), np.eye(4))


def test_zero_erosion_transmits_the_intent_at_any_depth() -> None:
    assert np.allclose(transmission(12, 0.0), np.eye(4))


@pytest.mark.parametrize("depth", range(7))
def test_ladder_erosion_is_a_truncated_binomial(depth: int) -> None:
    """The executed design is the intent shifted by Binomial(d, eps) rungs."""
    eps = 0.2
    p = transmission(depth, eps, "ladder")
    for start in range(4):
        expected = np.zeros(4)
        for shift in range(4 - start - 1):
            expected[start + shift] = binom.pmf(shift, depth, eps)
        expected[3] = 1.0 - expected[:3].sum()
        assert np.allclose(p[start], expected, atol=1e-12)


@pytest.mark.parametrize("eps", [0.05, 0.15, 0.3, 0.45])
@pytest.mark.parametrize("depth", range(8))
def test_fidelity_is_exactly_geometric(eps: float, depth: int) -> None:
    """A specification survives d hand-offs intact with probability (1-eps)^d."""
    values = fidelity(depth, eps, "ladder")
    assert values[:3] == pytest.approx((1.0 - eps) ** depth, abs=1e-12)
    assert values[3] == pytest.approx(1.0)  # the absorbing design cannot erode


@pytest.mark.parametrize("kernel", ("ladder", "collapse"))
@pytest.mark.parametrize("eps", [0.05, 0.2, 0.4])
def test_half_life_follows_the_second_eigenvalue(kernel: str, eps: float) -> None:
    assert second_eigenvalue(eps, kernel) == pytest.approx(1.0 - eps)
    half = specification_half_life(eps, kernel)
    assert (1.0 - eps) ** half == pytest.approx(0.5, abs=1e-12)


def test_perfect_transmission_has_an_infinite_half_life() -> None:
    assert specification_half_life(0.0) == float("inf")


def test_intent_separation_decays_with_depth() -> None:
    values = [intent_separation(d, 0.2) for d in range(10)]
    assert values[0] == pytest.approx(1.0)
    assert all(b <= a + 1e-12 for a, b in zip(values, values[1:]))


def test_absorbing_design_is_absorbing_for_directional_kernels() -> None:
    for kernel in ("ladder", "collapse"):
        m = handoff_kernel(0.3, kernel)
        assert m[3, 3] == pytest.approx(1.0)


def test_uniform_kernel_moves_mass_both_ways() -> None:
    """The unbiased control is deliberately not monotone in the erosion order."""
    m = handoff_kernel(0.3, "uniform")
    assert m[3, 0] > 0.0  # the absorbing design of the ladder is not absorbing here
    ladder = handoff_kernel(0.3, "ladder")
    assert ladder[3, 0] == 0.0


def test_audit_restores_part_of_the_intent() -> None:
    eps, depth = 0.25, 5
    plain = transmission(depth, eps)
    for layer in range(1, depth + 1):
        checked = transmission(depth, eps, audit=AuditPlan(layer, 1.0))
        # more of the AS intent survives, and less reaches the unsafe end
        assert checked[0, 0] >= plain[0, 0] - 1e-12
        assert checked[0, 3] <= plain[0, 3] + 1e-12


def test_a_later_audit_is_worth_more() -> None:
    eps, depth = 0.25, 6
    fidelities = [
        transmission(depth, eps, audit=AuditPlan(layer, 1.0))[0, 0]
        for layer in range(depth + 1)
    ]
    assert all(b >= a - 1e-12 for a, b in zip(fidelities, fidelities[1:]))


def test_audit_beyond_the_chain_never_fires() -> None:
    plain = transmission(3, 0.2)
    assert np.allclose(transmission(3, 0.2, audit=AuditPlan(5, 1.0)), plain)


def test_audit_of_zero_strength_changes_nothing() -> None:
    plain = transmission(4, 0.2)
    assert np.allclose(transmission(4, 0.2, audit=AuditPlan(2, 0.0)), plain)


def test_transmission_stack_matches_the_single_depth_call() -> None:
    stack = transmission_stack(5, 0.2)
    for d in range(6):
        assert np.allclose(stack[d], transmission(d, 0.2))


def test_design_space_is_depth_major_and_labelled() -> None:
    chain = ChainParams(max_depth=2)
    space = design_space(chain)
    assert len(space) == 3 * len(STRATEGIES)
    assert space[0] == (0, "AS")
    assert space[-1] == (2, "AU")
    assert design_labels(chain)[0] == "AS@0"


def test_attribution_and_benefit_follow_their_formulas() -> None:
    chain = ChainParams(phi=0.8, gain=2.0, curvature=0.25)
    assert chain.attribution(3) == pytest.approx(0.8**3)
    assert chain.organisational_benefit(3) == pytest.approx(2.0 * 3 - 0.25 * 9)


def test_invalid_parameters_are_rejected() -> None:
    with pytest.raises(ValueError):
        handoff_kernel(1.5)
    with pytest.raises(ValueError):
        transmission(-1, 0.2)
    with pytest.raises(ValueError):
        drift_matrix("nonsense")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ChainParams(phi=-0.1)
    with pytest.raises(ValueError):
        ChainParams(attribution_rule="nonsense")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ChainParams(attribution_floor=1.5)
    with pytest.raises(ValueError):
        drift_matrix("mixed:1.5")
    with pytest.raises(ValueError):
        drift_matrix("nonsense:0.5")
    with pytest.raises(ValueError):
        AuditPlan(layer=-1)


def test_attribution_rules_follow_their_definitions() -> None:
    depths = range(7)
    geo = ChainParams(max_depth=6, phi=0.75)
    assert [geo.attribution(d) for d in depths] == [pytest.approx(0.75**d) for d in depths]

    strict = ChainParams(max_depth=6, phi=0.75, attribution_rule="strict")
    assert [strict.attribution(d) for d in depths] == [1.0] * 7

    equal = ChainParams(max_depth=6, phi=0.75, attribution_rule="harmonic")
    assert [equal.attribution(d) for d in depths] == [
        pytest.approx(1.0 / (1 + d)) for d in depths
    ]

    floored = ChainParams(
        max_depth=6, phi=0.75, attribution_rule="floored", attribution_floor=0.3
    )
    assert all(floored.attribution(d) >= 0.3 - 1e-12 for d in depths)
    assert floored.attribution(0) == pytest.approx(1.0)
    # a floor of zero is the geometric rule and a floor of one is strict liability
    assert ChainParams(
        phi=0.75, attribution_rule="floored", attribution_floor=0.0
    ).attribution(4) == pytest.approx(0.75**4)
    assert ChainParams(
        phi=0.75, attribution_rule="floored", attribution_floor=1.0
    ).attribution(4) == pytest.approx(1.0)


def test_super_attribution_is_allowed_and_grows_with_depth() -> None:
    """phi > 1 charges a principal more the deeper it delegates."""
    chain = ChainParams(phi=1.2)
    values = [chain.attribution(d) for d in range(5)]
    assert all(b > a for a, b in zip(values, values[1:]))


def test_the_floor_makes_attributed_harm_non_decreasing_in_depth() -> None:
    """Proposition 8: above the depth where the floor binds, depth stops paying."""
    from dcascade.functionals import build_functionals
    from dcascade.race import RaceParams, build_race_tables
    from dcascade.theory import self_profile

    race = build_race_tables(RaceParams())
    chain = ChainParams(
        max_depth=6, eps=0.2, phi=0.75, attribution_rule="floored", attribution_floor=0.5
    )
    profile = self_profile(build_functionals(race, chain, 1.0, 20.0), "AS")
    assert all(b >= a - 1e-12 for a, b in zip(profile.attributed, profile.attributed[1:]))


def test_the_kernel_families_hold_the_spectral_gap_fixed() -> None:
    """Both interpolations leave the rate of forgetting at 1 - eps."""
    for eps in (0.05, 0.2, 0.4):
        for family in ("mixed", "severity"):
            for t in (0.0, 0.25, 0.5, 0.75, 1.0):
                assert second_eigenvalue(eps, f"{family}:{t}") == pytest.approx(
                    1.0 - eps, abs=1e-7
                )


def test_the_kernel_families_move_the_drift_direction() -> None:
    assert drift_direction("mixed:0.0") == pytest.approx(drift_direction("ladder"))
    assert drift_direction("mixed:1.0") == pytest.approx(0.0)
    assert drift_direction("severity:1.0") == pytest.approx(drift_direction("collapse"))
    # the direction falls along one family and rises along the other
    mixed = [drift_direction(f"mixed:{t}") for t in (0.0, 0.5, 1.0)]
    severity = [drift_direction(f"severity:{t}") for t in (0.0, 0.5, 1.0)]
    assert mixed == sorted(mixed, reverse=True)
    assert severity == sorted(severity)
