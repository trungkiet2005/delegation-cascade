"""Closed-form results and the structural quantities of the delegation model.

The three quantities that organise everything are

``d_half``
    the *specification half-life*, the number of hand-offs after which half of
    the principal's intent has been lost;

``phi*(d)``
    the *deniability threshold* at depth ``d``: the attribution retention below
    which adding one more hand-off strictly *reduces* the harm attributed to
    the principal even though it strictly increases the harm caused;

``L*(d)``
    the effective liability at which the depth-``(d+1)`` design stops out-
    competing the depth-``d`` design, which is finite and positive only above
    the deniability threshold.

Everything below is computed from the exact matrices of
:mod:`dcascade.functionals`; nothing is estimated by simulation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .chain import ChainParams, specification_half_life
from .dynamics import average_replicator_attractor, stationary_analysis_sml
from .functionals import (
    DelegationFunctionals,
    aggregate_unsafe_frequency,
    build_functionals,
    declaration_gap,
    depth_distribution,
    executed_distribution,
    intent_distribution,
    mean_social_payoff,
    process_unsafe_frequency,
)
from .race import STRATEGIES, RaceTables, action_paths


# --------------------------------------------------------------------------
# the erosion order is a harm order
# --------------------------------------------------------------------------


def erosion_monotone(tables: RaceTables, tol: float = 1e-12) -> bool:
    """Whether harm is non-decreasing along the erosion order, column by column.

    This is the structural fact behind every depth result: dropping a safety
    clause never lowers the number of Unsafe actions of the focal seat, whatever
    the opponent does.  When it holds, a hand-off kernel that only moves mass
    down the erosion order makes the realised harm of a chain non-decreasing in
    its depth, for every intent and every opponent.

    The property is a theorem rather than a numerical finding (Lemma 1 of the
    paper): against a fixed opponent the focal *action path* itself is
    non-decreasing along the erosion order, round by round, so the ordering
    survives every horizon law and every payoff parameter.  This function is
    therefore a regression check on the assembled matrices, not evidence for
    the claim.  :func:`erosion_monotone_pathwise` checks the stronger statement
    the proof actually uses.
    """
    return bool(np.all(np.diff(tables.unsafe_count, axis=0) >= -tol))


def erosion_monotone_frequency(tables: RaceTables, tol: float = 1e-12) -> bool:
    """The same monotonicity for the Unsafe frequency."""
    return bool(np.all(np.diff(tables.unsafe_frequency, axis=0) >= -tol))


def erosion_monotone_pathwise(horizons: tuple[int, ...] = (1, 2, 3, 7, 40, 401)) -> bool:
    """Whether the erosion order dominates the focal action path round by round.

    The pathwise statement of Lemma 1.  It involves no payoff parameter at all,
    so it is checked on the action paths directly.
    """
    for col in STRATEGIES:
        for lo, hi in zip(STRATEGIES, STRATEGIES[1:]):
            for n in horizons:
                a_lo, _ = action_paths(lo, col, n)
                a_hi, _ = action_paths(hi, col, n)
                if np.any(a_hi < a_lo):
                    return False
    return True


# --------------------------------------------------------------------------
# depth profiles against a fixed environment
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DepthProfile:
    """Depth-indexed quantities of one intent against a fixed resident design."""

    intent: str
    resident: tuple[int, str]
    depths: np.ndarray
    harm: np.ndarray
    """Harm actually caused, ``m(d)``."""

    attributed: np.ndarray
    """Harm attributed to the principal, ``phi ** d * m(d)``."""

    task: np.ndarray
    """Task payoff including the organisational benefit."""

    private: np.ndarray
    """Private payoff ``pi_P``."""

    social: np.ndarray
    """Social payoff ``pi_S``."""


def depth_profile(
    fun: DelegationFunctionals, intent: str, resident: tuple[int, str] | None = None
) -> DepthProfile:
    """Depth profile of one intent played against a fixed resident design."""
    if resident is None:
        resident = (0, "AS")
    j = fun.index(*resident)
    depths = np.array(fun.chain.depths)
    rows = [fun.index(d, intent) for d in depths]
    harm = fun.harm[rows, j]
    attributed = np.array([fun.chain.attribution(d) for d in depths]) * harm
    return DepthProfile(
        intent=intent,
        resident=resident,
        depths=depths,
        harm=harm,
        attributed=fun.lam * attributed,
        task=fun.task[rows, j],
        private=fun.pi_P[rows, j],
        social=fun.pi_S[rows, j],
    )


def self_profile(fun: DelegationFunctionals, intent: str) -> DepthProfile:
    """Depth profile of one intent against a resident population of itself.

    This is the profile that decides which depth is an equilibrium, because a
    monomorphic population of depth-``d`` principals is exactly the environment
    a depth-``(d+1)`` mutant faces.
    """
    depths = np.array(fun.chain.depths)
    rows = [fun.index(d, intent) for d in depths]
    harm = np.array([fun.harm[i, i] for i in rows])
    attributed = np.array([fun.chain.attribution(d) for d in depths]) * harm
    return DepthProfile(
        intent=intent,
        resident=(-1, intent),
        depths=depths,
        harm=harm,
        attributed=fun.lam * attributed,
        task=np.array([fun.task[i, i] for i in rows]),
        private=np.array([fun.pi_P[i, i] for i in rows]),
        social=np.array([fun.pi_S[i, i] for i in rows]),
    )


# --------------------------------------------------------------------------
# the liability shelter
# --------------------------------------------------------------------------


def deniability_threshold(profile: DepthProfile, depth: int) -> float | None:
    """Attribution retention below which a hand-off lowers the attributed harm.

    Adding a layer changes the attributed harm from ``phi ** d m(d)`` to
    ``phi ** (d+1) m(d+1)``, so it *lowers* it exactly when

    .. math:: \\phi < \\phi^{*}(d) = m(d) / m(d+1).

    Because the realised harm saturates as the specification is forgotten,
    ``phi*(d)`` rises towards one with depth: any attribution retention below
    one is eventually below the threshold, and from that depth on every extra
    hand-off is a strict reduction in attributed harm.  Returns ``None`` when
    the chain causes no harm at ``d + 1`` and the comparison is vacuous.
    """
    m_d = float(profile.harm[depth])
    m_next = float(profile.harm[depth + 1])
    if m_next <= 0.0:
        return None
    return m_d / m_next


def shelter_depth(profile: DepthProfile, phi: float) -> int | None:
    """First depth from which every further hand-off lowers the attributed harm.

    ``None`` if no such depth exists within the available range.
    """
    for d in range(len(profile.depths) - 1):
        thresholds = [
            deniability_threshold(profile, k) for k in range(d, len(profile.depths) - 1)
        ]
        if all(t is not None and phi < t for t in thresholds):
            return d
    return None


@dataclass(frozen=True)
class ShelterSummary:
    """How far the attributed harm falls below the realised harm."""

    intent: str
    max_depth: int
    realised_harm: float
    attributed_harm: float
    attributed_at_zero_depth: float
    shelter_factor: float
    """Realised harm divided by attributed harm at the deepest chain."""

    first_sheltering_depth: int | None


def shelter_summary(profile: DepthProfile, phi: float) -> ShelterSummary:
    """Summarise the liability shelter opened by the deepest available chain."""
    realised = float(profile.harm[-1])
    attributed = float(profile.attributed[-1])
    factor = float("inf") if attributed <= 0.0 else realised / attributed
    return ShelterSummary(
        intent=profile.intent,
        max_depth=int(profile.depths[-1]),
        realised_harm=realised,
        attributed_harm=attributed,
        attributed_at_zero_depth=float(profile.attributed[0]),
        shelter_factor=factor,
        first_sheltering_depth=shelter_depth(profile, phi),
    )


# --------------------------------------------------------------------------
# invasion of one depth by the next
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DepthInvasion:
    """Whether one more hand-off pays, in a resident population of depth ``d``."""

    intent: str
    depth: int
    benefit: float
    """Task-payoff gain of the extra layer, ``a(d+1, d) - a(d, d)``."""

    attributed_gain: float
    """Extra attributed harm per unit of effective liability."""

    critical_liability: float | None
    """``L`` at which the advantage vanishes; ``None`` if it never does."""

    direction: str
    """``"deeper_below"``, ``"deeper_above"`` or ``"liability_independent"``."""

    def deeper_invades_at(self, effective_liability: float) -> bool:
        """Whether the deeper design has a selective advantage at ``L``."""
        if self.direction == "liability_independent":
            return self.benefit > 0.0
        assert self.critical_liability is not None
        if self.direction == "deeper_below":
            return effective_liability < self.critical_liability
        return effective_liability > self.critical_liability


def depth_invasion(fun: DelegationFunctionals, intent: str, depth: int) -> DepthInvasion:
    """Closed-form condition for depth ``d + 1`` to invade a depth-``d`` resident.

    The invader's advantage is affine in the effective liability ``L``,

    .. math::
        \\pi_P(d{+}1, d) - \\pi_P(d, d)
        = \\underbrace{a(d{+}1, d) - a(d, d)}_{\\text{benefit}}
        - L\\,\\underbrace{\\big[\\phi^{d+1} m(d{+}1, d) - \\phi^{d} m(d, d)\\big]}
                          {}_{\\text{attributed gain}},

    so a single ratio decides it.  The sign of the attributed gain is the whole
    point: below the deniability threshold it is negative, liability *adds* to
    the advantage of the deeper design, and no level of liability can stop it.
    """
    i = fun.index(depth + 1, intent)
    j = fun.index(depth, intent)
    benefit = float(fun.task[i, j] - fun.task[j, j])
    attributed = float(
        fun.chain.attribution(depth + 1) * fun.harm[i, j]
        - fun.chain.attribution(depth) * fun.harm[j, j]
    )
    if abs(attributed) < 1e-14:
        return DepthInvasion(
            intent, depth, benefit, attributed, None, "liability_independent"
        )
    critical = benefit / attributed
    direction = "deeper_below" if attributed > 0 else "deeper_above"
    return DepthInvasion(intent, depth, benefit, attributed, critical, direction)


# --------------------------------------------------------------------------
# what an instruction is still worth at depth d
# --------------------------------------------------------------------------


def behavioural_leverage(
    fun: DelegationFunctionals, resident: tuple[int, str] | None = None
) -> np.ndarray:
    """Spread in realised Unsafe frequency across intents, by depth.

    ``max_{s, s'} |u(d, s) - u(d, s')|`` against a fixed environment.  This is
    what a safety instruction is still worth at depth ``d``: when it reaches
    zero, two principals issuing opposite instructions run the same ecosystem.
    """
    if resident is None:
        resident = (0, "AS")
    j = fun.index(*resident)
    out = []
    for d in fun.chain.depths:
        values = np.array([fun.unsafe_frequency[fun.index(d, s), j] for s in STRATEGIES])
        out.append(float(values.max() - values.min()))
    return np.array(out)


def clause_value(
    fun: DelegationFunctionals, resident: tuple[int, str] | None = None
) -> np.ndarray:
    """What one extra safety clause in the instruction is worth, by depth.

    The reduction in realised Unsafe frequency obtained by issuing ``AS``
    instead of ``CS`` -- one more clause in the specification -- against a fixed
    environment.  For the ladder kernel the two intents differ by a single rung,
    so the difference between their executed laws decays geometrically at the
    rate ``1 - eps`` and this is the quantity that decays with it.
    """
    if resident is None:
        resident = (0, "AS")
    j = fun.index(*resident)
    return np.array(
        [
            float(
                fun.unsafe_frequency[fun.index(d, "CS"), j]
                - fun.unsafe_frequency[fun.index(d, "AS"), j]
            )
            for d in fun.chain.depths
        ]
    )


def selection_leverage(
    fun: DelegationFunctionals, resident: tuple[int, str] | None = None
) -> np.ndarray:
    """Spread in private payoff across intents, by depth.

    The counterpart of :func:`behavioural_leverage` on the selection side: it is
    the payoff difference that selection has to work with when it tries to tell
    a safe declaration from an unsafe one.
    """
    if resident is None:
        resident = (0, "AS")
    j = fun.index(*resident)
    out = []
    for d in fun.chain.depths:
        values = np.array([fun.pi_P[fun.index(d, s), j] for s in STRATEGIES])
        out.append(float(values.max() - values.min()))
    return np.array(out)


def neutrality_depth(
    fun: DelegationFunctionals,
    population_size: int = 100,
    beta: float = 0.1,
    resident: tuple[int, str] | None = None,
) -> int | None:
    """First depth at which the declared intent is selectively neutral.

    In a population of size ``Z`` with selection intensity ``beta``, a payoff
    difference is invisible to selection once ``beta * delta << 1 / Z``: the
    fixation probability of the mutant is then within a vanishing distance of
    the neutral value ``1 / Z``.  Beyond this depth the intent coordinate drifts
    at random, so a declaration of safety carries no information about the
    ecosystem that produced it.
    """
    leverage = selection_leverage(fun, resident)
    scale = 1.0 / (beta * population_size)
    for d, value in zip(fun.chain.depths, leverage):
        if value < scale:
            return int(d)
    return None


# --------------------------------------------------------------------------
# equilibrium summaries
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Equilibrium:
    """Long-run outcome of one parameterisation."""

    frequencies: np.ndarray
    unsafe_frequency: float
    independent_unsafe_frequency: float
    mean_depth: float
    depth_distribution: dict[int, float]
    intent_distribution: dict[str, float]
    executed_distribution: np.ndarray
    declaration_gap: float
    social_payoff: float
    method: str


def equilibrium(
    fun: DelegationFunctionals,
    method: str = "sml",
    population_size: int = 100,
    beta: float = 0.1,
    n_starts: int = 120,
    seed: int = 20260818,
    precision_digits: int | None = None,
) -> Equilibrium:
    """Long-run design distribution under the private functional.

    ``method="sml"`` uses the small-mutation limit of the finite-population
    process; ``method="replicator"`` uses the basin-averaged attractor of the
    replicator flow.  The two answer different questions -- which design the
    process spends its time in, and which mixtures are stable -- and are
    reported side by side throughout.
    """
    if method == "sml":
        x = stationary_analysis_sml(
            fun.pi_P, fun.unsafe_frequency, population_size, beta, precision_digits
        ).strategy_frequencies
    elif method == "replicator":
        x = average_replicator_attractor(fun.pi_P, n_starts=n_starts, seed=seed)
    else:
        raise ValueError(f"unknown method {method!r}")

    independent_unsafe = aggregate_unsafe_frequency(x, fun.unsafe_frequency)
    process_unsafe = process_unsafe_frequency(x, fun.unsafe_frequency)
    return Equilibrium(
        frequencies=x,
        unsafe_frequency=process_unsafe if method == "sml" else independent_unsafe,
        independent_unsafe_frequency=independent_unsafe,
        mean_depth=float(x @ fun.depth),
        depth_distribution=depth_distribution(x, fun.depth),
        intent_distribution=intent_distribution(x, fun.intent),
        executed_distribution=executed_distribution(x, fun.transmission),
        declaration_gap=declaration_gap(x, fun.intent, fun.transmission),
        social_payoff=mean_social_payoff(x, fun.pi_S),
        method=method,
    )


def socially_optimal_depth(fun: DelegationFunctionals, intent: str = "AS") -> int:
    """Depth maximising the social payoff of a monomorphic population."""
    profile = self_profile(fun, intent)
    return int(profile.depths[int(np.argmax(profile.social))])


# --------------------------------------------------------------------------
# separating the two mechanisms
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MechanismDecomposition:
    """The 2x2 design that separates specification drift from deniability."""

    baseline: Equilibrium
    drift_only: Equilibrium
    deniability_only: Equilibrium
    both: Equilibrium
    interaction: float
    """``U(both) - U(drift) - U(deniability) + U(baseline)``."""

    drift_effect: float
    deniability_effect: float
    total_effect: float

    @property
    def interaction_share(self) -> float:
        """Share of the total increase that neither mechanism produces alone."""
        if abs(self.total_effect) < 1e-15:
            return 0.0
        return self.interaction / self.total_effect


def mechanism_decomposition(
    tables: RaceTables,
    chain: ChainParams,
    lam: float = 1.0,
    harm: float = 5.0,
    method: str = "sml",
    **kwargs,
) -> MechanismDecomposition:
    """Run the 2x2: drift on/off crossed with per-layer attribution on/off.

    The off cells are ``eps = 0`` (a specification that survives every hand-off)
    and ``phi = 1`` (strict pass-through liability).  ``chain`` supplies the on
    values.
    """
    from dataclasses import replace

    def cell(eps: float, phi: float) -> Equilibrium:
        return equilibrium(
            build_functionals(tables, replace(chain, eps=eps, phi=phi), lam, harm),
            method=method,
            **kwargs,
        )

    baseline = cell(0.0, 1.0)
    drift = cell(chain.eps, 1.0)
    deniability = cell(0.0, chain.phi)
    both = cell(chain.eps, chain.phi)

    u0 = baseline.unsafe_frequency
    ud = drift.unsafe_frequency
    up = deniability.unsafe_frequency
    ub = both.unsafe_frequency
    return MechanismDecomposition(
        baseline=baseline,
        drift_only=drift,
        deniability_only=deniability,
        both=both,
        interaction=ub - ud - up + u0,
        drift_effect=ud - u0,
        deniability_effect=up - u0,
        total_effect=ub - u0,
    )


def frozen_depth_counterfactual(
    tables: RaceTables,
    chain: ChainParams,
    lam: float = 1.0,
    harm: float = 5.0,
    method: str = "sml",
    **kwargs,
) -> dict[str, float]:
    """Split the joint effect into a mechanical part and a response part.

    Specification drift raises harm in two ways: mechanically, at whatever
    chain depth the population happens to run, and through the depth the
    population then chooses to run.  Holding the depth distribution at its
    pass-through value while switching the attribution on isolates the second
    channel, which is the response that per-layer liability disables.
    """
    from dataclasses import replace

    passthrough = build_functionals(tables, replace(chain, phi=1.0), lam, harm)
    full = build_functionals(tables, chain, lam, harm)
    eq_pass = equilibrium(passthrough, method=method, **kwargs)
    eq_full = equilibrium(full, method=method, **kwargs)

    # the same design frequencies, scored with the same behaviour matrix, isolate
    # how much of the difference is composition rather than behaviour
    frozen = (
        process_unsafe_frequency(eq_pass.frequencies, full.unsafe_frequency)
        if method == "sml"
        else aggregate_unsafe_frequency(eq_pass.frequencies, full.unsafe_frequency)
    )
    return {
        "unsafe_passthrough": eq_pass.unsafe_frequency,
        "unsafe_per_layer": eq_full.unsafe_frequency,
        "unsafe_frozen_composition": frozen,
        "composition_channel": eq_full.unsafe_frequency - frozen,
        "mean_depth_passthrough": eq_pass.mean_depth,
        "mean_depth_per_layer": eq_full.mean_depth,
    }


def half_life(chain: ChainParams) -> float:
    """Specification half-life of the chain, in hand-offs."""
    return specification_half_life(chain.eps, chain.kernel)


# --------------------------------------------------------------------------
# the depth-zero benchmark, and the depth at which a rule set there fails
# --------------------------------------------------------------------------


def invasion_threshold_depth_zero(
    tables: RaceTables, invader: str, resident: str
) -> float | None:
    r"""Effective liability at which a rare invader loses its advantage at depth zero.

    With no delegation the private functional is ``a - L m``, so the advantage
    of a rare invader is affine in ``L`` and the critical value is the ratio

    .. math:: L^{*} = \frac{a(i, j) - a(j, j)}{m(i, j) - m(j, j)}.

    ``None`` when the invader causes no excess harm and the comparison does not
    depend on liability at all.
    """
    i = tables.strategies.index(invader)
    j = tables.strategies.index(resident)
    harm_gain = float(tables.unsafe_count[i, j] - tables.unsafe_count[j, j])
    if abs(harm_gain) < 1e-12:
        return None
    return float(tables.payoff[i, j] - tables.payoff[j, j]) / harm_gain


def critical_liability(
    tables: RaceTables,
    method: str = "sml",
    tolerance: float = 1e-3,
    bounds: tuple[float, float] = (0.0, 200.0),
    iterations: int = 60,
    **kwargs,
) -> float:
    """Smallest effective liability at which the depth-zero race is safe.

    "Safe" means a long-run Unsafe frequency below ``tolerance``.  The quantity
    is monotone enough in ``L`` for a bisection, and it is the natural reference
    point for the delegation results: it is the liability a regulator would set
    if it looked only at principals who act for themselves.
    """
    chain = ChainParams(max_depth=0, eps=0.0, phi=1.0, gain=0.0)

    def unsafe(L: float) -> float:
        fun = build_functionals(tables, chain, 1.0, L)
        return equilibrium(fun, method=method, **kwargs).unsafe_frequency

    low, high = bounds
    if unsafe(high) > tolerance:
        return float("inf")
    for _ in range(iterations):
        mid = 0.5 * (low + high)
        if unsafe(mid) > tolerance:
            low = mid
        else:
            high = mid
    return float(high)


def attribution_failure_depth(
    effective_liability: float, critical: float, phi: float
) -> float:
    r"""Depth at which a liability calibrated at depth zero stops clearing the bar.

    The liability actually borne by a principal at depth ``d`` is ``L phi ** d``,
    so a rule that clears a threshold ``L_c`` at depth zero stops clearing it at

    .. math:: d^{\dagger} = \log(L_c / L) / \log \phi ,

    which is finite for every ``phi < 1``: no level of liability set at the top
    of a chain survives an unbounded number of hand-offs.
    """
    if phi >= 1.0:
        return float("inf")
    if effective_liability <= critical:
        return 0.0
    return float(np.log(critical / effective_liability) / np.log(phi))
