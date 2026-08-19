"""Four ways to intervene on a delegation cascade, and what each one buys.

The instruments act at different points of the chain and are not substitutes:

``pass_through``
    raise the attribution retention ``phi`` towards one, so that harm is traced
    back through the intermediaries.  Acts on the *incentive* to delegate.

``depth_cap``
    forbid chains longer than a stated number of hand-offs.  Acts on the
    *length* of the chain and forfeits the organisational benefit of the layers
    it removes.

``attestation``
    reduce the per-hand-off erosion probability by a factor, for instance by
    requiring each layer to restate and sign the specification it received.
    Acts on the *fidelity* of every hand-off.

``audit``
    insert one specification check at a chosen layer.  Acts on fidelity too,
    but only once, and where it is placed decides how much it is worth.

Each instrument is applied to the same baseline and scored on the same four
numbers, so the comparison is like for like.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .chain import AttributionRule, AuditPlan, ChainParams
from .functionals import build_functionals
from .race import RaceTables
from .theory import Equilibrium, equilibrium


@dataclass(frozen=True)
class InterventionOutcome:
    """Long-run outcome of one instrument at one setting."""

    instrument: str
    setting: float
    unsafe_frequency: float
    mean_depth: float
    social_payoff: float
    declaration_gap: float
    equilibrium: Equilibrium


def _score(
    tables: RaceTables,
    chain: ChainParams,
    instrument: str,
    setting: float,
    lam: float,
    harm: float,
    method: str,
    **kwargs,
) -> InterventionOutcome:
    fun = build_functionals(tables, chain, lam, harm)
    eq = equilibrium(fun, method=method, **kwargs)
    return InterventionOutcome(
        instrument=instrument,
        setting=float(setting),
        unsafe_frequency=eq.unsafe_frequency,
        mean_depth=eq.mean_depth,
        social_payoff=eq.social_payoff,
        declaration_gap=eq.declaration_gap,
        equilibrium=eq,
    )


def pass_through_sweep(
    tables: RaceTables,
    chain: ChainParams,
    phis: np.ndarray,
    lam: float = 1.0,
    harm: float = 5.0,
    method: str = "sml",
    **kwargs,
) -> list[InterventionOutcome]:
    """Outcome as the attribution retention is raised towards pass-through."""
    return [
        _score(tables, replace(chain, phi=float(p)), "pass_through", p, lam, harm, method, **kwargs)
        for p in phis
    ]


def depth_cap_sweep(
    tables: RaceTables,
    chain: ChainParams,
    caps: np.ndarray,
    lam: float = 1.0,
    harm: float = 5.0,
    method: str = "sml",
    **kwargs,
) -> list[InterventionOutcome]:
    """Outcome as the permitted chain length is reduced."""
    return [
        _score(
            tables, replace(chain, max_depth=int(c)), "depth_cap", c, lam, harm, method, **kwargs
        )
        for c in caps
    ]


def attestation_sweep(
    tables: RaceTables,
    chain: ChainParams,
    factors: np.ndarray,
    lam: float = 1.0,
    harm: float = 5.0,
    method: str = "sml",
    **kwargs,
) -> list[InterventionOutcome]:
    """Outcome as every hand-off is made more faithful by a factor ``rho``.

    ``rho = 1`` leaves the baseline erosion probability untouched and
    ``rho = 0`` makes transmission perfect.
    """
    return [
        _score(
            tables,
            replace(chain, eps=float(r) * chain.eps),
            "attestation",
            r,
            lam,
            harm,
            method,
            **kwargs,
        )
        for r in factors
    ]


def liability_sweep(
    tables: RaceTables,
    chain: ChainParams,
    liabilities: np.ndarray,
    lam: float = 1.0,
    harm: float = 20.0,
    method: str = "sml",
    **kwargs,
) -> list[InterventionOutcome]:
    """Outcome as the effective liability ``L = lambda h`` is raised.

    The instrument moves the liability rate ``lambda``, not the harm ``h``: a
    regulator can raise a penalty above compensatory level, but it cannot make
    an Unsafe action less damaging by legislating.  Holding ``h`` fixed is what
    keeps the social payoff comparable along the sweep.
    """
    return [
        _score(tables, chain, "liability", float(L), float(L) / harm, harm, method, **kwargs)
        for L in liabilities
    ]


def audit_placement_sweep(
    tables: RaceTables,
    chain: ChainParams,
    strength: float = 1.0,
    lam: float = 1.0,
    harm: float = 5.0,
    method: str = "sml",
    **kwargs,
) -> list[InterventionOutcome]:
    """Outcome as one check of fixed strength is moved down the chain.

    A check at layer ``k`` restores the intent, which then has ``d - k``
    hand-offs left to survive; the residual erosion is therefore
    ``M ** (d - k)`` instead of ``M ** d``.  Since the erosion kernel only moves
    mass towards the unsafe end of the order, the residual harm is
    non-increasing in ``k``: the same amount of checking is worth more the
    closer it is placed to the agent that acts.
    """
    return [
        _score(
            tables,
            replace(chain, audit=AuditPlan(layer=int(k), strength=strength)),
            "audit_layer",
            k,
            lam,
            harm,
            method,
            **kwargs,
        )
        for k in range(chain.max_depth + 1)
    ]


def attribution_floor_sweep(
    tables: RaceTables,
    chain: ChainParams,
    floors: np.ndarray,
    lam: float = 1.0,
    harm: float = 5.0,
    method: str = "sml",
    **kwargs,
) -> list[InterventionOutcome]:
    """Outcome as a statutory floor is placed under per-layer attribution.

    The instrument leaves the per-hand-off retention ``phi`` untouched and only
    forbids attribution from falling below ``a_min``.  It is the regime a
    regulator can write without having to trace responsibility through every
    intermediary: however long the chain, a fixed share of the harm comes back
    to the principal.
    """
    return [
        _score(
            tables,
            replace(chain, attribution_rule="floored", attribution_floor=float(a)),
            "attribution_floor",
            a,
            lam,
            harm,
            method,
            **kwargs,
        )
        for a in floors
    ]


def attribution_regime_comparison(
    tables: RaceTables,
    chain: ChainParams,
    lam: float = 1.0,
    harm: float = 5.0,
    method: str = "sml",
    floors: tuple[float, ...] = (0.25, 0.5),
    supers: tuple[float, ...] = (1.1, 1.25),
    **kwargs,
) -> list[InterventionOutcome]:
    """The baseline against the attribution regimes a reader might propose.

    Geometric attenuation is the baseline.  ``strict`` is top-level strict
    liability, which is also the joint-and-several regime.  ``harmonic`` splits
    the blame equally between the principal and its agents, so attribution
    still vanishes in the depth but only polynomially.  The floors are partial
    restorations, and the super-attribution settings charge a principal *more*
    the deeper it delegates.
    """
    out = [
        _score(tables, chain, "geometric", chain.phi, lam, harm, method, **kwargs),
        _score(
            tables,
            replace(chain, attribution_rule="strict"),
            "strict",
            1.0,
            lam,
            harm,
            method,
            **kwargs,
        ),
        _score(
            tables,
            replace(chain, attribution_rule="harmonic"),
            "harmonic",
            1.0,
            lam,
            harm,
            method,
            **kwargs,
        ),
    ]
    out += [
        _score(
            tables,
            replace(chain, attribution_rule="floored", attribution_floor=float(a)),
            "floored",
            a,
            lam,
            harm,
            method,
            **kwargs,
        )
        for a in floors
    ]
    out += [
        _score(
            tables,
            replace(chain, phi=float(p)),
            "super",
            p,
            lam,
            harm,
            method,
            **kwargs,
        )
        for p in supers
    ]
    return out


def instrument_frontier(
    outcomes: list[InterventionOutcome],
) -> tuple[np.ndarray, np.ndarray]:
    """``(social payoff, unsafe frequency)`` pairs traced by one instrument."""
    return (
        np.array([o.social_payoff for o in outcomes]),
        np.array([o.unsafe_frequency for o in outcomes]),
    )


def matched_effort_comparison(
    tables: RaceTables,
    chain: ChainParams,
    target: float,
    lam: float = 1.0,
    harm: float = 5.0,
    method: str = "sml",
    grid: int = 21,
    **kwargs,
) -> dict[str, dict[str, float]]:
    """Cheapest setting of each instrument that reaches a target Unsafe frequency.

    Instruments are not comparable in their own units, so each is scored by the
    social payoff it delivers at the weakest setting that meets the target.  An
    instrument that cannot reach the target at any setting is reported as such.
    """
    sweeps = {
        "pass_through": pass_through_sweep(
            tables, chain, np.linspace(chain.phi, 1.0, grid), lam, harm, method, **kwargs
        ),
        "depth_cap": depth_cap_sweep(
            tables, chain, np.arange(chain.max_depth, -1, -1), lam, harm, method, **kwargs
        ),
        "attestation": attestation_sweep(
            tables, chain, np.linspace(1.0, 0.0, grid), lam, harm, method, **kwargs
        ),
        "liability": liability_sweep(
            tables, chain, np.linspace(lam * harm, 20.0 * lam * harm, grid), lam, harm,
            method, **kwargs
        ),  # lam is re-derived inside from L and h
    }
    out: dict[str, dict[str, float]] = {}
    for name, outcomes in sweeps.items():
        hit = next((o for o in outcomes if o.unsafe_frequency <= target), None)
        if hit is None:
            best = min(outcomes, key=lambda o: o.unsafe_frequency)
            out[name] = {
                "reaches_target": 0.0,
                "best_unsafe": best.unsafe_frequency,
                "setting": best.setting,
                "social_payoff": best.social_payoff,
                "mean_depth": best.mean_depth,
            }
        else:
            out[name] = {
                "reaches_target": 1.0,
                "best_unsafe": hit.unsafe_frequency,
                "setting": hit.setting,
                "social_payoff": hit.social_payoff,
                "mean_depth": hit.mean_depth,
            }
    return out
