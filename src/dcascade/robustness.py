"""Robustness of the depth results across the free parameters of the model.

Everything reported in the manuscript is recomputed here over the interaction
parameters (prize, private risk, setback accounting), the delegation parameters
(drift kernel, organisational benefit, coordination cost, depth ceiling) and the
parameters of the evolutionary process (population size, selection intensity,
and the choice between the finite-population and the replicator reading).

The quantity carried through every sweep is the 2x2 of
:func:`dcascade.theory.mechanism_decomposition`: the long-run Unsafe frequency
with each mechanism switched off and on, and the interaction between them.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .chain import ChainParams
from .dynamics import stationary_analysis, stationary_analysis_sml
from .functionals import build_functionals
from .race import STRATEGIES, RaceParams, build_race_tables
from .theory import (
    behavioural_leverage,
    clause_value,
    deniability_threshold,
    equilibrium,
    mechanism_decomposition,
    self_profile,
)


@dataclass(frozen=True)
class Cell:
    """One point of a robustness sweep."""

    label: str
    setting: str
    baseline: float
    drift_only: float
    deniability_only: float
    both: float
    interaction: float
    interaction_share: float
    mean_depth_passthrough: float
    mean_depth_per_layer: float

    def as_row(self) -> dict[str, object]:
        return {
            "sweep": self.label,
            "setting": self.setting,
            "U_baseline": self.baseline,
            "U_drift_only": self.drift_only,
            "U_deniability_only": self.deniability_only,
            "U_both": self.both,
            "interaction": self.interaction,
            "interaction_share": self.interaction_share,
            "depth_passthrough": self.mean_depth_passthrough,
            "depth_per_layer": self.mean_depth_per_layer,
        }


def _cell(
    label: str,
    setting: str,
    tables,
    chain: ChainParams,
    lam: float,
    harm: float,
    method: str,
    **kwargs,
) -> Cell:
    d = mechanism_decomposition(tables, chain, lam, harm, method=method, **kwargs)
    return Cell(
        label=label,
        setting=setting,
        baseline=d.baseline.unsafe_frequency,
        drift_only=d.drift_only.unsafe_frequency,
        deniability_only=d.deniability_only.unsafe_frequency,
        both=d.both.unsafe_frequency,
        interaction=d.interaction,
        interaction_share=d.interaction_share,
        mean_depth_passthrough=d.drift_only.mean_depth,
        mean_depth_per_layer=d.both.mean_depth,
    )


def sweep_interaction_layer(
    chain: ChainParams,
    lam: float = 1.0,
    harm: float = 20.0,
    prizes: tuple[float, ...] = (50.0, 100.0, 200.0),
    risks: tuple[float, ...] = (0.1, 0.3, 0.6, 0.9),
    method: str = "sml",
    **kwargs,
) -> list[Cell]:
    """The 2x2 across the prize and the private setback risk of the race."""
    out = []
    for prize in prizes:
        for p_max in risks:
            tables = build_race_tables(RaceParams(prize=prize, p_max=p_max))
            out.append(
                _cell(
                    "interaction_layer",
                    f"prize={prize:g}, p_max={p_max:g}",
                    tables,
                    chain,
                    lam,
                    harm,
                    method,
                    **kwargs,
                )
            )
    return out


def sweep_setback_scope(
    chain: ChainParams, lam: float = 1.0, harm: float = 20.0, method: str = "sml", **kwargs
) -> list[Cell]:
    """The 2x2 under both readings of what a setback destroys."""
    return [
        _cell(
            "setback_scope",
            scope,
            build_race_tables(RaceParams(setback_scope=scope)),
            chain,
            lam,
            harm,
            method,
            **kwargs,
        )
        for scope in ("total", "prize")
    ]


def sweep_kernel(
    tables, chain: ChainParams, lam: float = 1.0, harm: float = 20.0, method: str = "sml", **kwargs
) -> list[Cell]:
    """The 2x2 for each drift kernel, including the unbiased control.

    The ``uniform`` kernel has the same spectral gap as ``ladder`` but no
    direction.  It is the control that separates *biased* erosion from noise:
    if the depth effect survives it, the effect is about transmission loss as
    such; if it does not, the effect is about which way the specification slips.
    """
    return [
        _cell(
            "kernel",
            kernel,
            tables,
            replace(chain, kernel=kernel),
            lam,
            harm,
            method,
            **kwargs,
        )
        for kernel in ("ladder", "collapse", "uniform")
    ]


def sweep_organisation(
    tables,
    chain: ChainParams,
    lam: float = 1.0,
    harm: float = 20.0,
    gains: tuple[float, ...] = (0.0, 0.5, 1.5, 3.0, 5.0),
    curvatures: tuple[float, ...] = (0.0, 0.1, 0.25),
    method: str = "sml",
    **kwargs,
) -> list[Cell]:
    """The 2x2 across the benefit and the coordination cost of a taller chain."""
    out = []
    for gain in gains:
        out.append(
            _cell(
                "gain", f"g={gain:g}", tables, replace(chain, gain=gain), lam, harm, method, **kwargs
            )
        )
    for curvature in curvatures:
        out.append(
            _cell(
                "curvature",
                f"c={curvature:g}",
                tables,
                replace(chain, curvature=curvature),
                lam,
                harm,
                method,
                **kwargs,
            )
        )
    return out


def sweep_process(
    tables,
    chain: ChainParams,
    lam: float = 1.0,
    harm: float = 20.0,
    sizes: tuple[int, ...] = (50, 100, 200),
    betas: tuple[float, ...] = (0.01, 0.02, 0.05, 0.1, 0.2),
    depths: tuple[int, ...] = (4, 6, 8),
) -> list[Cell]:
    """The 2x2 across the parameters of the evolutionary process."""
    out = []
    for z in sizes:
        out.append(
            _cell("population", f"Z={z}", tables, chain, lam, harm, "sml", population_size=z)
        )
    for beta in betas:
        out.append(
            _cell("selection", f"beta={beta:g}", tables, chain, lam, harm, "sml", beta=beta)
        )
    for depth in depths:
        out.append(
            _cell(
                "depth_ceiling",
                f"D={depth}",
                tables,
                replace(chain, max_depth=depth),
                lam,
                harm,
                "sml",
            )
        )
    out.append(_cell("dynamics", "replicator", tables, chain, lam, harm, "replicator"))
    out.append(_cell("dynamics", "sml", tables, chain, lam, harm, "sml"))
    return out


def sweep_liability(
    tables,
    chain: ChainParams,
    liabilities: np.ndarray,
    harm: float = 20.0,
    method: str = "sml",
    **kwargs,
) -> list[Cell]:
    """The 2x2 across the level of the liability instrument itself.

    The liability rate moves; the harm of an Unsafe action does not.
    """
    return [
        _cell("liability", f"L={L:g}", tables, chain, float(L) / harm, harm, method, **kwargs)
        for L in liabilities
    ]


# --------------------------------------------------------------------------
# checks of the reductions used elsewhere
# --------------------------------------------------------------------------


def sml_versus_full_chain(
    tables,
    chain: ChainParams,
    intent: str = "AS",
    lam: float = 1.0,
    harm: float = 20.0,
    population_size: int = 40,
    beta: float = 0.05,
    mutations: tuple[float, ...] = (0.05, 0.02, 0.01, 0.005),
    depths: tuple[int, ...] | None = None,
) -> list[dict[str, float]]:
    """Compare the small-mutation limit with the full mutation-selection chain.

    The full chain has ``C(Z + n - 1, n - 1)`` states and is only tractable on a
    small design space, so the intent is held fixed and the comparison runs over
    a subset of the depth coordinate.  The two readings should agree as the
    mutation rate falls.
    """
    if depths is None:
        depths = (0, chain.max_depth // 2, chain.max_depth)
    fun = build_functionals(tables, chain, lam, harm)
    idx = np.array([fun.index(d, intent) for d in depths])
    payoff = np.ascontiguousarray(fun.pi_P[np.ix_(idx, idx)])
    unsafe = np.ascontiguousarray(fun.unsafe_frequency[np.ix_(idx, idx)])

    depth_vector = np.array(depths, dtype=float)
    sml = stationary_analysis_sml(payoff, unsafe, population_size, beta)
    rows = [
        {
            "mu": 0.0,
            "unsafe_frequency": sml.unsafe_frequency,
            "mean_depth": float(sml.strategy_frequencies @ depth_vector),
        }
    ]
    for mu in mutations:
        full = stationary_analysis(payoff, unsafe, population_size, beta, mu)
        rows.append(
            {
                "mu": float(mu),
                "unsafe_frequency": full.unsafe_frequency,
                "mean_depth": float(full.strategy_frequencies @ depth_vector),
            }
        )
    return rows


def deniability_threshold_profile(
    tables, chain: ChainParams, intent: str, probe_depth: int = 40, lam: float = 1.0, harm: float = 20.0
) -> dict[str, object]:
    """Locate the depth from which extra hand-offs lower the attributed harm.

    The design space of the manuscript stops at a finite ceiling, but the
    statement that ``phi*(d) -> 1`` is about the limit, so the harm profile is
    probed well beyond the ceiling here.  The probe uses a chain with the same
    erosion parameters and a much larger depth range; nothing about the
    evolutionary dynamics enters.
    """
    probe = replace(chain, max_depth=probe_depth)
    fun = build_functionals(tables, probe, lam, harm)
    profile = self_profile(fun, intent)
    thresholds = np.array(
        [deniability_threshold(profile, d) or 0.0 for d in range(probe_depth)]
    )
    crossing = np.flatnonzero(thresholds > chain.phi)
    return {
        "intent": intent,
        "phi": chain.phi,
        "thresholds": thresholds,
        "harm": profile.harm,
        "attributed": profile.attributed,
        "first_sheltering_depth": int(crossing[0]) if crossing.size else None,
        "limit_threshold": float(thresholds[-1]),
    }


def leverage_decay(
    tables, chain: ChainParams, probe_depth: int = 40, lam: float = 1.0, harm: float = 20.0
) -> dict[str, np.ndarray]:
    """Behavioural leverage of the instruction as a function of depth."""
    probe = replace(chain, max_depth=probe_depth)
    fun = build_functionals(tables, probe, lam, harm)
    return {
        "depths": np.arange(probe_depth + 1),
        "leverage": behavioural_leverage(fun),
        "clause_value": clause_value(fun),
        "reference": (1.0 - chain.eps) ** np.arange(probe_depth + 1),
    }


def equilibrium_grid(
    tables,
    chain: ChainParams,
    phis: np.ndarray,
    epsilons: np.ndarray,
    lam: float = 1.0,
    harm: float = 20.0,
    method: str = "sml",
    **kwargs,
) -> dict[str, np.ndarray]:
    """Long-run outcome over the attribution / erosion plane."""
    unsafe = np.zeros((phis.size, epsilons.size))
    depth = np.zeros_like(unsafe)
    gap = np.zeros_like(unsafe)
    social = np.zeros_like(unsafe)
    for i, phi in enumerate(phis):
        for j, eps in enumerate(epsilons):
            fun = build_functionals(
                tables, replace(chain, phi=float(phi), eps=float(eps)), lam, harm
            )
            eq = equilibrium(fun, method=method, **kwargs)
            unsafe[i, j] = eq.unsafe_frequency
            depth[i, j] = eq.mean_depth
            gap[i, j] = eq.declaration_gap
            social[i, j] = eq.social_payoff
    interaction = (
        unsafe
        - unsafe[-1, :][None, :]
        - unsafe[:, 0][:, None]
        + unsafe[-1, 0]
    )
    return {
        "phis": phis,
        "epsilons": epsilons,
        "unsafe": unsafe,
        "depth": depth,
        "declaration_gap": gap,
        "social": social,
        "interaction": interaction,
    }


def executed_mix_by_depth(
    tables, chain: ChainParams, intent: str = "AS", lam: float = 1.0, harm: float = 20.0
) -> np.ndarray:
    """Executed-design mixture of one intent at every depth, shape ``(D+1, 4)``."""
    fun = build_functionals(tables, chain, lam, harm)
    rows = [fun.index(d, intent) for d in chain.depths]
    return fun.transmission[rows]


def strategy_labels() -> tuple[str, ...]:
    return STRATEGIES
