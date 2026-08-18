"""Payoff functionals of the delegation layer.

A design is a pair ``(d, s)``: a delegation depth and the intent the principal
issues.  Two designs meeting in the race produce, in expectation over the
executed designs and over the horizon,

.. math::

    a(d, s; d', s') &= p_{d,s}^{\\top} A\\, p_{d',s'} + B(d), \\\\
    m(d, s; d', s') &= p_{d,s}^{\\top} M\\, p_{d',s'},

where ``p`` is the transmission law of the chain, ``A`` and ``M`` are the task
payoff and unsafe-action matrices of the interaction layer, and ``B(d)`` is the
organisational benefit of the chain.  Everything is bilinear because the
specification is transmitted once and then played for the whole race, so the
executed designs of the two seats are independent draws.

Three functionals are built on those primitives.

``pi_P`` (private / selection functional)
    what the principal receives.  It carries only the *attributed* share of the
    harm, ``phi ** d``, because responsibility that has passed through ``d``
    intermediaries is only partly traced back.

``pi_S`` (social functional)
    the same accounting with the realised harm fully counted, regardless of how
    many hands it passed through.  It never enters the dynamics; it is the
    yardstick.

Their difference is the *dilution wedge*

.. math:: \\Delta(d) = \\pi_P - \\pi_S = h\\,(1 - \\lambda\\phi^{d})\\, m,

which is non-decreasing in ``d`` for every ``phi <= 1``: depth alone widens the
gap between what a principal optimises and what society bears, without any
change in what the principal intends.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .chain import ChainParams, design_labels, design_space, transmission_stack
from .race import STRATEGIES, RaceTables


@dataclass(frozen=True)
class DelegationFunctionals:
    """Payoff, harm and behaviour matrices over the ``(depth, intent)`` space."""

    designs: tuple[tuple[int, str], ...]
    labels: tuple[str, ...]
    depth: np.ndarray
    """Delegation depth of each design, shape ``(n,)``."""

    intent: tuple[str, ...]
    """Intent of each design."""

    transmission: np.ndarray
    """Executed-design law of each design, shape ``(n, 4)``."""

    task: np.ndarray
    """``a``: race payoff of the focal seat plus its organisational benefit."""

    harm: np.ndarray
    """``m``: expected number of Unsafe actions of the focal seat."""

    unsafe_frequency: np.ndarray
    """``u``: expected fraction of Unsafe rounds of the focal seat."""

    pi_P: np.ndarray
    """Private functional; this is what drives the dynamics."""

    pi_S: np.ndarray
    """Social functional; this is what the outcome is scored against."""

    wedge: np.ndarray
    """``pi_P - pi_S``, the dilution wedge."""

    lam: float
    harm_scale: float
    chain: ChainParams

    @property
    def effective_liability(self) -> float:
        """``L = lambda * h``, the only combination entering the dynamics."""
        return self.lam * self.harm_scale

    @property
    def n(self) -> int:
        return len(self.labels)

    def index(self, depth: int, intent: str) -> int:
        """Position of design ``(depth, intent)`` in the design space."""
        return self.designs.index((depth, intent))

    def depth_block(self, depth: int) -> np.ndarray:
        """Indices of every design at a given depth."""
        return np.flatnonzero(self.depth == depth)

    def intent_block(self, intent: str) -> np.ndarray:
        """Indices of every design carrying a given intent."""
        return np.array([i for i, s in enumerate(self.intent) if s == intent])


def build_functionals(
    tables: RaceTables,
    chain: ChainParams,
    lam: float = 1.0,
    harm: float = 5.0,
) -> DelegationFunctionals:
    """Assemble the delegation-layer functionals.

    Parameters
    ----------
    tables:
        Output of :func:`dcascade.race.build_race_tables`, in the erosion order.
    chain:
        Delegation-layer parameters.
    lam:
        Liability rate at depth zero.  ``lam = 1`` charges a principal who acts
        for itself exactly the harm it causes; ``lam < 1`` is incomplete
        compensation and ``lam > 1`` is a punitive multiplier.  It scales only
        the private functional, never the social one, so the social payoff is
        comparable across liability regimes.
    harm:
        External harm of one Unsafe action, in task-payoff units.  This is a
        property of the world, not of the legal regime, and is held fixed when
        the liability instrument is varied.
    """
    if tables.strategies != STRATEGIES:
        raise ValueError("the race tables must use the erosion order of STRATEGIES")
    if lam < 0.0:
        raise ValueError(f"lam must be non-negative, got {lam}")
    if harm < 0.0:
        raise ValueError(f"harm must be non-negative, got {harm}")

    designs = design_space(chain)
    labels = design_labels(chain)
    stack = transmission_stack(chain.max_depth, chain.eps, chain.kernel, chain.audit)

    p = np.stack([stack[d][STRATEGIES.index(s)] for d, s in designs])
    depth = np.array([d for d, _ in designs], dtype=int)
    intent = tuple(s for _, s in designs)

    task = p @ tables.payoff @ p.T
    task = task + np.array([chain.organisational_benefit(d) for d in depth])[:, None]
    harm_matrix = p @ tables.unsafe_count @ p.T
    unsafe = p @ tables.unsafe_frequency @ p.T

    attributed = np.array([chain.attribution(d) for d in depth])[:, None]
    pi_P = task - lam * harm * attributed * harm_matrix
    pi_S = task - harm * harm_matrix

    return DelegationFunctionals(
        designs=designs,
        labels=labels,
        depth=depth,
        intent=intent,
        transmission=p,
        task=task,
        harm=harm_matrix,
        unsafe_frequency=unsafe,
        pi_P=pi_P,
        pi_S=pi_S,
        wedge=pi_P - pi_S,
        lam=float(lam),
        harm_scale=float(harm),
        chain=chain,
    )


def aggregate_unsafe_frequency(x: np.ndarray, unsafe_frequency: np.ndarray) -> float:
    """Population-level Unsafe frequency ``U(x) = sum_ij x_i x_j u(i, j)``.

    This is the primary welfare-relevant observable of the model.  It needs no
    welfare weights, so the reported harm measure does not depend on the
    (necessarily arbitrary) choice of ``h``.
    """
    x = np.asarray(x, dtype=float)
    return float(x @ np.asarray(unsafe_frequency, dtype=float) @ x)


def mean_depth(x: np.ndarray, depth: np.ndarray) -> float:
    """Population-average delegation depth."""
    return float(np.asarray(x, dtype=float) @ np.asarray(depth, dtype=float))


def mean_social_payoff(x: np.ndarray, pi_S: np.ndarray) -> float:
    """Population-average social payoff."""
    x = np.asarray(x, dtype=float)
    return float(x @ pi_S @ x)


def intent_distribution(x: np.ndarray, intent: tuple[str, ...]) -> dict[str, float]:
    """Marginal distribution of the issued intent."""
    x = np.asarray(x, dtype=float)
    return {s: float(x[[i for i, t in enumerate(intent) if t == s]].sum()) for s in STRATEGIES}


def depth_distribution(x: np.ndarray, depth: np.ndarray) -> dict[int, float]:
    """Marginal distribution of the delegation depth."""
    x = np.asarray(x, dtype=float)
    return {int(d): float(x[depth == d].sum()) for d in np.unique(depth)}


def executed_distribution(x: np.ndarray, transmission: np.ndarray) -> np.ndarray:
    """Marginal distribution of the design that is actually executed."""
    x = np.asarray(x, dtype=float)
    return x @ np.asarray(transmission, dtype=float)


def declaration_gap(x: np.ndarray, intent: tuple[str, ...], transmission: np.ndarray) -> float:
    """Distance between what the population declares and what it executes.

    Total-variation distance between the marginal intent distribution and the
    marginal executed-design distribution.  It is the size of the discrepancy an
    outside observer would find between stated and revealed safety policy.
    """
    declared = np.array([intent_distribution(x, intent)[s] for s in STRATEGIES])
    executed = executed_distribution(x, transmission)
    return float(0.5 * np.abs(declared - executed).sum())
