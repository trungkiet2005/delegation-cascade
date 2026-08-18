"""The delegation chain: how a specification survives a hand-off, and how far.

A seat in the race is operated by a chain

    principal -> agent_1 -> ... -> agent_d -> the race,

where ``d`` is the *delegation depth*.  The principal issues one specification,
its *intent*, and every hand-off transmits that specification imperfectly.  The
design that is finally executed is therefore a random variable whose law is
determined by the intent and by the depth alone.

Erosion order
-------------
The four reduced designs are ordered by how many safety clauses their
specification carries::

    AS  ->  CS  ->  CAS  ->  AU
    2       1       0.5      0     clauses retained (informally)

``AS`` says "be safe whatever the other side does"; dropping the unconditional
clause gives ``CS``, "be safe but match the other side"; dropping the safe
opening gives ``CAS``, "match the other side, and open unsafe"; dropping the
matching clause gives ``AU``, "be unsafe".  Each hand-off can drop a clause but
never adds one back, so ``AU`` is absorbing.  This is what makes delegation
depth a *directional* channel rather than symmetric noise: the sub-agent that
paraphrases its instructions keeps the part that describes the task and loses
the part that qualifies it.

Hand-off kernels
----------------
One hand-off acts by the row-stochastic kernel

.. math:: M = (1 - \\varepsilon) I + \\varepsilon D,

with ``eps`` the per-hand-off erosion probability and ``D`` a drift kernel.
Three drift kernels are provided:

``ladder`` (default)
    one clause is lost per erosion event; ``D`` shifts the design one rung
    down the erosion order.  Then the executed design of a depth-``d`` chain
    is the intent shifted by ``Binomial(d, eps)`` rungs, truncated at ``AU``.

``collapse``
    an erosion event discards the specification entirely and the sub-agent
    falls back on the locally competitive design ``AU``.

``uniform``
    an erosion event replaces the specification by a uniformly random design.
    This is the *unbiased* control: it has the same spectral gap as ``ladder``
    but no direction, and it isolates how much of the depth effect is due to
    the bias of drift rather than to noise as such.

Because ``M`` is a stochastic matrix with a single recurrent class, the
influence of the intent on the executed design decays geometrically in depth
at the rate given by the second-largest eigenvalue modulus of ``M``.  The
resulting *specification half-life* is the number of hand-offs after which
half of the intent has been lost.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .race import STRATEGIES

KernelName = Literal["ladder", "collapse", "uniform"]

#: Number of designs in the erosion order.
N_DESIGNS = len(STRATEGIES)


def drift_matrix(kernel: KernelName = "ladder") -> np.ndarray:
    """The drift kernel ``D`` of one erosion event.

    Row = current specification, column = specification after the event, in the
    erosion order of :data:`dcascade.race.STRATEGIES`.
    """
    n = N_DESIGNS
    if kernel == "ladder":
        d = np.zeros((n, n))
        for i in range(n):
            d[i, min(i + 1, n - 1)] = 1.0
        return d
    if kernel == "collapse":
        d = np.zeros((n, n))
        d[:, n - 1] = 1.0
        return d
    if kernel == "uniform":
        return np.full((n, n), 1.0 / n)
    raise ValueError(f"unknown drift kernel {kernel!r}")


def handoff_kernel(eps: float, kernel: KernelName = "ladder") -> np.ndarray:
    """One hand-off: ``M = (1 - eps) I + eps D``."""
    if not 0.0 <= eps <= 1.0:
        raise ValueError(f"eps must lie in [0, 1], got {eps}")
    return (1.0 - eps) * np.eye(N_DESIGNS) + eps * drift_matrix(kernel)


@dataclass(frozen=True)
class AuditPlan:
    """A single specification check inserted into the chain.

    Attributes
    ----------
    layer:
        Position of the check, counted in hand-offs from the principal.  A
        check at ``layer = k`` inspects the specification held by agent ``k``.
        ``layer = 0`` is a check at the principal itself and is vacuous;
        ``layer = d`` checks the agent that actually acts.
    strength:
        Probability that the check detects a corrupted specification and
        restores the principal's intent.
    """

    layer: int
    strength: float = 1.0

    def __post_init__(self) -> None:
        if self.layer < 0:
            raise ValueError("audit layer must be non-negative")
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("audit strength must lie in [0, 1]")


def transmission(
    depth: int,
    eps: float,
    kernel: KernelName = "ladder",
    audit: AuditPlan | None = None,
) -> np.ndarray:
    """Law of the executed design, row = intent, column = executed design.

    Without an audit this is ``M ** depth``.  A check of strength ``alpha`` at
    layer ``k <= depth`` restores the intent with probability ``alpha``, after
    which the specification still has ``depth - k`` hand-offs to survive, so
    the law is

    .. math:: \\alpha\\, M^{d-k} + (1 - \\alpha)\\, M^{d}.

    A check placed beyond the end of a chain never fires and leaves the law
    unchanged.
    """
    if depth < 0:
        raise ValueError("depth must be non-negative")
    m = handoff_kernel(eps, kernel)
    full = np.linalg.matrix_power(m, depth)
    if audit is None or audit.layer > depth or audit.strength == 0.0:
        return full
    residual = np.linalg.matrix_power(m, depth - audit.layer)
    return audit.strength * residual + (1.0 - audit.strength) * full


def transmission_stack(
    max_depth: int,
    eps: float,
    kernel: KernelName = "ladder",
    audit: AuditPlan | None = None,
) -> np.ndarray:
    """Transmission laws for every depth ``0 .. max_depth``, shape ``(D+1, n, n)``."""
    return np.stack(
        [transmission(d, eps, kernel, audit) for d in range(max_depth + 1)]
    )


# --------------------------------------------------------------------------
# how fast the specification is forgotten
# --------------------------------------------------------------------------


def second_eigenvalue(eps: float, kernel: KernelName = "ladder") -> float:
    """Second-largest eigenvalue modulus of the hand-off kernel.

    The whole dependence of the executed design on the intent is carried by
    the non-unit spectrum of ``M``, so this number sets the rate at which
    delegation forgets what it was told.
    """
    eigenvalues = np.linalg.eigvals(handoff_kernel(eps, kernel))
    moduli = np.sort(np.abs(eigenvalues))[::-1]
    return float(moduli[1])


def specification_half_life(eps: float, kernel: KernelName = "ladder") -> float:
    """Hand-offs after which half of the intent has been lost.

    ``d_half = ln 2 / ln(1 / |lambda_2|)``.  For the ladder and collapse
    kernels ``|lambda_2| = 1 - eps``, so the half-life depends on nothing but
    the per-hand-off erosion probability.
    """
    lam2 = second_eigenvalue(eps, kernel)
    if lam2 <= 0.0:
        return 0.0
    if lam2 >= 1.0:
        return float("inf")
    return float(np.log(2.0) / np.log(1.0 / lam2))


def intent_influence(
    depth: int, eps: float, kernel: KernelName = "ladder", audit: AuditPlan | None = None
) -> float:
    """Total-variation spread of the executed design over the possible intents.

    ``max_{s, s'} || P_d(s, .) - P_d(s', .) ||_TV``.  This is the *entire*
    behavioural leverage an instruction has at depth ``d``: if it is zero, two
    principals issuing opposite instructions produce the same ecosystem.
    """
    p = transmission(depth, eps, kernel, audit)
    n = p.shape[0]
    spread = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            spread = max(spread, 0.5 * float(np.abs(p[i] - p[j]).sum()))
    return spread


def intent_separation(
    depth: int, eps: float, kernel: KernelName = "ladder", audit: AuditPlan | None = None
) -> float:
    """How much of one specification clause survives ``depth`` hand-offs.

    Total-variation distance between the executed laws of two intents that
    differ by a single clause, averaged over the adjacent pairs of the erosion
    order.  For the ladder kernel this is exactly ``(1 - eps) ** depth`` in the
    absence of truncation at the absorbing design, so the surviving fraction of
    an instruction is geometric in the number of hand-offs.
    """
    p = transmission(depth, eps, kernel, audit)
    n = p.shape[0]
    return float(
        np.mean([0.5 * np.abs(p[i] - p[i + 1]).sum() for i in range(n - 1)])
    )


def fidelity(depth: int, eps: float, kernel: KernelName = "ladder") -> np.ndarray:
    """Probability that each intent is executed as issued, per intent."""
    return np.diag(transmission(depth, eps, kernel)).copy()


# --------------------------------------------------------------------------
# the design space
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ChainParams:
    """Parameters of the delegation layer.

    Attributes
    ----------
    max_depth:
        Largest delegation depth available.  A depth cap is the model of a
        rule that forbids chains longer than ``max_depth`` hand-offs.
    eps:
        Per-hand-off erosion probability.
    phi:
        Per-layer attribution retention.  A principal at depth ``d`` is held
        responsible for the fraction ``phi ** d`` of the harm its chain causes.
        ``phi = 1`` is strict pass-through liability; ``phi < 1`` is per-layer
        liability, under which each intermediary absorbs part of the blame.
    gain:
        Net organisational benefit of one hand-off, in payoff units.  It is a
        *real* benefit: delegation genuinely produces more, so the term enters
        the social functional as well as the private one.
    curvature:
        Coefficient of the quadratic coordination cost ``curvature * d ** 2``,
        the classical loss-of-control cost of a taller hierarchy.  Zero by
        default, so that in the baseline the only force limiting depth is the
        harm the principal internalises.
    kernel:
        Drift kernel of one erosion event.
    audit:
        Optional specification check inserted into every chain.
    """

    max_depth: int = 4
    eps: float = 0.15
    phi: float = 0.6
    gain: float = 1.5
    curvature: float = 0.0
    kernel: KernelName = "ladder"
    audit: AuditPlan | None = None

    def __post_init__(self) -> None:
        if self.max_depth < 0:
            raise ValueError("max_depth must be non-negative")
        if not 0.0 <= self.phi <= 1.0:
            raise ValueError("phi must lie in [0, 1]")

    @property
    def depths(self) -> tuple[int, ...]:
        return tuple(range(self.max_depth + 1))

    def organisational_benefit(self, depth: int) -> float:
        """Net benefit of operating a chain of depth ``d``."""
        return self.gain * depth - self.curvature * depth * depth

    def attribution(self, depth: int) -> float:
        """Fraction of the realised harm attributed to the principal."""
        return float(self.phi**depth)


def design_space(chain: ChainParams) -> tuple[tuple[int, str], ...]:
    """Every ``(depth, intent)`` design, depth-major."""
    return tuple((d, s) for d in chain.depths for s in STRATEGIES)


def design_labels(chain: ChainParams) -> tuple[str, ...]:
    """Compact labels ``intent@depth`` for the designs of the space."""
    return tuple(f"{s}@{d}" for d, s in design_space(chain))
