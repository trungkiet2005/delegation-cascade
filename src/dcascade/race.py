"""Exact evaluation of the reduced AI-race game.

The interaction layer reproduces the repeated two-player race of Fernandez
Domingos and Han (2026): in every round each player chooses Safe (``S``) or
Unsafe (``U``); Unsafe pays more in the stage game and advances the race
faster, but accumulates private setback risk for a race winner.

This module is deliberately *unmodified* with respect to the single-layer
version of the game: the whole contribution of the present study sits in the
delegation layer built on top of it (:mod:`dcascade.chain`), where a seat is
operated by a chain of delegated agents and the design that is executed need
not be the design the principal intended.  Keeping the interaction layer fixed
is what makes the depth effects attributable to delegation alone.

The four reduced designs (AS, CS, CAS, AU) are deterministic, so the whole
action path of an ordered pair is fixed once the pair is fixed.  The only
stochastic primitive is the horizon ``T``.  Every matchup is therefore
evaluated by exact expectation over the horizon distribution rather than by
Monte Carlo sampling, which removes simulation noise from the payoff matrix
entirely.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

SAFE = 0
UNSAFE = 1

#: Designs in the *erosion order*: each successive design carries one fewer
#: safety clause than the one before it (see :mod:`dcascade.chain`).
STRATEGIES: tuple[str, ...] = ("AS", "CS", "CAS", "AU")
STRATEGY_INDEX: dict[str, int] = {s: i for i, s in enumerate(STRATEGIES)}

#: First-round action and conditionality of each reduced design,
#: ``(first_action, copies_opponent)``.
_STRATEGY_SPEC: dict[str, tuple[int, bool]] = {
    "AS": (SAFE, False),
    "CS": (SAFE, True),
    "CAS": (UNSAFE, True),
    "AU": (UNSAFE, False),
}


@dataclass(frozen=True)
class RaceParams:
    """Parameters of the repeated race.

    Defaults reproduce the experimental protocol of the source study.

    Attributes
    ----------
    stage_payoffs:
        Row = own action, column = opponent action, ordered ``(S, U)``.
    step_safe, step_unsafe:
        Race progress contributed by one Safe / Unsafe round.
    min_rounds:
        Number of guaranteed rounds before the stochastic stopping rule starts.
    stop_prob:
        Per-round stopping probability after ``min_rounds``.
    prize:
        Prize awarded to the progress leader; split evenly on a tie.
    p_max:
        Treatment-level maximum private setback risk.
    setback_scope:
        ``"total"`` removes accumulated stage payoffs *and* prize from an
        affected player (the reading used here); ``"prize"`` removes only the
        prize.  Reported results are checked against both.
    max_rounds:
        Truncation of the horizon distribution; the residual mass is
        ``(1 - stop_prob) ** (max_rounds - min_rounds)``.
    """

    stage_payoffs: np.ndarray = field(
        default_factory=lambda: np.array([[1.0, 0.6], [2.4, 2.0]])
    )
    step_safe: float = 1.0
    step_unsafe: float = 1.5
    min_rounds: int = 5
    stop_prob: float = 0.2
    prize: float = 100.0
    p_max: float = 0.6
    setback_scope: Literal["total", "prize"] = "total"
    max_rounds: int = 400

    def horizon_distribution(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(T_values, probabilities)`` of the truncated horizon law.

        ``T = min_rounds + G`` with ``G`` geometric on ``{0, 1, 2, ...}``.  The
        truncated tail mass is assigned to ``max_rounds`` so that the
        probabilities sum to one exactly.
        """
        t_values = np.arange(self.min_rounds, self.max_rounds + 1)
        g = t_values - self.min_rounds
        probs = self.stop_prob * (1.0 - self.stop_prob) ** g
        probs[-1] = max(0.0, 1.0 - probs[:-1].sum())
        return t_values, probs

    @property
    def expected_horizon(self) -> float:
        t_values, probs = self.horizon_distribution()
        return float(t_values @ probs)


def action_paths(row: str, col: str, n_rounds: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic action sequences of ``row`` against ``col``.

    Returns two integer arrays of length ``n_rounds`` holding the actions of
    the focal (row) player and of the opponent (column) player.
    """
    a_row = np.empty(n_rounds, dtype=np.int8)
    a_col = np.empty(n_rounds, dtype=np.int8)
    first_row, cond_row = _STRATEGY_SPEC[row]
    first_col, cond_col = _STRATEGY_SPEC[col]

    a_row[0], a_col[0] = first_row, first_col
    for t in range(1, n_rounds):
        a_row[t] = a_col[t - 1] if cond_row else first_row
        a_col[t] = a_row[t - 1] if cond_col else first_col
    return a_row, a_col


@dataclass(frozen=True)
class MatchupOutcome:
    """Horizon-averaged quantities for one ordered pair of executed designs."""

    payoff: float
    """Expected task payoff of the focal seat."""

    unsafe_count: float
    """Expected number of Unsafe actions of the focal seat."""

    unsafe_frequency: float
    """Expected fraction of Unsafe rounds of the focal seat."""

    win_probability: float
    """Probability that the focal seat strictly leads at the horizon."""

    tie_probability: float
    """Probability of equal progress at the horizon."""

    setback_probability: float
    """Expected realised setback probability."""


def evaluate_matchup(row: str, col: str, params: RaceParams) -> MatchupOutcome:
    """Evaluate one ordered pair of executed designs exactly over the horizon law."""
    t_values, probs = params.horizon_distribution()
    n = int(t_values[-1])
    a_row, a_col = action_paths(row, col, n)

    stage = params.stage_payoffs[a_row, a_col]
    cum_stage = np.cumsum(stage)

    steps = np.where(a_row == UNSAFE, params.step_unsafe, params.step_safe)
    steps_opp = np.where(a_col == UNSAFE, params.step_unsafe, params.step_safe)
    progress = np.cumsum(steps)
    progress_opp = np.cumsum(steps_opp)

    n_unsafe = np.cumsum(a_row == UNSAFE)

    idx = t_values - 1  # cumulative arrays are 0-indexed
    horizon = t_values.astype(float)

    stage_total = cum_stage[idx]
    unsafe_count = n_unsafe[idx].astype(float)
    unsafe_freq = unsafe_count / horizon
    q = params.p_max * unsafe_freq

    wins = progress[idx] > progress_opp[idx]
    ties = progress[idx] == progress_opp[idx]
    at_risk = wins | ties
    prize = np.where(wins, params.prize, 0.0) + np.where(ties, params.prize / 2.0, 0.0)

    if params.setback_scope == "total":
        kept = stage_total + prize
        payoff_t = np.where(at_risk, (1.0 - q) * kept, stage_total)
    else:
        payoff_t = stage_total + np.where(at_risk, (1.0 - q) * prize, prize)

    return MatchupOutcome(
        payoff=float(payoff_t @ probs),
        unsafe_count=float(unsafe_count @ probs),
        unsafe_frequency=float(unsafe_freq @ probs),
        win_probability=float(wins.astype(float) @ probs),
        tie_probability=float(ties.astype(float) @ probs),
        setback_probability=float((at_risk * q) @ probs),
    )


@dataclass(frozen=True)
class RaceTables:
    """Matrices of horizon-averaged matchup quantities.

    Every matrix is indexed ``[i, j]`` = focal design ``i`` against opponent
    design ``j``, with the design order of :data:`STRATEGIES`.
    """

    strategies: tuple[str, ...]
    payoff: np.ndarray
    unsafe_count: np.ndarray
    unsafe_frequency: np.ndarray
    win_probability: np.ndarray
    tie_probability: np.ndarray
    setback_probability: np.ndarray
    params: RaceParams


def build_race_tables(
    params: RaceParams, strategies: tuple[str, ...] = STRATEGIES
) -> RaceTables:
    """Evaluate every ordered pair of executed designs."""
    n = len(strategies)
    payoff = np.zeros((n, n))
    unsafe_count = np.zeros((n, n))
    unsafe_freq = np.zeros((n, n))
    win = np.zeros((n, n))
    tie = np.zeros((n, n))
    setback = np.zeros((n, n))

    for i, row in enumerate(strategies):
        for j, col in enumerate(strategies):
            out = evaluate_matchup(row, col, params)
            payoff[i, j] = out.payoff
            unsafe_count[i, j] = out.unsafe_count
            unsafe_freq[i, j] = out.unsafe_frequency
            win[i, j] = out.win_probability
            tie[i, j] = out.tie_probability
            setback[i, j] = out.setback_probability

    return RaceTables(
        strategies=tuple(strategies),
        payoff=payoff,
        unsafe_count=unsafe_count,
        unsafe_frequency=unsafe_freq,
        win_probability=win,
        tie_probability=tie,
        setback_probability=setback,
        params=params,
    )
