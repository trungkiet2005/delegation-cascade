"""The baseline parameterisation used everywhere in the manuscript.

Every script, figure and test reads its defaults from here, so a change of
baseline propagates to the whole study and cannot leave one figure behind.

The values are chosen as follows.

``RACE``
    the protocol of the source experiment, unmodified.

``HARM = 20``
    about three times the effective liability at which the *depth-zero* race is
    already safe (``dcascade.theory.critical_liability`` returns 6.52 for this
    game).  The delegation results are therefore not about a race that was
    unsafe to begin with: at depth zero this liability level leaves a long-run
    Unsafe frequency of zero.

``PHI = 0.75``
    a quarter of the responsibility is lost at each hand-off.  Values from
    0.6 to 1 are swept.

``EPS = 0.20``
    one hand-off in five drops a safety clause, a specification half-life of
    3.1 hand-offs.  Values from 0 to 0.4 are swept.

``GAIN = 1.5``
    the net organisational benefit of one hand-off, about 2.5 per cent of the
    payoff of a safe seat.  Values from 0 to 5 are swept, including the ablation
    ``GAIN = 0`` in which delegation has no benefit at all and depth can only be
    selected as a liability shelter.

``MAX_DEPTH = 6``
    six hand-offs, giving a design space of 28 designs.  Ceilings of 4 and 8 are
    checked.
"""

from __future__ import annotations

from .chain import ChainParams
from .race import RaceParams

#: Interaction layer, exactly as in the source experiment.
RACE = RaceParams()

#: Delegation layer.
CHAIN = ChainParams(max_depth=6, eps=0.20, phi=0.75, gain=1.5, curvature=0.0)

#: Liability pass-through at depth zero, and the harm of one Unsafe action.
LAM = 1.0
HARM = 20.0

#: Finite population used for the stationary analysis.
POPULATION = 100
BETA = 0.05

#: Number of interior starts for the basin-averaged replicator attractor.
REPLICATOR_STARTS = 200
SEED = 20260818


def effective_liability() -> float:
    """``L = lambda h``, the only combination that enters the dynamics."""
    return LAM * HARM
