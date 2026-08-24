"""Numerical-method benchmarks for the AMC manuscript.

Everything reported in the "Numerical method" section of ``main.tex`` is
produced here.  Nothing in this script changes the model; it measures the cost,
the accuracy and the numerical stability of the scheme used to evaluate it.

    python paper-amc/scripts/run_numerics.py

Writes ``paper-amc/numerics_generated.tex`` and
``paper-amc/numerics_key_numbers.json``.
"""

from __future__ import annotations

import json
import math
import platform
import time
from dataclasses import replace
from pathlib import Path

import mpmath as mp
import numpy as np
import scipy
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph

from dcascade import config
from dcascade.chain import ChainParams
from dcascade.dynamics import (
    fixation_probability,
    sml_transition_matrix,
    sparse_stationary_distribution,
    stationary_analysis,
    stationary_analysis_sml,
)
from dcascade.functionals import build_functionals
from dcascade.race import STRATEGIES, RaceParams, action_paths, build_race_tables
from dcascade.robustness import sml_versus_full_chain

from egttools.analytical import PairwiseComparison
from egttools.games import Matrix2PlayerGameHolder

OUT = Path(__file__).resolve().parents[1]
RESULTS: dict = {}
TABLES = build_race_tables(config.RACE)


def _functionals(chain: ChainParams):
    return build_functionals(TABLES, chain, lam=config.LAM, harm=config.HARM)


def _chain(max_depth: int) -> ChainParams:
    return ChainParams(
        max_depth=max_depth,
        eps=config.CHAIN.eps,
        phi=config.CHAIN.phi,
        gain=config.CHAIN.gain,
        curvature=config.CHAIN.curvature,
    )


# --------------------------------------------------------------------------
# 1. size of the state space the reduction avoids
# --------------------------------------------------------------------------


def state_space_sizes() -> None:
    rows = []
    for max_depth in (4, 6, 8):
        n = 4 * (max_depth + 1)
        for Z in (50, 100, 200):
            rows.append(
                {
                    "max_depth": max_depth,
                    "n_designs": n,
                    "Z": Z,
                    "n_states": math.comb(Z + n - 1, n - 1),
                    "sml_fixations": n * (n - 1),
                    "sml_flops": n * (n - 1) * Z,
                }
            )
    RESULTS["state_space"] = rows


# --------------------------------------------------------------------------
# 2. exact horizon expectation against Monte Carlo
# --------------------------------------------------------------------------


def _simulate_matchup(
    row: str, col: str, params: RaceParams, n: int, rng
) -> tuple[float, float]:
    """Monte-Carlo estimate of (task payoff, expected Unsafe count) of ``row``."""
    horizons = params.min_rounds + rng.geometric(params.stop_prob, size=n) - 1
    horizons = np.minimum(horizons, params.max_rounds)
    longest = int(horizons.max())
    a_row, a_col = action_paths(row, col, longest)

    stage = params.stage_payoffs[a_row, a_col]
    cum_stage = np.concatenate([[0.0], np.cumsum(stage)])
    steps = np.where(a_row == 1, params.step_unsafe, params.step_safe)
    steps_opp = np.where(a_col == 1, params.step_unsafe, params.step_safe)
    prog = np.concatenate([[0.0], np.cumsum(steps)])
    prog_opp = np.concatenate([[0.0], np.cumsum(steps_opp)])
    unsafe = np.concatenate([[0.0], np.cumsum(a_row == 1)])

    t = horizons
    own, opp = prog[t], prog_opp[t]
    base = cum_stage[t]
    n_unsafe = unsafe[t]

    wins = own > opp
    ties = own == opp
    at_risk = wins | ties
    prize = np.where(wins, params.prize, 0.0) + np.where(ties, params.prize / 2.0, 0.0)

    # only a seat that leads or ties at the horizon carries the setback risk
    risk = np.clip(params.p_max * n_unsafe / np.maximum(t, 1), 0.0, 1.0)
    hit = at_risk & (rng.random(n) < risk)
    if params.setback_scope == "total":
        total = np.where(hit, 0.0, base + prize)
    else:
        total = base + np.where(hit, 0.0, prize)
    return float(total.mean()), float(n_unsafe.mean())


def monte_carlo_comparison(replicates: int = 8) -> None:
    """Sampling error of the payoff matrix, averaged over independent replicates."""
    params = config.RACE
    rng = np.random.default_rng(20260820)
    rows = []
    for n in (10**3, 10**4, 10**5, 10**6):
        sq_a, sq_m, worst_a = [], [], 0.0
        for _ in range(replicates):
            for i, row in enumerate(STRATEGIES):
                for j, col in enumerate(STRATEGIES):
                    a_hat, m_hat = _simulate_matchup(row, col, params, n, rng)
                    da = a_hat - TABLES.payoff[i, j]
                    sq_a.append(da * da)
                    sq_m.append((m_hat - TABLES.unsafe_count[i, j]) ** 2)
                    worst_a = max(worst_a, abs(da))
        rows.append(
            {
                "n_episodes": n,
                "replicates": replicates,
                "rms_error_a": float(np.sqrt(np.mean(sq_a))),
                "rms_error_m": float(np.sqrt(np.mean(sq_m))),
                "max_abs_error_a": worst_a,
            }
        )
    RESULTS["monte_carlo"] = rows


# --------------------------------------------------------------------------
# 3. numerical stability of the fixation sum
# --------------------------------------------------------------------------


def _exponents(a: np.ndarray, invader: int, resident: int, Z: int, beta: float):
    k = np.arange(1, Z, dtype=float)
    pi_inv = ((k - 1.0) * a[invader, invader] + (Z - k) * a[invader, resident]) / (
        Z - 1.0
    )
    pi_res = (k * a[resident, invader] + (Z - k - 1.0) * a[resident, resident]) / (
        Z - 1.0
    )
    return -beta * np.cumsum(pi_inv - pi_res)


def _fixation_naive(
    a: np.ndarray, invader: int, resident: int, Z: int, beta: float
) -> float:
    """The same formula evaluated without factoring out the largest exponent."""
    with np.errstate(over="ignore"):
        return float(1.0 / (1.0 + np.exp(_exponents(a, invader, resident, Z, beta)).sum()))


def _fixation_exact(
    a: np.ndarray, invader: int, resident: int, Z: int, beta: float, digits: int = 60
) -> "mp.mpf":
    """The same sum evaluated in ``digits``-digit arithmetic, as the reference."""
    with mp.workdps(digits):
        exponent = _exponents(a, invader, resident, Z, beta)
        total = mp.mpf(1)
        for e in exponent:
            total += mp.e ** mp.mpf(float(e))
        return 1 / total


def stability() -> None:
    """Overflow behaviour of the fixation sum against a 60-digit reference.

    The naive evaluation of~\\eqref{eq:fixation} exponentiates a partial sum
    whose magnitude is $O(\\beta Z \\max|\\pi_I - \\pi_R|)$; the stabilised
    evaluation factors out the largest exponent first.  Both are compared with
    the same sum evaluated in 60-digit arithmetic.
    """
    fun = _functionals(config.CHAIN)
    a = np.ascontiguousarray(fun.pi_P)
    n = a.shape[0]
    Z = config.POPULATION
    tiny = float(np.finfo(float).tiny)

    out = {}
    for beta in (config.BETA, 0.2):
        worst_exponent = 0.0
        n_naive_lost = 0
        n_recovered = 0
        smallest_recovered = np.inf
        max_rel_stable = 0.0
        n_representable = 0
        for r in range(n):
            for i in range(n):
                if i == r:
                    continue
                worst_exponent = max(
                    worst_exponent, float(np.abs(_exponents(a, i, r, Z, beta)).max())
                )
                stable = fixation_probability(a, i, r, Z, beta)
                naive = _fixation_naive(a, i, r, Z, beta)
                ref = _fixation_exact(a, i, r, Z, beta)
                if (not np.isfinite(naive)) or naive == 0.0:
                    n_naive_lost += 1
                    if stable > 0.0:
                        n_recovered += 1
                        smallest_recovered = min(smallest_recovered, stable)
                if ref > tiny:  # representable as a normal double
                    n_representable += 1
                    max_rel_stable = max(
                        max_rel_stable, float(abs(mp.mpf(stable) - ref) / ref)
                    )
        out["beta=%g" % beta] = {
            "worst_abs_exponent": worst_exponent,
            "n_naive_nonfinite_or_zero": n_naive_lost,
            "n_recovered_by_stabilised_form": n_recovered,
            "smallest_recovered_value": (
                None if not np.isfinite(smallest_recovered) else smallest_recovered
            ),
            "n_pairs_representable": n_representable,
            "max_relative_error_stabilised_vs_60_digits": max_rel_stable,
        }

    RESULTS["stability"] = {
        "n_designs": n,
        "n_ordered_pairs": n * (n - 1),
        "population_size": Z,
        "reference_digits": 60,
        "double_overflow_threshold": float(np.log(np.finfo(float).max)),
        "by_beta": out,
    }

    # Positivity preserves connectivity, but this checks the stationary
    # quantities themselves against the arbitrary-precision reference.
    stationary_checks = []
    unsafe = np.ascontiguousarray(fun.unsafe_frequency)
    for beta in (config.BETA, 0.2):
        fast = stationary_analysis_sml(a, unsafe, Z, beta)
        reference = stationary_analysis_sml(a, unsafe, Z, beta, precision_digits=60)
        stationary_checks.append(
            {
                "beta": beta,
                "total_variation": float(
                    0.5 * np.abs(fast.strategy_frequencies - reference.strategy_frequencies).sum()
                ),
                "max_absolute_difference": float(
                    np.abs(fast.strategy_frequencies - reference.strategy_frequencies).max()
                ),
                "unsafe_fast": fast.unsafe_frequency,
                "unsafe_reference": reference.unsafe_frequency,
                "independent_unsafe_fast": fast.independent_unsafe_frequency,
                "independent_unsafe_reference": reference.independent_unsafe_frequency,
            }
        )
    RESULTS["stationary_end_to_end"] = stationary_checks


# --------------------------------------------------------------------------
# 4. cross-check of the closed form against the general-purpose routine
# --------------------------------------------------------------------------


def cross_check() -> None:
    """Closed-form fixation probability against EGTtools, where it is feasible."""
    rows = []
    fun = _functionals(_chain(0))
    a = np.ascontiguousarray(fun.pi_P)
    n = a.shape[0]
    for Z in (50, 100, 200):
        for beta in (config.BETA, 0.2):
            game = Matrix2PlayerGameHolder(n, a)  # must outlive the evolver
            ev = PairwiseComparison(Z, game)
            worst = 0.0
            n_truncated = 0
            for i in range(n):
                for r in range(n):
                    if i == r:
                        continue
                    ours = fixation_probability(a, i, r, Z, beta)
                    theirs = float(ev.calculate_fixation_probability(i, r, beta))
                    if theirs == 0.0 and ours > 0.0:
                        n_truncated += 1
                    worst = max(worst, abs(ours - theirs))
            rows.append(
                {
                    "n_designs": n,
                    "Z": Z,
                    "beta": beta,
                    "max_abs_difference": worst,
                    "n_truncated_to_zero_by_egttools": n_truncated,
                }
            )
            del ev, game
    RESULTS["cross_check"] = rows


# --------------------------------------------------------------------------
# 5. small-mutation limit against the full mutation-selection chain
# --------------------------------------------------------------------------


def sml_versus_full() -> None:
    """The reduction against the full chain on a design space small enough for it."""
    chain = config.CHAIN
    depths = (0, chain.max_depth // 2, chain.max_depth)
    Z = 40
    mutations = (0.05, 0.02, 0.01, 0.005, 0.002, 0.001)
    rows = sml_versus_full_chain(
        TABLES,
        chain,
        intent="AS",
        lam=config.LAM,
        harm=config.HARM,
        population_size=Z,
        beta=config.BETA,
        mutations=mutations,
        depths=depths,
    )
    fun = _functionals(chain)
    idx = np.array([fun.index(d, "AS") for d in depths])
    payoff = np.ascontiguousarray(fun.pi_P[np.ix_(idx, idx)])
    unsafe = np.ascontiguousarray(fun.unsafe_frequency[np.ix_(idx, idx)])
    sml = stationary_analysis_sml(payoff, unsafe, population_size=Z, beta=config.BETA)

    RESULTS["sml_versus_full"] = {
        "n_designs": len(depths),
        "depths": list(depths),
        "Z": Z,
        "n_states": math.comb(Z + len(depths) - 1, len(depths) - 1),
        "sml_unsafe": sml.unsafe_frequency,
        "rows": [
            {
                "mu": r["mu"],
                "full_chain_unsafe": r["unsafe_frequency"],
                "abs_difference": abs(r["unsafe_frequency"] - sml.unsafe_frequency),
            }
            for r in rows
        ],
    }

    # The two single-mechanism cells of Table 4 are themselves only 0.018 and
    # 0.012, so a residual of 0.012 in one cell is not by itself informative.
    # What the decomposition reports is the difference of differences, so
    # recompute all four cells of the 2x2 on this same sub-space and compare
    # the interaction index rather than the cells.
    cells = {"baseline": (0.0, 1.0), "erosion": (config.CHAIN.eps, 1.0),
             "attribution": (0.0, config.CHAIN.phi),
             "joint": (config.CHAIN.eps, config.CHAIN.phi)}

    def _sub(eps: float, phi: float):
        fun_c = _functionals(replace(chain, eps=eps, phi=phi))
        ix = np.array([fun_c.index(d, "AS") for d in depths])
        return (np.ascontiguousarray(fun_c.pi_P[np.ix_(ix, ix)]),
                np.ascontiguousarray(fun_c.unsafe_frequency[np.ix_(ix, ix)]))

    sml_cells = {}
    for name, (eps, phi) in cells.items():
        pay, uns = _sub(eps, phi)
        sml_cells[name] = stationary_analysis_sml(
            pay, uns, population_size=Z, beta=config.BETA
        ).unsafe_frequency
    interaction_sml = (sml_cells["joint"] - sml_cells["erosion"]
                       - sml_cells["attribution"] + sml_cells["baseline"])

    interaction_rows = []
    for mu in mutations:
        full_cells = {}
        for name, (eps, phi) in cells.items():
            pay, uns = _sub(eps, phi)
            full_cells[name] = stationary_analysis(
                pay, uns, Z, config.BETA, mu
            ).unsafe_frequency
        inter = (full_cells["joint"] - full_cells["erosion"]
                 - full_cells["attribution"] + full_cells["baseline"])
        interaction_rows.append({
            "mu": float(mu),
            "cells": {k: float(v) for k, v in full_cells.items()},
            "interaction_full": float(inter),
            "abs_difference": abs(float(inter) - float(interaction_sml)),
        })

    joint_err = {r["mu"]: r["abs_difference"] for r in RESULTS["sml_versus_full"]["rows"]
                 if r["mu"] > 0}
    RESULTS["sml_versus_full"]["interaction_sml"] = float(interaction_sml)
    RESULTS["sml_versus_full"]["interaction_rows"] = interaction_rows
    RESULTS["sml_versus_full"]["error_over_mu"] = {
        str(m): e / m for m, e in joint_err.items()
    }
    finest = sorted(joint_err)[:2]
    if len(finest) == 2:
        m2, m1 = finest[0], finest[1]          # m2 = mu/2 (finest), m1 = mu
        u1 = next(r["full_chain_unsafe"] for r in RESULTS["sml_versus_full"]["rows"]
                  if r["mu"] == m1)
        u2 = next(r["full_chain_unsafe"] for r in RESULTS["sml_versus_full"]["rows"]
                  if r["mu"] == m2)
        RESULTS["sml_versus_full"]["richardson"] = {
            "mu_pair": [m1, m2],
            "extrapolated": 2 * u2 - u1,
            "abs_difference": abs(2 * u2 - u1 - sml.unsafe_frequency),
        }


# --------------------------------------------------------------------------
# 5b. the condition Proposition 9 needs, certified on every reported regime
# --------------------------------------------------------------------------


def _closed_classes(p: np.ndarray) -> tuple[int, int, int]:
    """(components, closed classes, zero off-diagonal entries) of a computed P."""
    n = p.shape[0]
    off = ~np.eye(n, dtype=bool)
    pos = (p > 0.0) & off
    ncomp, labels = csgraph.connected_components(
        sp.csr_matrix(pos), directed=True, connection="strong"
    )
    closed = 0
    for c in range(ncomp):
        inside = np.where(labels == c)[0]
        outside = np.setdiff1d(np.arange(n), inside)
        if outside.size == 0 or not pos[np.ix_(inside, outside)].any():
            closed += 1
    return int(ncomp), int(closed), int((~pos & off).sum())


def closed_class_certification() -> None:
    """Check the hypothesis of Proposition 9 across every regime the paper reports."""
    chain = config.CHAIN
    Z, beta = config.POPULATION, config.BETA
    worst = {"n_components_max": 1, "n_closed_max": 1, "zero_entries_max": 0}
    checked = 0

    def _check(ch: ChainParams, z: int, b: float) -> None:
        nonlocal checked, worst
        fun = _functionals(ch)
        p = sml_transition_matrix(np.ascontiguousarray(fun.pi_P), z, b)
        ncomp, closed, zeros = _closed_classes(p)
        worst["n_components_max"] = max(worst["n_components_max"], ncomp)
        worst["n_closed_max"] = max(worst["n_closed_max"], closed)
        worst["zero_entries_max"] = max(worst["zero_entries_max"], zeros)
        checked += 1

    for phi in np.linspace(0.6, 1.0, 21):
        for eps in np.linspace(0.0, 0.4, 21):
            _check(replace(chain, eps=float(eps), phi=float(phi)), Z, beta)
    for z in (50, 100, 200):
        for b in (0.01, 0.05, 0.1, 0.2):
            _check(chain, z, b)
    for kernel in ("ladder", "collapse", "uniform"):
        _check(replace(chain, kernel=kernel), Z, beta)
    for ceiling in (4, 6, 8):
        _check(replace(chain, max_depth=ceiling), Z, beta)
    for gain in (0.0, 1.5, 5.0):
        _check(replace(chain, gain=gain), Z, beta)

    RESULTS["closed_class_certification"] = {
        "regimes_checked": checked,
        "unique_stationary_everywhere": worst["n_closed_max"] == 1,
        **worst,
    }


# --------------------------------------------------------------------------
# 6. wall-clock cost
# --------------------------------------------------------------------------


def timing() -> None:
    chain = config.CHAIN
    fun = _functionals(chain)
    a = np.ascontiguousarray(fun.pi_P)
    u = np.ascontiguousarray(fun.unsafe_frequency)
    Z, beta = config.POPULATION, config.BETA

    reps = 20
    stationary_analysis_sml(a, u, population_size=Z, beta=beta)
    t0 = time.perf_counter()
    for _ in range(reps):
        stationary_analysis_sml(a, u, population_size=Z, beta=beta)
    t_one = (time.perf_counter() - t0) / reps

    t0 = time.perf_counter()
    for _ in range(reps):
        _functionals(chain)
    t_build = (time.perf_counter() - t0) / reps

    # The three subtracted terms of the interaction index are themselves grid
    # points, one per row and one per column, so the plane is 441 stationary
    # regimes and not 4 x 441.  Measure it rather than extrapolating.
    t0 = time.perf_counter()
    for phi in np.linspace(0.6, 1.0, 21):
        for eps in np.linspace(0.0, 0.4, 21):
            f = _functionals(replace(chain, eps=float(eps), phi=float(phi)))
            stationary_analysis_sml(
                np.ascontiguousarray(f.pi_P),
                np.ascontiguousarray(f.unsafe_frequency),
                population_size=Z,
                beta=beta,
            )
    t_grid = time.perf_counter() - t0

    RESULTS["timing"] = {
        "n_designs": int(a.shape[0]),
        "Z": Z,
        "seconds_per_stationary_regime": t_one,
        "seconds_per_functional_build": t_build,
        "grid_points": 21 * 21,
        "measured_grid_seconds": t_grid,
        "exponentials_per_regime": int(a.shape[0]) * (int(a.shape[0]) - 1) * Z,
        "machine": {
            "processor": platform.processor(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
    }


# --------------------------------------------------------------------------
# 7. residual of the stationary solve
# --------------------------------------------------------------------------


def residuals() -> None:
    fun = _functionals(config.CHAIN)
    a = np.ascontiguousarray(fun.pi_P)
    p = sml_transition_matrix(a, config.POPULATION, config.BETA)
    row_err = float(np.abs(p.sum(axis=1) - 1.0).max())
    x = sparse_stationary_distribution(sp.csr_matrix(p))
    resid = float(np.abs(x @ p - x).max())
    RESULTS["residuals"] = {
        "max_row_sum_error": row_err,
        "max_stationarity_residual": resid,
        "min_entry": float(x.min()),
    }


# --------------------------------------------------------------------------
# LaTeX emission
# --------------------------------------------------------------------------


def _sci(x) -> str:
    if isinstance(x, int) and abs(x) < 10**6:
        return "{:,}".format(x).replace(",", "\\,")
    x = float(x)
    if x == 0:
        return "0"
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10.0**e
    if abs(m - round(m)) < 5e-3:
        return "10^{%d}" % e if round(m) == 1 else "%d\\times 10^{%d}" % (round(m), e)
    return "%.1f\\times 10^{%d}" % (m, e)


def emit_tex() -> None:
    ss = {(r["max_depth"], r["Z"]): r for r in RESULTS["state_space"]}
    lines = ["% generated by paper-amc/scripts/run_numerics.py -- do not edit"]

    lines += [
        r"\begin{table}[pos=t]",
        r"\centering",
        r"\caption{Cost of direct and reduced stationary analysis. The full "
        r"chain has $\binom{Z+n-1}{n-1}$ population states; the reduction uses "
        r"$n(n-1)Z$ exponentials, where $n=4(\bar D+1)$. The baseline is "
        r"$\bar D=6$, $Z=100$.}",
        r"\label{tab:cost}",
        r"\begin{tabular*}{\tblwidth}{@{}LLLRR@{}}",
        r"\toprule",
        r"$\bar{D}$ & $n$ & $Z$ & states of the full chain & "
        r"exponentials of the reduction \\",
        r"\midrule",
    ]
    for max_depth in (4, 6, 8):
        for Z in (50, 100, 200):
            r = ss[(max_depth, Z)]
            lines.append(
                "%d & %d & %d & $%s$ & $%s$ \\\\"
                % (max_depth, r["n_designs"], Z, _sci(r["n_states"]), _sci(r["sml_flops"]))
            )
    lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table}", ""]

    sv = RESULTS["sml_versus_full"]
    lines += [
        r"\begin{table}[pos=t]",
        r"\centering",
        r"\caption{Five numerical checks. Discrepancy is payoff-matrix RMSE for "
        r"Monte Carlo, worst relative error for normal fixation probabilities, "
        r"total variation (TV) and unsafe-frequency error for stationary output, "
        r"worst absolute error for EGTtools, and unsafe-frequency error for the "
        r"finite-mutation chain.}",
        r"\label{tab:verification}",
        r"\begin{tabular*}{\tblwidth}{@{}LLR@{}}",
        r"\toprule",
        r"check & setting & discrepancy \\",
        r"\midrule",
    ]
    for r in RESULTS["monte_carlo"]:
        lines.append(
            "Monte Carlo vs.\\ exact & $N=%s$ episodes & $%.3f$ \\\\"
            % (_sci(r["n_episodes"]), r["rms_error_a"])
        )
    lines.append(r"\midrule")
    for key, r in RESULTS["stability"]["by_beta"].items():
        lines.append(
            "stabilised vs.\\ $60$-digit sum & $n=%d$, $Z=%d$, $\\beta=%s$ & $%s$ \\\\"
            % (
                RESULTS["stability"]["n_designs"],
                RESULTS["stability"]["population_size"],
                key.split("=")[1],
                _sci(r["max_relative_error_stabilised_vs_60_digits"]),
            )
        )
    lines.append(r"\midrule")
    for r in RESULTS["stationary_end_to_end"]:
        lines.append(
            "stationary vector vs.\\ $60$-digit reference & $n=28$, $Z=100$, "
            "$\\beta=%g$ & TV $%s$; $U$ gap $%s$ \\\\"
            % (r["beta"], _sci(r["total_variation"]), _sci(abs(r["unsafe_fast"] - r["unsafe_reference"])))
        )
    lines.append(r"\midrule")
    for r in RESULTS["cross_check"]:
        lines.append(
            "closed form vs.\\ EGTtools & $n=%d$, $Z=%d$, $\\beta=%g$ & $%s$ \\\\"
            % (r["n_designs"], r["Z"], r["beta"], _sci(r["max_abs_difference"]))
        )
    lines.append(r"\midrule")
    for r in [row for row in sv["rows"] if row["mu"] > 0]:
        lines.append(
            "small mutation vs.\\ full chain & $\\mu=%s$ & $%.4f$ \\\\"
            % (_sci(r["mu"]), r["abs_difference"])
        )
    lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table}"]

    (OUT / "numerics_generated.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    for step in (
        state_space_sizes,
        monte_carlo_comparison,
        stability,
        cross_check,
        sml_versus_full,
        closed_class_certification,
        timing,
        residuals,
    ):
        step()
        print(step.__name__, "done", flush=True)
    emit_tex()
    (OUT / "numerics_key_numbers.json").write_text(
        json.dumps(RESULTS, indent=2, default=float), encoding="utf-8"
    )
    print(json.dumps(RESULTS, indent=2, default=float))


if __name__ == "__main__":
    main()
