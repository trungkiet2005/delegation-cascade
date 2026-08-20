"""Compute every table and every number quoted in the manuscript.

Usage::

    python scripts/run_analysis.py [--outdir results]

Writes ``results/tables/*.csv``, the LaTeX tables included by the manuscript,
and ``results/key_numbers.json``, which holds every scalar the text quotes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dcascade import config as cfg
from dcascade import interventions as iv
from dcascade import theory as th
from dcascade.chain import (
    ChainParams,
    handoff_kernel,
    intent_separation,
    second_eigenvalue,
    specification_half_life,
    transmission,
)
from dcascade.functionals import build_functionals
from dcascade.race import STRATEGIES, build_race_tables
from dcascade.robustness import deniability_threshold_profile, leverage_decay

ROOT = Path(__file__).resolve().parents[1]


def _frame(matrix: np.ndarray, labels) -> pd.DataFrame:
    return pd.DataFrame(matrix, index=list(labels), columns=list(labels))


def main(outdir: Path) -> None:
    tables_dir = outdir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    key: dict[str, object] = {}

    race = build_race_tables(cfg.RACE)
    chain = cfg.CHAIN
    fun = build_functionals(race, chain, cfg.LAM, cfg.HARM)
    sml = dict(population_size=cfg.POPULATION, beta=cfg.BETA)
    rep = dict(n_starts=cfg.REPLICATOR_STARTS, seed=cfg.SEED)

    # ---------------------------------------------------------------- layer 0
    _frame(race.payoff, STRATEGIES).to_csv(tables_dir / "race_payoff.csv")
    _frame(race.unsafe_count, STRATEGIES).to_csv(tables_dir / "race_unsafe_count.csv")
    _frame(race.unsafe_frequency, STRATEGIES).to_csv(tables_dir / "race_unsafe_frequency.csv")

    key["expected_horizon"] = cfg.RACE.expected_horizon
    key["erosion_order_is_harm_order"] = th.erosion_monotone(race)
    key["erosion_order_is_frequency_order"] = th.erosion_monotone_frequency(race)
    key["erosion_order_is_pathwise_order"] = th.erosion_monotone_pathwise()
    key["invasion_cas_into_as"] = th.invasion_threshold_depth_zero(race, "CAS", "AS")
    key["invasion_cas_into_cs"] = th.invasion_threshold_depth_zero(race, "CAS", "CS")
    key["critical_liability_depth_zero"] = th.critical_liability(race, "sml", **sml)
    key["effective_liability"] = cfg.effective_liability()
    key["liability_margin_over_critical"] = (
        cfg.effective_liability() / key["critical_liability_depth_zero"]
    )
    key["attribution_failure_depth"] = th.attribution_failure_depth(
        cfg.effective_liability(), float(key["critical_liability_depth_zero"]), chain.phi
    )

    # -------------------------------------------------------- the hand-off law
    _frame(handoff_kernel(chain.eps, chain.kernel), STRATEGIES).to_csv(
        tables_dir / "handoff_kernel.csv"
    )
    rows = []
    for d in chain.depths:
        p = transmission(d, chain.eps, chain.kernel)
        for i, intent in enumerate(STRATEGIES):
            rows.append({"depth": d, "intent": intent, **dict(zip(STRATEGIES, p[i]))})
    pd.DataFrame(rows).to_csv(tables_dir / "transmission.csv", index=False)

    key["specification_half_life"] = specification_half_life(chain.eps, chain.kernel)
    key["second_eigenvalue"] = second_eigenvalue(chain.eps, chain.kernel)
    key["fidelity_at_ceiling"] = float(
        transmission(chain.max_depth, chain.eps, chain.kernel)[0, 0]
    )

    decay = leverage_decay(race, chain, probe_depth=40, lam=cfg.LAM, harm=cfg.HARM)
    separation = np.array(
        [intent_separation(int(d), chain.eps, chain.kernel) for d in decay["depths"]]
    )
    fidelities = np.array(
        [float(transmission(int(d), chain.eps, chain.kernel)[0, 0]) for d in decay["depths"]]
    )
    pd.DataFrame(
        {
            "depth": decay["depths"],
            "fidelity": fidelities,
            "geometric_reference": decay["reference"],
            "intent_separation": separation,
            "intent_range": decay["leverage"],
            "clause_value": decay["clause_value"],
        }
    ).to_csv(tables_dir / "leverage_decay.csv", index=False)
    key["fidelity_is_geometric_max_error"] = float(np.abs(fidelities - decay["reference"]).max())
    key["intent_separation_at_ceiling"] = float(separation[chain.max_depth])
    key["separation_geometric_max_error"] = float(
        np.abs(separation[: chain.max_depth + 1] - decay["reference"][: chain.max_depth + 1]).max()
    )
    key["clause_value_peak_depth"] = int(np.argmax(decay["clause_value"]))

    # ------------------------------------------------------- depth profiles
    profile_rows = []
    for intent in STRATEGIES:
        prof = th.self_profile(fun, intent)
        for k, d in enumerate(prof.depths):
            profile_rows.append(
                {
                    "intent": intent,
                    "depth": int(d),
                    "harm": prof.harm[k],
                    "attributed_harm": prof.attributed[k],
                    "task": prof.task[k],
                    "private": prof.private[k],
                    "social": prof.social[k],
                }
            )
    pd.DataFrame(profile_rows).to_csv(tables_dir / "depth_profiles.csv", index=False)

    shelter = deniability_threshold_profile(
        race, chain, "AS", probe_depth=40, lam=cfg.LAM, harm=cfg.HARM
    )
    pd.DataFrame(
        {
            "depth": np.arange(len(shelter["thresholds"])),
            "phi_star": shelter["thresholds"],
            "harm": shelter["harm"][: len(shelter["thresholds"])],
            "attributed": shelter["attributed"][: len(shelter["thresholds"])],
        }
    ).to_csv(tables_dir / "deniability_thresholds.csv", index=False)
    key["shelter_onset_depth"] = shelter["first_sheltering_depth"]
    key["phi_star_limit"] = shelter["limit_threshold"]

    prof_as = th.self_profile(fun, "AS")
    key["harm_at_ceiling_AS"] = float(prof_as.harm[-1])
    key["attributed_at_ceiling_AS"] = float(prof_as.attributed[-1])
    key["shelter_factor_AS"] = float(prof_as.harm[-1] / prof_as.attributed[-1])
    key["socially_optimal_depth_AS"] = th.socially_optimal_depth(fun, "AS")

    # --------------------------------------------------- depth invasion ladder
    inv_rows = []
    for intent in STRATEGIES:
        for d in range(chain.max_depth):
            inv = th.depth_invasion(fun, intent, d)
            inv_rows.append(
                {
                    "intent": intent,
                    "depth": d,
                    "benefit": inv.benefit,
                    "attributed_gain": inv.attributed_gain,
                    "critical_liability": inv.critical_liability,
                    "direction": inv.direction,
                    "deeper_invades": inv.deeper_invades_at(cfg.effective_liability()),
                }
            )
    pd.DataFrame(inv_rows).to_csv(tables_dir / "depth_invasion.csv", index=False)

    # ------------------------------------------------------------- the 2x2
    decomposition = {}
    for method, kwargs in (("sml", sml), ("replicator", rep)):
        dec = th.mechanism_decomposition(
            race, chain, cfg.LAM, cfg.HARM, method=method, **kwargs
        )
        decomposition[method] = dec
        key[f"decomposition_{method}"] = {
            "baseline": dec.baseline.unsafe_frequency,
            "drift_only": dec.drift_only.unsafe_frequency,
            "deniability_only": dec.deniability_only.unsafe_frequency,
            "both": dec.both.unsafe_frequency,
            "drift_effect": dec.drift_effect,
            "deniability_effect": dec.deniability_effect,
            "total_effect": dec.total_effect,
            "interaction": dec.interaction,
            "interaction_share": dec.interaction_share,
            "depth_baseline": dec.baseline.mean_depth,
            "depth_drift_only": dec.drift_only.mean_depth,
            "depth_deniability_only": dec.deniability_only.mean_depth,
            "depth_both": dec.both.mean_depth,
            "gap_both": dec.both.declaration_gap,
            "social_baseline": dec.baseline.social_payoff,
            "social_both": dec.both.social_payoff,
        }

    dec = decomposition["sml"]
    pd.DataFrame(
        [
            {
                "cell": name,
                "erosion": e,
                "attribution": p,
                "unsafe_frequency": q.unsafe_frequency,
                "mean_depth": q.mean_depth,
                "declaration_gap": q.declaration_gap,
                "social_payoff": q.social_payoff,
                **{f"intent_{k}": v for k, v in q.intent_distribution.items()},
            }
            for name, e, p, q in (
                ("baseline", 0.0, 1.0, dec.baseline),
                ("drift only", chain.eps, 1.0, dec.drift_only),
                ("deniability only", 0.0, chain.phi, dec.deniability_only),
                ("both", chain.eps, chain.phi, dec.both),
            )
        ]
    ).to_csv(tables_dir / "decomposition.csv", index=False)

    frozen = th.frozen_depth_counterfactual(race, chain, cfg.LAM, cfg.HARM, method="sml", **sml)
    key["frozen_composition"] = frozen
    pd.DataFrame([frozen]).to_csv(tables_dir / "frozen_composition.csv", index=False)

    # -------------------------------------------------------- the equilibrium
    eq = th.equilibrium(fun, method="sml", **sml)
    key["baseline_equilibrium"] = {
        "unsafe_frequency": eq.unsafe_frequency,
        "mean_depth": eq.mean_depth,
        "declaration_gap": eq.declaration_gap,
        "social_payoff": eq.social_payoff,
        "intent": eq.intent_distribution,
        "executed": dict(zip(STRATEGIES, eq.executed_distribution.tolist())),
        "dominant_design": fun.labels[int(np.argmax(eq.frequencies))],
        "dominant_share": float(eq.frequencies.max()),
    }
    pd.DataFrame(
        {"design": fun.labels, "frequency": eq.frequencies, "depth": fun.depth,
         "intent": fun.intent}
    ).to_csv(tables_dir / "equilibrium_designs.csv", index=False)

    # ------------------------------------------------------------ instruments
    sweeps = {
        "depth_cap": iv.depth_cap_sweep(
            race, chain, np.arange(chain.max_depth, -1, -1), cfg.LAM, cfg.HARM, "sml", **sml
        ),
        "pass_through": iv.pass_through_sweep(
            race, chain, np.linspace(chain.phi, 1.0, 26), cfg.LAM, cfg.HARM, "sml", **sml
        ),
        "attestation": iv.attestation_sweep(
            race, chain, np.linspace(1.0, 0.0, 21), cfg.LAM, cfg.HARM, "sml", **sml
        ),
        "audit_layer": iv.audit_placement_sweep(
            race, chain, 1.0, cfg.LAM, cfg.HARM, "sml", **sml
        ),
        "audit_layer_half": iv.audit_placement_sweep(
            race, chain, 0.5, cfg.LAM, cfg.HARM, "sml", **sml
        ),
        "liability": iv.liability_sweep(
            race, chain, np.geomspace(1.0, 200.0, 40), cfg.LAM, cfg.HARM, "sml", **sml
        ),
    }
    rows = []
    for name, outcomes in sweeps.items():
        for o in outcomes:
            rows.append(
                {
                    "instrument": name,
                    "setting": o.setting,
                    "unsafe_frequency": o.unsafe_frequency,
                    "mean_depth": o.mean_depth,
                    "social_payoff": o.social_payoff,
                    "declaration_gap": o.declaration_gap,
                }
            )
    pd.DataFrame(rows).to_csv(tables_dir / "instruments.csv", index=False)

    caps = {int(o.setting): o for o in sweeps["depth_cap"]}
    key["depth_cap"] = {
        str(k): {
            "unsafe_frequency": o.unsafe_frequency,
            "social_payoff": o.social_payoff,
            "mean_depth": o.mean_depth,
        }
        for k, o in sorted(caps.items())
    }
    best_cap = max(caps.values(), key=lambda o: o.social_payoff)
    key["welfare_optimal_cap"] = {
        "cap": int(best_cap.setting),
        "unsafe_frequency": best_cap.unsafe_frequency,
        "social_payoff": best_cap.social_payoff,
        "social_gain_over_uncapped": best_cap.social_payoff
        - caps[chain.max_depth].social_payoff,
        "unsafe_reduction_over_uncapped": caps[chain.max_depth].unsafe_frequency
        - best_cap.unsafe_frequency,
    }


    pt = sweeps["pass_through"]
    below = [o for o in pt if o.unsafe_frequency <= 0.5 * pt[0].unsafe_frequency]
    key["pass_through_halving_phi"] = below[0].setting if below else None
    # the attribution retention at which the ceiling stops binding: located on the
    # sweep, and compared with the closed form implied by the depth-zero threshold
    jumps = np.diff([o.mean_depth for o in pt])
    key["pass_through_critical_phi"] = float(pt[int(np.argmin(jumps)) + 1].setting)
    key["pass_through_critical_phi_predicted"] = float(
        (key["critical_liability_depth_zero"] / cfg.effective_liability())
        ** (1.0 / chain.max_depth)
    )

    key["pass_through_at_099"] = next(
        o.unsafe_frequency for o in pt if abs(o.setting - 0.99) < 1e-9
    ) if any(abs(o.setting - 0.99) < 1e-9 for o in pt) else None

    att = sweeps["attestation"]  # ordered from no attestation to perfect
    key["attestation_non_monotone"] = bool(
        np.any(np.diff([o.unsafe_frequency for o in att]) > 1e-6)
    )
    values = np.array([o.unsafe_frequency for o in att])
    rises = np.flatnonzero(np.diff(values) > 1e-6)
    key["attestation_perverse"] = {
        "no_attestation_unsafe": float(values[0]),
        "full_attestation_unsafe": float(values[-1]),
        "best_before_reversal": float(values[: rises[0] + 1].min()) if rises.size else None,
        "setting_before_reversal": float(att[int(np.argmin(values[: rises[0] + 1]))].setting)
        if rises.size
        else None,
        "worst_after_reversal": float(values[rises[0] + 1 :].max()) if rises.size else None,
        "setting_after_reversal": float(
            att[int(rises[0] + 1 + np.argmax(values[rises[0] + 1 :]))].setting
        )
        if rises.size
        else None,
    }

    audit = sweeps["audit_layer"]
    key["audit_placement"] = {
        str(int(o.setting)): o.unsafe_frequency for o in audit
    }
    key["audit_late_vs_early"] = audit[0].unsafe_frequency / max(
        audit[-1].unsafe_frequency, 1e-12
    )

    key["matched_effort"] = iv.matched_effort_comparison(
        race, chain, 0.05, cfg.LAM, cfg.HARM, "sml", **sml
    )

    # ------------------------------------------------------- attribution regimes
    # Does the shelter depend on attribution being geometric, and can it be closed
    # without restoring attribution hand-off by hand-off?
    regimes = iv.attribution_regime_comparison(
        race, chain, cfg.LAM, cfg.HARM, "sml", **sml
    )
    pd.DataFrame(
        [
            {
                "regime": o.instrument,
                "setting": o.setting,
                "unsafe_frequency": o.unsafe_frequency,
                "mean_depth": o.mean_depth,
                "declaration_gap": o.declaration_gap,
                "social_payoff": o.social_payoff,
            }
            for o in regimes
        ]
    ).to_csv(tables_dir / "attribution_regimes.csv", index=False)
    key["attribution_regimes"] = {
        f"{o.instrument}@{o.setting:g}": {
            "unsafe_frequency": o.unsafe_frequency,
            "mean_depth": o.mean_depth,
            "social_payoff": o.social_payoff,
        }
        for o in regimes
    }

    floors = np.round(np.arange(0.0, 1.001, 0.01), 3)
    floor_sweep = iv.attribution_floor_sweep(
        race, chain, floors, cfg.LAM, cfg.HARM, "sml", **sml
    )
    pd.DataFrame(
        [
            {
                "floor": o.setting,
                "unsafe_frequency": o.unsafe_frequency,
                "mean_depth": o.mean_depth,
                "social_payoff": o.social_payoff,
                "declaration_gap": o.declaration_gap,
            }
            for o in floor_sweep
        ]
    ).to_csv(tables_dir / "attribution_floor.csv", index=False)

    binding = float(chain.phi**chain.max_depth)
    onset = next(
        (o.setting for o in floor_sweep if o.mean_depth < chain.max_depth - 0.01), None
    )
    key["attribution_floor"] = {
        "attribution_at_ceiling": binding,
        "onset_floor": onset,
        "floor_for_half_the_harm": next(
            (
                o.setting
                for o in floor_sweep
                if o.unsafe_frequency <= 0.5 * floor_sweep[0].unsafe_frequency
            ),
            None,
        ),
        "welfare_recovered_at_half": None,
    }
    strict = next(o for o in regimes if o.instrument == "strict")
    base = floor_sweep[0]
    at_half = next(o for o in floor_sweep if abs(o.setting - 0.5) < 1e-9)
    key["attribution_floor"]["welfare_recovered_at_half"] = (
        at_half.social_payoff - base.social_payoff
    ) / (strict.social_payoff - base.social_payoff)

    # a floor and a hard depth cap trace the same frontier: for every cap, the floor
    # that matches its social payoff also matches its Unsafe frequency
    cap_match = []
    for c in sweeps["depth_cap"]:
        k = int(np.argmin([abs(o.social_payoff - c.social_payoff) for o in floor_sweep]))
        o = floor_sweep[k]
        cap_match.append(
            {
                "cap": int(c.setting),
                "U_cap": c.unsafe_frequency,
                "social_cap": c.social_payoff,
                "matched_floor": o.setting,
                "U_floor": o.unsafe_frequency,
                "social_floor": o.social_payoff,
                "abs_gap": abs(o.unsafe_frequency - c.unsafe_frequency),
            }
        )
    pd.DataFrame(cap_match).to_csv(tables_dir / "floor_versus_cap.csv", index=False)
    gaps = np.array([r["abs_gap"] for r in cap_match])
    matched = np.array([abs(r["social_floor"] - r["social_cap"]) < 1e-6 for r in cap_match])
    key["floor_versus_cap"] = {
        "max_gap": float(gaps.max()),
        "mean_gap": float(gaps.mean()),
        "mean_gap_over_welfare_matched": float(gaps[matched].mean()) if matched.any() else None,
        "n_welfare_matched": int(matched.sum()),
        "caps_matched_within_001": int((gaps < 0.01).sum()),
        "n_caps": int(gaps.size),
        "rows": cap_match,
    }

    # How far does the floor/cap agreement travel?  The manuscript quotes a range
    # over phi, eps and the ceiling and a separate sensitivity to the
    # organisational benefit, so store both rather than asserting them.
    key["floor_versus_cap_sensitivity"] = _floorcap_sensitivity(race)

    _write_latex_tables(tables_dir, race, fun, dec, sweeps, key, regimes)

    with open(outdir / "key_numbers.json", "w", encoding="utf-8") as handle:
        json.dump(key, handle, indent=2, default=_default)
    print(f"wrote {outdir / 'key_numbers.json'} and {len(list(tables_dir.glob('*.csv')))} tables")


def _floorcap_sensitivity(race) -> dict:
    """Mean floor-versus-cap gap under variations of the four free parameters.

    The manuscript separates two claims: that the agreement holds across the
    transmission and attribution parameters and the ceiling, and that it is
    sensitive to the organisational benefit.  Both are computed here so neither
    has to be asserted.
    """
    from dataclasses import replace as _replace

    sml = dict(population_size=cfg.POPULATION, beta=cfg.BETA)

    def _outcome(chain):
        fun = build_functionals(race, chain, cfg.LAM, cfg.HARM)
        eq = th.equilibrium(fun, "sml", **sml)
        return eq.unsafe_frequency, eq.social_payoff

    def _mean_gap(chain) -> float:
        caps = [_outcome(_replace(chain, max_depth=d)) for d in range(chain.max_depth + 1)]
        floors = [
            (a, *_outcome(_replace(chain, attribution_rule="floored",
                                   attribution_floor=float(a))))
            for a in np.round(np.arange(0.0, 1.001, 0.01), 2)
        ]
        gaps = []
        for u_cap, s_cap in caps:
            k = int(np.argmin([abs(s - s_cap) for _, _, s in floors]))
            gaps.append(abs(floors[k][1] - u_cap))
        return float(np.mean(gaps))

    grid = {
        "baseline": cfg.CHAIN,
        "phi=0.6": _replace(cfg.CHAIN, phi=0.6),
        "phi=0.9": _replace(cfg.CHAIN, phi=0.9),
        "eps=0.10": _replace(cfg.CHAIN, eps=0.10),
        "eps=0.40": _replace(cfg.CHAIN, eps=0.40),
        "Dbar=4": _replace(cfg.CHAIN, max_depth=4),
        "Dbar=8": _replace(cfg.CHAIN, max_depth=8),
        "g=0": _replace(cfg.CHAIN, gain=0.0),
        "g=5": _replace(cfg.CHAIN, gain=5.0),
    }
    cells = {label: _mean_gap(chain) for label, chain in grid.items()}
    core = [v for k, v in cells.items() if not k.startswith("g=")]
    return {
        "mean_gap_by_cell": cells,
        "core_range": [float(min(core)), float(max(core))],
        "benefit_range": [float(cells["g=0"]), float(cells["g=5"])],
    }


def _default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    raise TypeError(type(obj))


def _write_latex_tables(tables_dir: Path, race, fun, dec, sweeps, key, regimes) -> None:
    """Emit the LaTeX tables that the manuscript includes verbatim."""
    lines: list[str] = []

    lines.append("% generated by scripts/run_analysis.py -- do not edit")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Interaction layer at depth zero. Task payoff $a(i,j)$ and expected "
        r"number of Unsafe actions $m(i,j)$ of the focal design $i$ against $j$, "
        r"evaluated exactly over the horizon law. Reading down a column, $m$ never "
        r"falls: the erosion order is a harm order.}"
    )
    lines.append(r"\label{tab:race}")
    lines.append(r"\begin{tabular}{l" + "r" * 8 + "}")
    lines.append(r"\toprule")
    lines.append(
        r"& \multicolumn{4}{c}{$a(i,j)$} & \multicolumn{4}{c}{$m(i,j)$} \\"
    )
    lines.append(r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}")
    lines.append("$i$ & " + " & ".join(STRATEGIES) + " & " + " & ".join(STRATEGIES) + r" \\")
    lines.append(r"\midrule")
    for i, s in enumerate(STRATEGIES):
        payoff = " & ".join(f"{v:.2f}" for v in race.payoff[i])
        harm = " & ".join(f"{v:.2f}" for v in race.unsafe_count[i])
        lines.append(f"{s} & {payoff} & {harm} " + r"\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{The two mechanisms, separately and together. Long-run Unsafe "
        r"frequency $U$, mean delegation depth $\bar{d}$, declaration gap and social "
        r"payoff in the stationary regime of the finite-population process. "
        r"Neither mechanism does much alone; the interaction "
        rf"is {dec.interaction:.3f}, or {100 * dec.interaction_share:.0f}\% of the "
        r"joint effect.}"
    )
    lines.append(r"\label{tab:decomposition}")
    lines.append(r"\begin{tabular}{llrrrr}")
    lines.append(r"\toprule")
    lines.append(
        r"erosion $\varepsilon$ & attribution $\phi$ & $U$ & $\bar{d}$ & gap & $\pi_S$ \\"
    )
    lines.append(r"\midrule")
    for name, e, p, q in (
        ("$0$", "$1$", 0, dec.baseline),
        (rf"${fun.chain.eps:g}$", "$1$", 1, dec.drift_only),
        ("$0$", rf"${fun.chain.phi:g}$", 2, dec.deniability_only),
        (rf"${fun.chain.eps:g}$", rf"${fun.chain.phi:g}$", 3, dec.both),
    ):
        lines.append(
            f"{name} & {e} & {q.unsafe_frequency:.4f} & {q.mean_depth:.2f} & "
            f"{q.declaration_gap:.3f} & {q.social_payoff:.1f} " + r"\\"
        )
    lines.append(r"\midrule")
    lines.append(
        rf"\multicolumn{{2}}{{l}}{{interaction}} & {dec.interaction:+.4f} & "
        rf"& & \\"
    )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    caps = {int(o.setting): o for o in sweeps["depth_cap"]}
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Depth ceilings. Long-run Unsafe frequency and social payoff when "
        r"chains longer than $\bar{D}$ hand-offs are forbidden. The social payoff is "
        r"maximised well inside the ceiling the market selects.}"
    )
    lines.append(r"\label{tab:cap}")
    lines.append(r"\begin{tabular}{lrrr}")
    lines.append(r"\toprule")
    lines.append(r"$\bar{D}$ & $U$ & $\bar{d}$ & $\pi_S$ \\")
    lines.append(r"\midrule")
    for cap in sorted(caps):
        o = caps[cap]
        lines.append(
            f"{cap} & {o.unsafe_frequency:.4f} & {o.mean_depth:.2f} & {o.social_payoff:.1f} "
            + r"\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    (tables_dir / "tables.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # the attribution-regime tables are read much later in the manuscript, so they
    # go to their own file and are input where they are discussed
    lines = ["% generated by scripts/run_analysis.py -- do not edit"]
    label = {
        "geometric": r"geometric, $\phi^{d}$",
        "strict": r"strict, $1$",
        "harmonic": r"equal split, $1/(1+d)$",
        "floored": r"floored, $\max(\phi^{d}, a_{\min})$",
        "super": r"super-attribution, $\phi^{d}$",
    }
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Attribution regimes. Long-run Unsafe frequency, mean delegation "
        r"depth and social payoff when the law mapping depth to the principal's share "
        r"of the harm is changed, with everything else at the baseline. The shelter "
        r"survives every rule whose attribution vanishes in the depth, including the "
        r"equal split. A rule bounded below by $a_{\min}$ closes it only above the "
        r"depth $d_{0}$ at which the bound binds, $\phi^{d_{0}}=a_{\min}$, which is "
        r"why $a_{\min}=0.25$ ($d_{0}=4.8$) still leaves mean depth at $5.07$ while "
        r"$a_{\min}=0.5$ ($d_{0}=2.4$) brings it to $2.99$.}"
    )
    lines.append(r"\label{tab:regimes}")
    lines.append(r"\begin{tabular}{llrrrr}")
    lines.append(r"\toprule")
    lines.append(
        r"rule & setting & $r(\bar{D})$ & $U$ & $\bar{d}$ & $\pi_S$ \\"
    )
    lines.append(r"\midrule")
    for o in regimes:
        if o.instrument == "geometric":
            setting, terminal = rf"$\phi={o.setting:g}$", fun.chain.phi**fun.chain.max_depth
        elif o.instrument == "floored":
            setting, terminal = rf"$a_{{\min}}={o.setting:g}$", o.setting
        elif o.instrument == "super":
            setting, terminal = rf"$\phi={o.setting:g}$", o.setting**fun.chain.max_depth
        else:
            setting, terminal = "--", 1.0 if o.instrument == "strict" else 1.0 / (
                1.0 + fun.chain.max_depth
            )
        lines.append(
            f"{label[o.instrument]} & {setting} & {terminal:.3f} & "
            f"{o.unsafe_frequency:.4f} & {o.mean_depth:.2f} & {o.social_payoff:.1f} "
            + r"\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    match = key["floor_versus_cap"]
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{A floor under attribution reproduces a depth cap. For each "
        r"ceiling $\bar{D}$, the attribution floor $a_{\min}$ whose social payoff "
        r"comes closest to that ceiling's, and the Unsafe frequency each delivers. "
        r"Wherever a floor can match a ceiling on welfare, at $\bar{D}=2$ to $6$, it "
        r"matches its Unsafe frequency to within $0.0011$. The two shallowest "
        r"ceilings are out of a floor's reach, since even $a_{\min}=1$ leaves social "
        r"payoff at $58.3$ against $59.0$ and $60.5$; the mean gap of "
        rf"{match['mean_gap']:.3f} and the largest of {match['max_gap']:.3f} include "
        r"those two rows and so overstate the disagreement, the mean over the five "
        r"matched ceilings being $0.0003$.}"
    )
    lines.append(r"\label{tab:floorcap}")
    lines.append(r"\begin{tabular}{lrrrrr}")
    lines.append(r"\toprule")
    lines.append(
        r"& \multicolumn{2}{c}{depth cap} & \multicolumn{3}{c}{matched floor} \\"
    )
    lines.append(r"\cmidrule(lr){2-3}\cmidrule(lr){4-6}")
    lines.append(r"$\bar{D}$ & $U$ & $\pi_S$ & $a_{\min}$ & $U$ & $\pi_S$ \\")
    lines.append(r"\midrule")
    for row in sorted(match["rows"], key=lambda r: r["cap"]):
        lines.append(
            f"{row['cap']} & {row['U_cap']:.4f} & {row['social_cap']:.1f} & "
            f"{row['matched_floor']:.2f} & {row['U_floor']:.4f} & "
            f"{row['social_floor']:.1f} " + r"\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    (tables_dir / "tables_regimes.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    main(args.outdir)
