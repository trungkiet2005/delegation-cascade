"""Recompute the main result across every free parameter of the model.

Usage::

    python scripts/run_robustness.py [--outdir results] [--quick]

Writes ``results/tables/robustness_*.csv`` and ``results/robustness_summary.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dcascade import config as cfg
from dcascade import robustness as rb
from dcascade.race import build_race_tables

ROOT = Path(__file__).resolve().parents[1]


def main(outdir: Path, quick: bool) -> None:
    tables_dir = outdir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    race = build_race_tables(cfg.RACE)
    chain = cfg.CHAIN
    sml = dict(population_size=cfg.POPULATION, beta=cfg.BETA)
    summary: dict[str, object] = {}

    cells = []
    cells += rb.sweep_interaction_layer(
        chain,
        cfg.LAM,
        cfg.HARM,
        prizes=(100.0,) if quick else (50.0, 100.0, 200.0),
        risks=(0.3, 0.6) if quick else (0.1, 0.3, 0.6, 0.9),
        method="sml",
        **sml,
    )
    cells += rb.sweep_setback_scope(chain, cfg.LAM, cfg.HARM, "sml", **sml)
    cells += rb.sweep_kernel(race, chain, cfg.LAM, cfg.HARM, "sml", **sml)
    cells += rb.sweep_organisation(race, chain, cfg.LAM, cfg.HARM, method="sml", **sml)
    cells += rb.sweep_process(race, chain, cfg.LAM, cfg.HARM)
    cells += rb.sweep_liability(
        race, chain, np.array([5.0, 10.0, 20.0, 40.0, 80.0]), cfg.HARM, "sml", **sml
    )

    frame = pd.DataFrame([c.as_row() for c in cells])
    frame.to_csv(tables_dir / "robustness_cells.csv", index=False)

    positive = frame[frame.U_both > frame.U_baseline + 1e-9]
    summary["n_cells"] = int(len(frame))
    summary["n_cells_with_depth_effect"] = int(len(positive))
    summary["interaction_positive_share"] = float((frame.interaction > 1e-6).mean())
    summary["interaction_median"] = float(frame.interaction.median())
    summary["interaction_iqr"] = [
        float(frame.interaction.quantile(0.25)),
        float(frame.interaction.quantile(0.75)),
    ]
    summary["interaction_share_median"] = float(
        positive.interaction_share.median() if len(positive) else float("nan")
    )
    summary["depth_rises_under_per_layer_share"] = float(
        (frame.depth_per_layer > frame.depth_passthrough + 1e-6).mean()
    )
    for label in sorted(frame.sweep.unique()):
        block = frame[frame.sweep == label]
        summary[f"sweep_{label}"] = {
            "n": int(len(block)),
            "interaction_min": float(block.interaction.min()),
            "interaction_max": float(block.interaction.max()),
            "U_both_min": float(block.U_both.min()),
            "U_both_max": float(block.U_both.max()),
        }

    kernels = frame[frame.sweep == "kernel"].set_index("setting")
    summary["kernel_comparison"] = {
        str(k): {
            "U_both": float(kernels.loc[k, "U_both"]),
            "U_baseline": float(kernels.loc[k, "U_baseline"]),
            "interaction": float(kernels.loc[k, "interaction"]),
            "depth_per_layer": float(kernels.loc[k, "depth_per_layer"]),
        }
        for k in kernels.index
    }

    # the small-mutation reduction against the full mutation-selection chain
    chain_check = rb.sml_versus_full_chain(
        race, chain, "AS", cfg.LAM, cfg.HARM, population_size=60, beta=cfg.BETA,
        depths=(0, chain.max_depth // 2, chain.max_depth),
    )
    pd.DataFrame(chain_check).to_csv(tables_dir / "robustness_sml_check.csv", index=False)
    sml_row = chain_check[0]
    finest = chain_check[-1]
    summary["sml_versus_full_chain"] = {
        "sml_unsafe": sml_row["unsafe_frequency"],
        "full_chain_unsafe_at_smallest_mu": finest["unsafe_frequency"],
        "absolute_difference": abs(sml_row["unsafe_frequency"] - finest["unsafe_frequency"]),
        "sml_mean_depth": sml_row["mean_depth"],
        "full_chain_mean_depth": finest["mean_depth"],
    }

    # the attribution / erosion plane
    n = 9 if quick else 21
    grid = rb.equilibrium_grid(
        race,
        chain,
        phis=np.linspace(0.6, 1.0, n),
        epsilons=np.linspace(0.0, 0.4, n),
        lam=cfg.LAM,
        harm=cfg.HARM,
        method="sml",
        **sml,
    )
    np.savez(outdir / "grid.npz", **{k: v for k, v in grid.items()})
    pd.DataFrame(
        grid["unsafe"], index=[f"{p:.3f}" for p in grid["phis"]],
        columns=[f"{e:.3f}" for e in grid["epsilons"]]
    ).to_csv(tables_dir / "robustness_grid_unsafe.csv")
    pd.DataFrame(
        grid["interaction"], index=[f"{p:.3f}" for p in grid["phis"]],
        columns=[f"{e:.3f}" for e in grid["epsilons"]]
    ).to_csv(tables_dir / "robustness_grid_interaction.csv")
    interior = grid["interaction"][:-1, 1:]
    summary["grid"] = {
        "share_positive_interaction": float((interior > 1e-6).mean()),
        "max_interaction": float(interior.max()),
        "max_unsafe": float(grid["unsafe"].max()),
        "max_declaration_gap": float(grid["declaration_gap"].max()),
    }

    with open(outdir / "robustness_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=float)
    print(
        f"wrote {outdir / 'robustness_summary.json'}: {summary['n_cells']} cells, "
        f"interaction positive in {100 * summary['interaction_positive_share']:.0f}% of them"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=ROOT / "results")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    main(args.outdir, args.quick)
