r"""Generate every manuscript figure.

Every figure is saved at the standard width ``dcascade.plotting.FIG_WIDTH`` with
a fixed (uncropped) bounding box and is included at ``\linewidth``, so all
figures are reduced by the same factor on the page and their text renders at the
same size.  Font sizes come from ``dcascade.plotting.FS`` and are never chosen
per panel.

Usage
-----
    python scripts/make_figures.py [--outdir results/figures] [--quick] [--only 3 5]
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from dcascade import config as cfg
from dcascade import interventions as iv
from dcascade import robustness as rb
from dcascade import theory as th
from dcascade.chain import AuditPlan, handoff_kernel, specification_half_life, transmission
from dcascade.functionals import build_functionals
from dcascade.plotting import (
    FS,
    PALETTE,
    STRATEGY_LABEL,
    depth_colour,
    fitted_legend,
    new_figure,
    panel_title,
    save,
    use_paper_style,
)
from dcascade.race import STRATEGIES, build_race_tables

ROOT = Path(__file__).resolve().parents[1]
SML = dict(population_size=cfg.POPULATION, beta=cfg.BETA)


# --------------------------------------------------------------------------
# Figure 2: what a hand-off does to a specification
# --------------------------------------------------------------------------


def figure_transmission(race, chain, outdir: Path) -> None:
    fig, axes = new_figure(4.9, nrows=2, ncols=2)
    ax = axes[0, 0]
    kernel = handoff_kernel(chain.eps, chain.kernel)
    im = ax.imshow(kernel, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(4), STRATEGIES)
    ax.set_yticks(range(4), STRATEGIES)
    ax.set_xlabel("specification after the hand-off")
    ax.set_ylabel("specification before")
    ax.grid(False)
    for i in range(4):
        for j in range(4):
            v = kernel[i, j]
            if v > 1e-9:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=FS["tiny"], color="white" if v > 0.55 else "black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.ax.tick_params(labelsize=FS["tick"])
    panel_title(ax, "A", rf"hand-off kernel $M$, $\varepsilon={chain.eps:g}$")

    ax = axes[0, 1]
    depths = np.arange(chain.max_depth + 1)
    mix = np.stack([transmission(int(d), chain.eps, chain.kernel)[0] for d in depths])
    bottom = np.zeros_like(depths, dtype=float)
    for k, design in enumerate(STRATEGIES):
        ax.bar(depths, mix[:, k], bottom=bottom, width=0.72,
               color=PALETTE[design], label=design, edgecolor="white", linewidth=0.5)
        bottom += mix[:, k]
    ax.set_xlabel("delegation depth $d$")
    ax.set_ylabel("share of executions")
    ax.set_ylim(0, 1.22)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticks(depths)
    ax.grid(False)
    fitted_legend(ax, ncol=4, loc="upper center", handlelength=1.0,
                  columnspacing=0.7, handletextpad=0.35)
    panel_title(ax, "B", "what an $\\mathsf{AS}$ instruction becomes")

    ax = axes[1, 0]
    grid = np.linspace(0, chain.max_depth, 300)
    for eps in (0.05, 0.10, 0.20, 0.35):
        colour = plt.get_cmap("copper")(0.15 + 2.1 * eps)
        ax.plot(grid, (1.0 - eps) ** grid, color=colour,
                lw=1.9 if abs(eps - chain.eps) < 1e-9 else 1.2,
                label=rf"$\varepsilon={eps:g}$")
        half = specification_half_life(eps, chain.kernel)
        if half <= chain.max_depth:
            ax.plot([half], [0.5], marker="o", ms=3.4, color=colour, zorder=5)
    ax.axhline(0.5, color=PALETTE["neutral"], lw=0.7, ls=":")
    # the label sits just under the guide line, at the left end where every curve
    # is still close to one
    ax.text(0.38, 0.47, "half-life", ha="left", va="top",
            fontsize=FS["annot"], color=PALETTE["neutral"])
    ax.set_xlabel("delegation depth $d$")
    ax.set_ylabel(r"$\Pr[\text{intact}] = (1-\varepsilon)^d$")
    # a small margin above the data so the legend cannot sit on a curve
    ax.set_ylim(0, 1.18)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    fitted_legend(ax, ncol=2, loc="upper right", handlelength=1.4, columnspacing=0.9)
    panel_title(ax, "C", "the specification is forgotten geometrically")

    ax = axes[1, 1]
    fun = build_functionals(race, chain, cfg.LAM, cfg.HARM)
    prof = th.self_profile(fun, "AS")
    ax.plot(prof.depths, prof.harm, "-o", ms=3.4, color=PALETTE["realised"],
            label="harm caused, $m(d)$")
    ax.plot(prof.depths, prof.attributed, "-s", ms=3.4, color=PALETTE["attributed"],
            label=rf"harm attributed, $\phi^d m(d)$")
    ax.fill_between(prof.depths, prof.attributed, prof.harm, color=PALETTE["realised"],
                    alpha=0.12, lw=0)
    # the wedge is a narrow curved triangle with no room for a label inside it;
    # the two curves are named in the legend and the caption says what the shading
    # means, so nothing is annotated here
    ax.set_xlabel("delegation depth $d$")
    ax.set_ylabel("expected Unsafe actions")
    ax.set_xticks(prof.depths)
    ax.set_ylim(-0.10, prof.harm[-1] * 1.12)
    fitted_legend(ax, loc="upper left", handlelength=1.6)
    panel_title(ax, "D", rf"$\phi={chain.phi:g}$: the two harms diverge")

    save(fig, outdir / "fig02_transmission")


# --------------------------------------------------------------------------
# Figure 3: depth as a liability shelter
# --------------------------------------------------------------------------


def figure_shelter(race, chain, outdir: Path, probe: int = 30) -> None:
    fig, axes = new_figure(4.9, nrows=2, ncols=2)
    fun = build_functionals(race, chain, cfg.LAM, cfg.HARM)
    shelter = rb.deniability_threshold_profile(
        race, chain, "AS", probe_depth=probe, lam=cfg.LAM, harm=cfg.HARM
    )
    thresholds = shelter["thresholds"]

    ax = axes[0, 0]
    ax.plot(np.arange(len(thresholds)), thresholds, "-", color=PALETTE["CAS"], lw=1.6)
    ax.axhline(chain.phi, color=PALETTE["attributed"], lw=1.2, ls="--")
    ax.axhline(1.0, color=PALETTE["neutral"], lw=0.7, ls=":")
    onset = shelter["first_sheltering_depth"]
    if onset is not None:
        ax.axvline(onset, color=PALETTE["neutral"], lw=0.7, ls="--")
        ax.annotate(
            rf"shelter opens at $d={onset}$",
            xy=(onset, chain.phi), xytext=(onset + 2.4, 0.50),
            fontsize=FS["annot"], color=PALETTE["neutral"],
            arrowprops=dict(arrowstyle="->", lw=0.7, color=PALETTE["neutral"]),
        )
    ax.text(probe, chain.phi - 0.04, rf"$\phi={chain.phi:g}$", ha="right", va="top",
            fontsize=FS["annot"], color=PALETTE["attributed"])
    ax.set_xlabel("delegation depth $d$")
    ax.set_ylabel(r"$\phi^{*}(d) = m(d)/m(d{+}1)$")
    ax.set_ylim(0, 1.09)
    panel_title(ax, "A", "the deniability threshold rises to one")

    ax = axes[0, 1]
    harm = np.maximum(shelter["harm"], 1e-4)
    attributed = np.maximum(shelter["attributed"], 1e-4)
    ax.semilogy(np.arange(len(harm)), harm, color=PALETTE["realised"], lw=1.6,
                label="harm caused")
    ax.semilogy(np.arange(len(attributed)), attributed, color=PALETTE["attributed"],
                lw=1.6, label="harm attributed")
    ax.set_xlabel("delegation depth $d$")
    ax.set_ylabel("expected Unsafe actions")
    ax.set_ylim(1e-3, 40)
    # both curves climb steeply out of the bottom-left corner; the wide empty band
    # between them on the right is the only place a legend does not sit on a line
    fitted_legend(ax, loc="center right", handlelength=1.6)
    panel_title(ax, "B", "harm caused vs. harm attributed")

    ax = axes[1, 0]
    depths = np.arange(chain.max_depth)
    for intent in STRATEGIES[:3]:
        gains = np.array(
            [th.depth_invasion(fun, intent, int(d)).attributed_gain for d in depths]
        )
        ax.plot(depths, gains, "-o", ms=3.4, color=PALETTE[intent],
                label=rf"intent $\mathsf{{{intent}}}$")
    ax.axhline(0.0, color=PALETTE["neutral"], lw=0.9)
    lower = ax.get_ylim()[0]
    ax.axhspan(lower, 0.0, color=PALETTE["realised"], alpha=0.07, lw=0)
    ax.text(chain.max_depth - 1.05, 0.55 * lower, "liability rewards\none more hand-off",
            ha="right", va="center", fontsize=FS["annot"], color=PALETTE["neutral"])
    ax.set_ylim(lower, ax.get_ylim()[1])
    ax.set_xlabel("resident depth $d$")
    ax.set_ylabel(r"change in attributed harm")
    ax.set_xticks(depths)
    # only the first point of the CS curve rises above 0.2, so a single row of
    # entries fits in the strip along the top without touching any of them
    fitted_legend(ax, loc="upper right", ncol=3, handlelength=1.0,
                  columnspacing=0.7, handletextpad=0.35)
    panel_title(ax, "C", "the bill for one more hand-off")

    ax = axes[1, 1]
    prof = th.self_profile(fun, "AS")
    ax.plot(prof.depths, prof.private, "-o", ms=3.4, color=PALETTE["attributed"])
    ax.plot(prof.depths, prof.social, "-s", ms=3.4, color=PALETTE["realised"])
    private_best = int(prof.depths[int(np.argmax(prof.private))])
    social_best = int(prof.depths[int(np.argmax(prof.social))])
    outcome = round(th.equilibrium(fun, method="sml", **SML).mean_depth)
    top = float(max(prof.private.max(), prof.social.max()))
    bottom = float(min(prof.private.min(), prof.social.min()))
    span = top - bottom

    # the two curves separate widely on the right, so they are named where they
    # run rather than in a box that would have to sit on one of them
    ax.text(0.76 * chain.max_depth, prof.private.max() - 0.06 * span,
            r"private $\pi_P$", ha="left", va="bottom",
            fontsize=FS["annot"], color=PALETTE["attributed"])
    ax.text(0.82 * chain.max_depth, prof.social[social_best] * 0.55,
            r"social $\pi_S$", ha="left", va="bottom",
            fontsize=FS["annot"], color=PALETTE["realised"])

    # markers that land on the same depth share one entry, so that coincident
    # optima never print two strings on top of each other; the key is written as
    # one block in the empty lower-left corner rather than beside each line
    marks: dict[int, list[str]] = {}
    for x, label in (
        (private_best, "best for principals"),
        (social_best, "society"),
        (int(outcome), "where selection ends up"),
    ):
        marks.setdefault(int(x), []).append(label)
    for x in marks:
        ax.axvline(x, color=PALETTE["neutral"], lw=0.9, ls="--")
    key = "\n".join(
        rf"$d\!=\!{x}$: " + " and ".join(labels) for x, labels in sorted(marks.items())
    )
    ax.text(min(marks) + 0.12, bottom - 0.02 * span, key, ha="left", va="bottom",
            fontsize=FS["annot"], color=PALETTE["neutral"], linespacing=1.35)
    ax.set_xlabel("delegation depth $d$")
    ax.set_ylabel("payoff of a monomorphic population")
    ax.set_xticks(prof.depths)
    panel_title(ax, "D", "the selected depth suits nobody")

    save(fig, outdir / "fig03_shelter")


# --------------------------------------------------------------------------
# Figure 4: the two mechanisms, separately and together
# --------------------------------------------------------------------------


def figure_decomposition(race, chain, outdir: Path) -> None:
    dec = th.mechanism_decomposition(race, chain, cfg.LAM, cfg.HARM, method="sml", **SML)
    cells = [
        ("neither", dec.baseline, PALETTE["baseline"]),
        ("erosion\nonly", dec.drift_only, PALETTE["drift"]),
        ("per-layer\nonly", dec.deniability_only, PALETTE["deniability"]),
        ("both", dec.both, PALETTE["both"]),
    ]
    fig, axes = new_figure(4.9, nrows=2, ncols=2)

    ax = axes[0, 0]
    positions = np.arange(4)
    values = [c[1].unsafe_frequency for c in cells]
    ax.bar(positions, values, width=0.62, color=[c[2] for c in cells])
    for x, v in zip(positions, values):
        ax.text(x, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=FS["annot"])
    additive = values[1] + values[2] - values[0]
    ax.plot([2.58, 3.60], [additive, additive], color=PALETTE["neutral"], lw=1.1, ls="--")
    ax.annotate("", xy=(3.46, values[3]), xytext=(3.46, additive),
                arrowprops=dict(arrowstyle="<->", lw=0.8, color=PALETTE["neutral"]))
    ax.text(3.58, 0.5 * (values[3] + additive),
            f"interaction\n{dec.interaction:+.3f}", ha="left", va="center",
            fontsize=FS["annot"], color=PALETTE["neutral"])
    ax.text(3.58, additive, "additive\nprediction", ha="left", va="center",
            fontsize=FS["annot"], color=PALETTE["neutral"])
    ax.set_xticks(positions, [c[0] for c in cells])
    ax.set_ylabel("long-run Unsafe frequency $U$")
    ax.set_xlim(-0.62, 4.50)
    ax.set_ylim(0, max(values) * 1.24)
    panel_title(ax, "A", "neither mechanism does this alone")

    ax = axes[0, 1]
    depths = [c[1].mean_depth for c in cells]
    ax.bar(positions, depths, width=0.62, color=[c[2] for c in cells])
    for x, v in zip(positions, depths):
        ax.text(x, v + 0.10, f"{v:.2f}", ha="center", va="bottom", fontsize=FS["annot"])
    ax.axhline(chain.max_depth, color=PALETTE["neutral"], lw=0.8, ls=":")
    ax.text(1.0, chain.max_depth + 0.14, "ceiling", ha="center", va="bottom",
            fontsize=FS["annot"], color=PALETTE["neutral"])
    ax.set_xticks(positions, [c[0] for c in cells])
    ax.set_ylabel(r"mean delegation depth $\bar{d}$")
    ax.set_ylim(0, chain.max_depth * 1.22)
    panel_title(ax, "B", "erosion alone shortens the chain")

    ax = axes[1, 0]
    frozen = th.frozen_depth_counterfactual(
        race, chain, cfg.LAM, cfg.HARM, method="sml", **SML
    )
    steps = [
        ("pass-through", frozen["unsafe_passthrough"], PALETTE["drift"]),
        ("same designs,\nper-layer harm", frozen["unsafe_frozen_composition"], PALETTE["neutral"]),
        ("designs respond", frozen["unsafe_per_layer"], PALETTE["both"]),
    ]
    for k, (label, value, colour) in enumerate(steps):
        ax.bar(k, value, width=0.6, color=colour)
        ax.text(k, value + 0.012, f"{value:.3f}", ha="center", va="bottom",
                fontsize=FS["annot"])
    ax.set_xticks(range(3), [s[0] for s in steps])
    ax.set_ylabel("long-run Unsafe frequency $U$")
    ax.set_ylim(0, frozen["unsafe_per_layer"] * 1.34)
    ax.annotate(
        f"composition channel\n{frozen['composition_channel']:+.3f}",
        xy=(1.66, 0.55 * frozen["unsafe_per_layer"]),
        xytext=(0.90, 0.78 * frozen["unsafe_per_layer"]),
        ha="center", va="center", fontsize=FS["annot"], color=PALETTE["neutral"],
        arrowprops=dict(arrowstyle="->", lw=0.7, color=PALETTE["neutral"]),
    )
    panel_title(ax, "C", "the rise is composition, not behaviour")

    ax = axes[1, 1]
    declared = np.array([dec.both.intent_distribution[s] for s in STRATEGIES])
    executed = dec.both.executed_distribution
    width = 0.36
    positions = np.arange(4)
    ax.bar(positions - width / 2, declared, width=width, color=PALETTE["attributed"],
           label="declared by principals")
    ax.bar(positions + width / 2, executed, width=width, color=PALETTE["realised"],
           label="executed by agents")
    ax.set_xticks(positions, STRATEGIES)
    ax.set_ylabel("share of the population")
    ax.set_ylim(0, 1.30)
    ax.text(1.32, 0.60, f"declaration gap {dec.both.declaration_gap:.2f}",
            ha="left", va="bottom", fontsize=FS["annot"], color=PALETTE["neutral"])
    fitted_legend(ax, loc="upper right", handlelength=1.2, ncol=1)
    panel_title(ax, "D", "declared safety, executed risk")

    save(fig, outdir / "fig04_decomposition")


# --------------------------------------------------------------------------
# Figure 5: the attribution / erosion plane
# --------------------------------------------------------------------------


def figure_plane(race, chain, outdir: Path, n: int = 21) -> None:
    grid = rb.equilibrium_grid(
        race,
        chain,
        phis=np.linspace(0.6, 1.0, n),
        epsilons=np.linspace(0.0, 0.4, n),
        lam=cfg.LAM,
        harm=cfg.HARM,
        method="sml",
        **SML,
    )
    extent = [grid["epsilons"][0], grid["epsilons"][-1], grid["phis"][0], grid["phis"][-1]]
    panels = [
        ("A", grid["unsafe"], "long-run Unsafe frequency $U$", "magma_r", None),
        ("B", grid["depth"], r"mean delegation depth $\bar{d}$", "viridis", None),
        ("C", grid["declaration_gap"], "declaration gap", "cividis", None),
        ("D", grid["interaction"], "interaction index", "coolwarm", "symmetric"),
    ]
    fig, axes = new_figure(4.9, nrows=2, ncols=2)
    for (letter, data, title, cmap, scale), ax in zip(panels, axes.ravel()):
        kwargs = {}
        if scale == "symmetric":
            span = float(np.abs(data).max())
            kwargs = dict(vmin=-span, vmax=span)
        im = ax.imshow(data, origin="lower", extent=extent, aspect="auto", cmap=cmap, **kwargs)
        ax.plot([chain.eps], [chain.phi], marker="*", ms=8, color="white",
                markeredgecolor="black", markeredgewidth=0.5, zorder=5)
        ax.set_xlabel(r"erosion probability $\varepsilon$")
        ax.set_ylabel(r"attribution retention $\phi$")
        ax.grid(False)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cbar.ax.tick_params(labelsize=FS["tick"])
        panel_title(ax, letter, title)
    save(fig, outdir / "fig05_plane")


# --------------------------------------------------------------------------
# Figure 6: what the four instruments do
# --------------------------------------------------------------------------


def figure_instruments(race, chain, outdir: Path, quick: bool = False) -> None:
    fig, axes = new_figure(4.9, nrows=2, ncols=2)
    n = 13 if quick else 26

    ax = axes[0, 0]
    sweep = iv.pass_through_sweep(
        race, chain, np.linspace(chain.phi, 1.0, n), cfg.LAM, cfg.HARM, "sml", **SML
    )
    settings = np.array([o.setting for o in sweep])
    unsafe = np.array([o.unsafe_frequency for o in sweep])
    depth = np.array([o.mean_depth for o in sweep])
    ax.plot(settings, unsafe, "-o", ms=3.0, color=PALETTE["realised"], label="$U$")
    twin = ax.twinx()
    twin.plot(settings, depth, "-s", ms=3.0, color=PALETTE["attributed"],
              label=r"$\bar{d}$")
    twin.set_ylabel(r"mean depth $\bar{d}$", color=PALETTE["attributed"])
    twin.tick_params(axis="y", labelsize=FS["tick"], colors=PALETTE["attributed"])
    twin.grid(False)
    twin.spines["right"].set_visible(True)
    twin.spines["right"].set_color(PALETTE["attributed"])
    ax.set_xlabel(r"attribution retention $\phi$")
    ax.set_ylabel("Unsafe frequency $U$", color=PALETTE["realised"])
    ax.tick_params(axis="y", labelsize=FS["tick"], colors=PALETTE["realised"])
    handles = [Line2D([], [], color=PALETTE["realised"], marker="o", ms=3.0, label="$U$"),
               Line2D([], [], color=PALETTE["attributed"], marker="s", ms=3.0,
                      label=r"$\bar{d}$")]
    fitted_legend(ax, handles=handles, loc="upper right", ncol=2, handlelength=1.4)
    panel_title(ax, "A", "attribution: a threshold, not a dial")

    ax = axes[0, 1]
    factors = np.linspace(1.0, 0.0, n)
    sweep = iv.attestation_sweep(race, chain, factors, cfg.LAM, cfg.HARM, "sml", **SML)
    x = np.array([o.setting for o in sweep]) * chain.eps
    y = np.array([o.unsafe_frequency for o in sweep])
    ax.plot(x, y, "-o", ms=3.0, color=PALETTE["drift"])
    # x runs from the untreated erosion probability down to zero, so a rise in y
    # marks a range where making every hand-off more faithful made things worse
    rises = np.flatnonzero(np.diff(y) > 1e-6)
    if rises.size:
        peak = int(rises[0])
        while peak + 1 < len(y) and y[peak + 1] >= y[peak]:
            peak += 1
        ax.annotate("more faithful,\nless safe", xy=(x[peak], y[peak]),
                    xytext=(x[peak] + 0.055, y[peak] + 0.035), ha="left", va="bottom",
                    fontsize=FS["annot"], color=PALETTE["neutral"],
                    arrowprops=dict(arrowstyle="->", lw=0.7, color=PALETTE["neutral"]))
    ax.set_xlabel(r"erosion probability after the improvement, $\kappa\varepsilon$")
    ax.set_ylabel("Unsafe frequency $U$")
    panel_title(ax, "B", "fidelity is not monotone")

    ax = axes[1, 0]
    for strength, marker, style in ((1.0, "o", "-"), (0.5, "s", "--")):
        sweep = iv.audit_placement_sweep(
            race, chain, strength, cfg.LAM, cfg.HARM, "sml", **SML
        )
        ax.plot([o.setting for o in sweep], [o.unsafe_frequency for o in sweep],
                style, marker=marker, ms=3.2, color=PALETTE["CAS"] if strength == 1.0
                else PALETTE["attributed"], label=rf"$\alpha={strength:g}$")
    ax.set_xlabel("layer at which the check is placed, $k$")
    ax.set_ylabel("Unsafe frequency $U$")
    ax.set_xticks(range(chain.max_depth + 1))
    low, high = ax.get_ylim()
    ax.set_ylim(low - 0.16 * (high - low), high)
    ax.text(0.04, 0.015, "at the principal", transform=ax.transAxes,
            fontsize=FS["annot"], color=PALETTE["neutral"], ha="left")
    ax.text(0.96, 0.015, "at the agent that acts", transform=ax.transAxes,
            fontsize=FS["annot"], color=PALETTE["neutral"], ha="right")
    fitted_legend(ax, loc="upper right", handlelength=1.8)
    panel_title(ax, "C", "check late, not early")

    ax = axes[1, 1]
    caps = np.arange(chain.max_depth, -1, -1)
    sweep = iv.depth_cap_sweep(race, chain, caps, cfg.LAM, cfg.HARM, "sml", **SML)
    xs = np.array([o.setting for o in sweep])
    ax.plot(xs, [o.unsafe_frequency for o in sweep], "-o", ms=3.2,
            color=PALETTE["realised"], label="$U$")
    twin = ax.twinx()
    twin.plot(xs, [o.social_payoff for o in sweep], "-s", ms=3.2,
              color=PALETTE["safe"], label=r"$\pi_S$")
    twin.set_ylabel(r"social payoff $\pi_S$", color=PALETTE["safe"])
    twin.tick_params(axis="y", labelsize=FS["tick"], colors=PALETTE["safe"])
    twin.grid(False)
    twin.spines["right"].set_visible(True)
    twin.spines["right"].set_color(PALETTE["safe"])
    ax.set_xlabel(r"permitted chain length $\bar{D}$")
    ax.set_ylabel("Unsafe frequency $U$", color=PALETTE["realised"])
    ax.tick_params(axis="y", labelsize=FS["tick"], colors=PALETTE["realised"])
    ax.set_xticks(range(chain.max_depth + 1))
    handles = [Line2D([], [], color=PALETTE["realised"], marker="o", ms=3.2, label="$U$"),
               Line2D([], [], color=PALETTE["safe"], marker="s", ms=3.2, label=r"$\pi_S$")]
    fitted_legend(ax, handles=handles, loc="center left", ncol=1, handlelength=1.4)
    panel_title(ax, "D", "a ceiling buys both at once")

    save(fig, outdir / "fig06_instruments")


# --------------------------------------------------------------------------
# Figure 7: what each instrument costs
# --------------------------------------------------------------------------


def figure_frontier(race, chain, outdir: Path, quick: bool = False) -> None:
    n = 11 if quick else 21
    sweeps = {
        "depth ceiling": (
            iv.depth_cap_sweep(race, chain, np.arange(chain.max_depth, -1, -1),
                               cfg.LAM, cfg.HARM, "sml", **SML),
            PALETTE["AS"], "o",
        ),
        "fidelity": (
            iv.attestation_sweep(race, chain, np.linspace(1.0, 0.0, n),
                                 cfg.LAM, cfg.HARM, "sml", **SML),
            PALETTE["CS"], "s",
        ),
        "attribution": (
            iv.pass_through_sweep(race, chain, np.linspace(chain.phi, 1.0, n),
                                  cfg.LAM, cfg.HARM, "sml", **SML),
            PALETTE["CAS"], "^",
        ),
        "liability level": (
            iv.liability_sweep(race, chain, np.geomspace(cfg.effective_liability(), 200.0, n),
                               cfg.LAM, cfg.HARM, "sml", **SML),
            PALETTE["AU"], "v",
        ),
    }
    fig, axes = new_figure(2.7, nrows=1, ncols=2)

    ax = axes[0]
    for name, (outcomes, colour, marker) in sweeps.items():
        social, unsafe = iv.instrument_frontier(outcomes)
        ax.plot(unsafe, social, marker=marker, ms=3.0, lw=1.2, color=colour, label=name)
    baseline = sweeps["depth ceiling"][0][0]
    ax.plot([baseline.unsafe_frequency], [baseline.social_payoff], marker="*", ms=9,
            color="white", markeredgecolor="black", markeredgewidth=0.6, zorder=6)
    # every frontier converges on the untreated point, so there is no room beside
    # it for a label; the caption identifies the star instead
    ax.set_xlabel("long-run Unsafe frequency $U$")
    ax.set_ylabel(r"social payoff $\pi_S$")
    fitted_legend(ax, loc="lower left", handlelength=1.6, ncol=2, columnspacing=0.9)
    panel_title(ax, "A", "safety bought, welfare kept")

    ax = axes[1]
    matched = iv.matched_effort_comparison(
        race, chain, 0.05, cfg.LAM, cfg.HARM, "sml", **SML
    )
    # the JSON keys are historical; the labels are the manuscript's own names
    names = ["depth_cap", "attestation", "pass_through", "liability"]
    pretty = ["depth\nceiling", "fidelity", "attribution", "liability\nlevel"]
    colours = [PALETTE["AS"], PALETTE["CS"], PALETTE["CAS"], PALETTE["AU"]]
    values = [matched[k]["social_payoff"] for k in names]
    ax.bar(range(len(names)), values, width=0.62, color=colours)
    for k, name in enumerate(names):
        ax.text(k, values[k] + 1.0, f"{values[k]:.1f}", ha="center", va="bottom",
                fontsize=FS["annot"])
    ax.axhline(baseline.social_payoff, color=PALETTE["neutral"], lw=0.9, ls="--")
    # the bars start at zero, so the band under the reference line is the only
    # place the label does not sit on one
    ax.text(-0.42, baseline.social_payoff - 1.0, "no intervention",
            ha="left", va="top", fontsize=FS["annot"], color=PALETTE["neutral"])
    ax.set_xticks(range(len(names)), pretty)
    ax.set_ylabel(r"social payoff $\pi_S$ at $U \leq 0.05$")
    ax.set_ylim(min(0.0, baseline.social_payoff) - 5.0, max(values) * 1.15)
    panel_title(ax, "B", "cheapest route to the target")

    save(fig, outdir / "fig07_frontier")


# --------------------------------------------------------------------------
# Figure 8: robustness
# --------------------------------------------------------------------------


def figure_robustness(race, chain, outdir: Path, quick: bool = False) -> None:
    fig, axes = new_figure(4.9, nrows=2, ncols=2)

    ax = axes[0, 0]
    cells = []
    cells += rb.sweep_interaction_layer(
        chain, cfg.LAM, cfg.HARM,
        prizes=(100.0,) if quick else (50.0, 100.0, 200.0),
        risks=(0.3, 0.6) if quick else (0.1, 0.3, 0.6, 0.9),
        method="sml", **SML,
    )
    cells += rb.sweep_setback_scope(chain, cfg.LAM, cfg.HARM, "sml", **SML)
    cells += rb.sweep_organisation(race, chain, cfg.LAM, cfg.HARM, method="sml", **SML)
    cells += rb.sweep_process(race, chain, cfg.LAM, cfg.HARM)
    groups: dict[str, list[float]] = {}
    for cell in cells:
        groups.setdefault(cell.label, []).append(cell.interaction)
    labels = list(groups)
    pretty = {
        "interaction_layer": "prize, risk",
        "setback_scope": "setback rule",
        "gain": "benefit $g$",
        "curvature": "coordination cost",
        "population": "population $Z$",
        "selection": r"selection $\beta$",
        "depth_ceiling": "ceiling $D$",
        "dynamics": "dynamics",
    }
    for k, label in enumerate(labels):
        values = groups[label]
        ax.scatter(values, np.full(len(values), k), s=13,
                   color=PALETTE["accent"], zorder=3, edgecolor="none")
    ax.axvline(0.0, color=PALETTE["neutral"], lw=0.8)
    ax.set_yticks(range(len(labels)), [pretty.get(l, l) for l in labels])
    ax.set_xlabel("interaction index")
    ax.set_ylim(-0.6, len(labels) - 0.4)
    panel_title(ax, "A", "the interaction across every sweep")

    ax = axes[0, 1]
    kernels = rb.sweep_kernel(race, chain, cfg.LAM, cfg.HARM, "sml", **SML)
    names = [c.setting for c in kernels]
    width = 0.26
    positions = np.arange(len(names))
    series = [
        ("neither", [c.baseline for c in kernels], PALETTE["baseline"]),
        ("erosion only", [c.drift_only for c in kernels], PALETTE["drift"]),
        ("both", [c.both for c in kernels], PALETTE["both"]),
    ]
    for k, (label, values, colour) in enumerate(series):
        ax.bar(positions + (k - 1) * width, values, width=width, color=colour, label=label)
    ax.set_xticks(positions, names)
    ax.set_ylabel("long-run Unsafe frequency $U$")
    ceiling = max(max(s[1]) for s in series) * 1.34
    ax.set_ylim(0, ceiling)
    for k, cell in enumerate(kernels):
        if cell.both < 0.02 * ceiling:
            ax.annotate("chains never form", xy=(positions[k] + width, 0.012 * ceiling),
                        xytext=(positions[k], 0.30 * ceiling), ha="center", va="bottom",
                        fontsize=FS["annot"], color=PALETTE["neutral"],
                        arrowprops=dict(arrowstyle="->", lw=0.7, color=PALETTE["neutral"]))
    fitted_legend(ax, ncol=3, loc="upper center", handlelength=1.1, columnspacing=0.8)
    panel_title(ax, "B", "gradual erosion is the danger")

    ax = axes[1, 0]
    rows = rb.sml_versus_full_chain(
        race, chain, "AS", cfg.LAM, cfg.HARM, population_size=60, beta=cfg.BETA,
        depths=(0, chain.max_depth // 2, chain.max_depth),
    )
    mus = np.array([r["mu"] for r in rows[1:]])
    values = np.array([r["unsafe_frequency"] for r in rows[1:]])
    ax.semilogx(mus, values, "-o", ms=3.4, color=PALETTE["CAS"],
                label="full mutation-selection chain")
    ax.axhline(rows[0]["unsafe_frequency"], color=PALETTE["attributed"], lw=1.2, ls="--",
               label="small-mutation limit")
    ax.set_xlabel(r"mutation rate $\mu$")
    ax.set_ylabel("Unsafe frequency $U$")
    # the curve descends from left to right, so it leaves the lower-left corner
    # free while the lower-right corner is where it ends
    fitted_legend(ax, loc="lower left", handlelength=1.8)
    panel_title(ax, "C", "the small-mutation reduction")

    ax = axes[1, 1]
    liabilities = np.geomspace(2.0, 400.0, 15 if quick else 30)
    for phi in (0.75, 0.85, 0.95, 1.0):
        values = []
        for L in liabilities:
            fun = build_functionals(
                race, replace(chain, phi=phi), float(L) / cfg.HARM, cfg.HARM
            )
            values.append(th.equilibrium(fun, method="sml", **SML).unsafe_frequency)
        ax.semilogx(liabilities, values, "-", lw=1.5,
                    color=depth_colour(int(round((phi - 0.75) / 0.25 * 3)), 3),
                    label=rf"$\phi={phi:g}$")
    ax.set_xlabel("effective liability $L$")
    ax.set_ylabel("Unsafe frequency $U$")
    fitted_legend(ax, loc="upper right", ncol=2, handlelength=1.5, columnspacing=0.9)
    panel_title(ax, "D", "liability is not a substitute")

    save(fig, outdir / "fig08_robustness")


FIGURES = {
    2: figure_transmission,
    3: figure_shelter,
    4: figure_decomposition,
    5: figure_plane,
    6: figure_instruments,
    7: figure_frontier,
    8: figure_robustness,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=ROOT / "results" / "figures")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--only", type=int, nargs="*", default=None)
    args = parser.parse_args()

    use_paper_style()
    args.outdir.mkdir(parents=True, exist_ok=True)
    race = build_race_tables(cfg.RACE)
    chain = cfg.CHAIN

    wanted = args.only or sorted(FIGURES)
    for number in wanted:
        builder = FIGURES[number]
        kwargs = {}
        if number in (6, 7, 8):
            kwargs["quick"] = args.quick
        if number == 5:
            kwargs["n"] = 9 if args.quick else 21
        if number == 3:
            kwargs["probe"] = 30
        builder(race, chain, args.outdir, **kwargs)
        print(f"  figure {number}: {builder.__name__}")
    print(f"wrote {len(wanted)} figures to {args.outdir}")


if __name__ == "__main__":
    main()
