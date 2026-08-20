# Delegation Cascades in an Evolutionary AI Race

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![EGTtools](https://img.shields.io/badge/built%20with-EGTtools-brightgreen.svg)](https://github.com/Socrats/EGTTools)
[![tests](https://img.shields.io/badge/tests-185%20passing-success.svg)](tests/)

Reproduction code for a study of what happens to a safety instruction when it is
passed down a chain of delegated agents before anything is done with it.

A principal enters a competitive race but never acts in it. It issues a
specification to an agent, which issues one to a sub-agent, and so on for `d`
hand-offs. Two things happen along the chain and both are geometric in `d`. The
specification is transmitted imperfectly, so it survives `d` hand-offs intact
with probability `(1-eps)^d`. Responsibility is traced back only in part, so a
principal at depth `d` is charged the fraction `phi^d` of the harm it causes.
The first effect makes depth expensive; the second makes it cheap. Which one
wins, and what a population of principals does about it, is what the model
answers.

The interaction layer is the repeated two-player AI race of
Fernández Domingos & Han (2026) with the reduced strategy set AS / CS / CAS / AU,
used **without modification**, so that every depth effect is attributable to the
delegation layer built on top of it.

## Headline results

| | Result |
|---|---|
| **1** | **Depth is an unbounded liability shelter.** Realised harm saturates as the specification is forgotten while attributed harm carries the extra factor `phi` per layer, so for every `phi < 1` there is a depth beyond which each further hand-off strictly lowers the bill and strictly raises the damage. Probing to depth 40 puts the deniability threshold `phi* = m(d)/m(d+1)` at `0.99993`. Beyond that depth, raising liability makes the deeper design *more* attractive. |
| **2** | **Neither mechanism is dangerous alone.** Starting from a race liability has already made safe (`L = 20`, which is `3.1x` the level at which the depth-zero game reaches unsafe frequency zero), erosion alone gives `0.018` and per-layer attribution alone gives `0.012`, but together they give `0.322`. The interaction, `+0.293`, is **91% of the joint effect**. |
| **3** | **The interaction is a lost response, not worse behaviour.** Under pass-through the population answers erosion by shortening its chains from 6.00 to 1.99 hand-offs. Scoring that same population with the per-layer bill changes the unsafe frequency by less than `1e-9`; the whole of the remaining `+0.305` is which designs the population comes to contain. |
| **4** | **Declared safety decouples from executed safety.** In equilibrium every principal issues `AS`, the safest available instruction, and the population executes `AS` only 26% of the time. Declared and executed behaviour differ by `0.74` in total variation. |
| **5** | **Attribution is a threshold, not a dial.** Nothing happens until `phi = 0.86`, close to the closed form `(L_c/L)^(1/D) = 0.830`; at `phi = 0.85`, which returns 38% of responsibility over six layers, the ecosystem is unchanged. |
| **6** | **Fidelity instruments have a perverse range.** Attestation improves matters down to `eps = 0.07` (`U = 0.032`) and then reverses: at `eps = 0.04`, `U = 0.239`. Over-specification was the equilibrium response to erosion, and a partial fix removes it. The same reversal appears in audit placement at full strength. |
| **7** | **Check late, and cap the depth.** The same check is worth `18x` more at the agent that acts than at the principal. A ceiling of one hand-off removes the unsafe behaviour and raises social payoff from `-0.8` to `60.5`; it is the only instrument that is monotone and improves both objectives together. |
| **8** | **Graceful erosion is the dangerous kind.** Replacing graded erosion with catastrophic failure (`collapse`) drops the joint cell to `3e-5` and mean depth to `0.0002`: chains that fail loudly are not built. Undirected noise (`uniform`) is *worse* than graded erosion (`0.517`), so the effect is not about the direction of the slippage. |

Across 43 robustness cells the interaction is positive in 88%, with median
`+0.292` and interquartile range `[0.242, 0.297]`; mean delegation depth is
higher under per-layer attribution than under pass-through in **every** cell.

## Installation

```bash
git clone https://github.com/trungkiet2005/delegation-cascade.git
cd delegation-cascade
pip install -e ".[dev]"
```

Requires Python ≥ 3.10, `numpy`, `scipy`, `matplotlib`, `pandas` and
[`egttools`](https://github.com/Socrats/EGTTools) ≥ 0.1.14.

## Reproducing the paper

```bash
python scripts/run_analysis.py     # tables + every number quoted in the text
python scripts/run_robustness.py   # 43-cell sweep, kernels, the (phi, eps) plane
python scripts/make_figures.py     # Figures 2-8
python scripts/build_paper.py      # stage figures, run pdflatex/bibtex
pytest                             # 185 checks of the analytics
```

`scripts/make_figures.py --quick` runs a coarser grid in about a minute, and
`--only 4 5` rebuilds a subset. Every figure is saved at the same width
(`dcascade.plotting.FIG_WIDTH`) with an uncropped bounding box and is included at
`\linewidth`, so all figures are reduced by the same factor on the page and their
text renders at the same size; `tests/test_figures.py` enforces this. Everything
is deterministic: the only randomness is the choice of initial conditions for
basin averaging, which is seeded.

Outputs:

```text
results/key_numbers.json         every scalar quoted in the manuscript
results/robustness_summary.json  43-cell sweep, kernel comparison, grid summary
results/grid.npz                 the (phi, eps) plane
results/tables/*.csv             transmission, profiles, thresholds, instruments
results/tables/tables.tex        LaTeX tables included by the manuscript
results/figures/fig0*.pdf        publication figures
paper/main.pdf                   the compiled manuscript (built here, not committed:
                                 the journals are told the paper has not appeared
                                 anywhere, so no PDF of it is published in this repo)
```

## Package layout

```text
src/dcascade/
  race.py           exact evaluation of the reduced race over the horizon law
  chain.py          hand-off kernels, transmission laws, half-life, audits
  functionals.py    task, private and social functionals; the dilution wedge
  dynamics.py       replicator, closed-form Fermi fixation, stationary regimes
  theory.py         deniability thresholds, the shelter, invasion, the 2x2
  interventions.py  the four instruments and the matched-effort comparison
  robustness.py     sweeps across every free parameter, and the reduction checks
  config.py         the one baseline every script and test reads
  plotting.py       figure style: one saved width and one font scale for all
```

### Two notes on the numerics

**The interaction layer is evaluated exactly.** The four reduced designs are
deterministic, so the action path of an ordered pair is fixed once the pair is
fixed and the only stochastic primitive is the horizon `T`. Every matchup is
therefore evaluated by exact expectation over the horizon distribution instead of
Monte Carlo sampling, which removes simulation noise from the payoff matrix
entirely. `tests/test_race.py` checks the exact values against a 200 000-draw
simulation.

**Fixation probabilities are computed in closed form.** The joint design space
has `4(D+1)` designs -- 28 at the baseline ceiling -- and the generic EGTtools
route is organised around the full population state space, whose size grows like
`C(Z+n-1, n-1)`. We evaluate the birth-death chain of the Fermi process directly
instead, with the dominant term factored out so that strongly disadvantaged
mutants do not underflow. `tests/test_dynamics.py` checks the result against
EGTtools on small games, and records one case where EGTtools truncates a
fixation probability of `3.29e-8` to exactly zero while the closed form agrees
with a 60-digit evaluation to ten significant figures.

## Relation to the companion study

This is the second of two studies built on the same interaction layer. The first
asks what happens when the payoff that governs replication differs from the
payoff earned in the game ([*deployment-layer selection*](https://github.com/trungkiet2005/deployment-layer-selection)); this one asks what
happens when the party being selected is several hand-offs away from the party
that acts. The two share the race engine and the audit discipline, and they
agree on the depth-zero benchmark: the liability at which `CAS` stops invading
`AS` in the single-layer game is `42.77` in both.

## Citing

If you use this code, please cite both this repository and the toolbox it is
built on:

> Fernández Domingos, E., Santos, F. C. & Lenaerts, T. *EGTtools: Evolutionary
> game dynamics in Python.* iScience **26**, 106419 (2023).

The interaction layer is adapted from:

> Fernández Domingos, E. & Han, T. A. *Falling Behind Drives Unsafe Development
> in an Idealised AI Race Experiment.* arXiv:2607.26034 (2026).

## License

MIT — see [LICENSE](LICENSE).
