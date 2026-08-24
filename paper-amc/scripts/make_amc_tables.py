"""Stage generated figures and split tables into CAS-ready submission files.

The analysis scripts of the repository emit ``paper/tables_generated.tex``,
``paper/regimes_generated.tex`` and ``paper/robustness_generated.tex``, each
holding several tables.  The AMC manuscript uses a subset of them and places
them individually, so this script writes one file per table.

Two mechanical adaptations are applied and nothing else; no number, caption or
column is touched:

* ``\\begin{table}[t]`` becomes ``\\begin{table}[pos=t]``, because the Elsevier
  CAS classes give the ``table`` and ``figure`` environments a key--value
  optional argument rather than a float specifier;
* a provenance header is prepended.

The seven publication figures are copied directly from ``results/figures`` so
the AMC package cannot retain an older figure after the analysis is rerun.

    python paper-amc/scripts/make_amc_tables.py
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "paper"
DST = ROOT / "paper-amc"

SOURCES = (
    "tables_generated.tex",
    "regimes_generated.tex",
    "robustness_generated.tex",
)

#: Emitted by ``paper-amc/scripts/run_numerics.py`` into ``paper-amc`` itself
#: rather than into ``paper``, and split here for the same reason.  Leaving it
#: out is how ``tab_cost`` and ``tab_verification`` went stale against their
#: own generator once already.
LOCAL_SOURCES = ("numerics_generated.tex",)

HEADER = (
    "%% one table per file, split from {where}/{source} by\n"
    "%% paper-amc/scripts/make_amc_tables.py -- do not edit\n"
)

FIGURES = (
    "fig02_transmission.pdf",
    "fig03_shelter.pdf",
    "fig04_decomposition.pdf",
    "fig05_plane.pdf",
    "fig06_instruments.pdf",
    "fig07_frontier.pdf",
    "fig08_robustness.pdf",
)


def _split(text: str, header: str) -> None:
    for block in re.findall(r"\\begin\{table\}.*?\\end\{table\}", text, flags=re.S):
        key = re.search(r"\\label\{tab:([A-Za-z]+)\}", block).group(1)
        block = block.replace(r"\begin{table}[t]", r"\begin{table}[pos=t]")
        out = DST / ("tab_%s.tex" % key)
        out.write_text(header + block + "\n", encoding="utf-8")
        print(out.relative_to(ROOT))


def main() -> None:
    for source in SOURCES:
        _split(
            (SRC / source).read_text(encoding="utf-8"),
            HEADER.format(where="paper", source=source),
        )
    for source in LOCAL_SOURCES:
        _split(
            (DST / source).read_text(encoding="utf-8"),
            HEADER.format(where="paper-amc", source=source),
        )
    figure_dir = DST / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for name in FIGURES:
        source = ROOT / "results" / "figures" / name
        if not source.is_file():
            raise SystemExit(f"missing {source.relative_to(ROOT)}; run scripts/make_figures.py")
        shutil.copy2(source, figure_dir / name)
        print((figure_dir / name).relative_to(ROOT))


if __name__ == "__main__":
    main()
