"""Split the study's generated table files into one CAS-ready file per table.

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

    python paper-amc/scripts/make_amc_tables.py
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "paper"
DST = ROOT / "paper-amc"

SOURCES = (
    "tables_generated.tex",
    "regimes_generated.tex",
    "robustness_generated.tex",
)

HEADER = (
    "%% one table per file, split from paper/{source} by\n"
    "%% paper-amc/scripts/make_amc_tables.py -- do not edit\n"
)


def main() -> None:
    for source in SOURCES:
        text = (SRC / source).read_text(encoding="utf-8")
        for block in re.findall(r"\\begin\{table\}.*?\\end\{table\}", text, flags=re.S):
            key = re.search(r"\\label\{tab:([A-Za-z]+)\}", block).group(1)
            block = block.replace(r"\begin{table}[t]", r"\begin{table}[pos=t]")
            out = DST / ("tab_%s.tex" % key)
            out.write_text(
                HEADER.format(source=source) + block + "\n", encoding="utf-8"
            )
            print(out.relative_to(ROOT))


if __name__ == "__main__":
    main()
