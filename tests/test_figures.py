"""Checks on the figure style.

The manuscript includes every figure at ``\\linewidth``, so two figures saved at
different widths are reduced by different factors and their text ends up at
different sizes on the page.  These tests pin the two things that guarantee a
uniform reduction: one saved width for all figures, and one font scale.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from dcascade.plotting import FIG_WIDTH, FS

FIGDIR = Path(__file__).resolve().parents[1] / "results" / "figures"
MEDIABOX = re.compile(rb"/MediaBox\s*\[([^\]]*)\]")


def _width_inches(pdf: Path) -> float:
    match = MEDIABOX.search(pdf.read_bytes())
    assert match is not None, f"no MediaBox in {pdf.name}"
    box = [float(v) for v in match.group(1).split()]
    return (box[2] - box[0]) / 72.0


@pytest.mark.parametrize("pdf", sorted(FIGDIR.glob("fig*.pdf")) or [None])
def test_every_figure_has_the_standard_width(pdf: Path | None) -> None:
    if pdf is None:
        pytest.skip("figures have not been generated yet")
    assert _width_inches(pdf) == pytest.approx(FIG_WIDTH, abs=0.02)


def test_font_scale_is_ordered_and_legible() -> None:
    # the smallest text must stay above 6 pt once reduced onto the page
    text_width = 6.268  # inches, article a4paper with 1 in margins
    reduction = text_width / FIG_WIDTH
    assert min(FS.values()) * reduction > 6.0
    assert FS["tiny"] <= FS["annot"] <= FS["legend"] <= FS["title"]
    assert FS["tick"] <= FS["label"]
