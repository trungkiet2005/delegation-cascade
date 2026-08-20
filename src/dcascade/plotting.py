r"""Publication figure style and reusable plotting helpers.

Every figure in the manuscript is saved at exactly :data:`FIG_WIDTH` inches and
included at ``\linewidth``, so all figures are reduced by the same factor on the
page.  Font sizes are therefore comparable across figures and are drawn from the
single scale in :data:`FS` rather than chosen per panel.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

#: Width in inches of every saved figure.  The text block of the manuscript is
#: 6.27 in, so figures are reduced by 0.91 uniformly.
FIG_WIDTH = 6.9

#: The only font sizes used anywhere.  Values are points before the uniform
#: 0.91 reduction, so the smallest text on the page is about 6.4 pt.
FS = {
    "title": 9.5,   # panel titles
    "label": 9.0,   # axis labels, simplex vertex labels
    "tick": 8.0,    # tick labels
    "legend": 8.0,  # legend entries
    "annot": 7.5,   # in-axes annotations
    "tiny": 7.0,    # dense in-axes annotations (cell values, threshold marks)
}

#: Colour-blind-safe qualitative palette (Okabe-Ito), ordered along the erosion
#: order so that a legend reads as a gradient from safe to unsafe.
PALETTE = {
    "AS": "#0072B2",
    "CS": "#009E73",
    "CAS": "#E69F00",
    "AU": "#D55E00",
    "accent": "#CC79A7",
    "neutral": "#4D4D4D",
    "grid": "#D9D9D9",
    "unsafe": "#B2182B",
    "safe": "#2166AC",
    "realised": "#B2182B",
    "attributed": "#2166AC",
    "drift": "#009E73",
    "deniability": "#CC79A7",
    "both": "#B2182B",
    "baseline": "#4D4D4D",
}

STRATEGY_LABEL = {
    "AS": "AS (always safe)",
    "CS": "CS (conditionally safe)",
    "CAS": "CAS (conditionally unsafe)",
    "AU": "AU (always unsafe)",
}

#: Short glosses used in legends where the full label does not fit.
STRATEGY_SHORT = {
    "AS": "AS",
    "CS": "CS",
    "CAS": "CAS",
    "AU": "AU",
}

CELL_LABEL = {
    "baseline": "neither",
    "drift": "erosion only",
    "deniability": "per-layer only",
    "both": "both",
}


def depth_colour(depth: int, max_depth: int):
    """Sequential colour for a delegation depth."""
    import matplotlib.pyplot as _plt

    return _plt.get_cmap("viridis")(0.08 + 0.84 * depth / max(max_depth, 1))


def use_paper_style() -> None:
    """Apply the manuscript figure style."""
    mpl.rcParams.update(
        {
            "figure.dpi": 160,
            # matplotlib always emits an ``imshow`` panel as a raster XObject, so
            # the heatmap figures are combination art and have to clear the
            # 500 dpi minimum after the on-page reduction, which 400 does not
            "savefig.dpi": 600,
            # embed real fonts rather than Type 3 bitmaps
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            # a fixed saved size is what keeps the on-page scale uniform, so the
            # bounding box must not be cropped to the ink
            "savefig.bbox": None,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "STIXGeneral"],
            "mathtext.fontset": "stix",
            "font.size": FS["label"],
            "axes.titlesize": FS["title"],
            "axes.labelsize": FS["label"],
            "legend.fontsize": FS["legend"],
            "xtick.labelsize": FS["tick"],
            "ytick.labelsize": FS["tick"],
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": PALETTE["grid"],
            "grid.linewidth": 0.5,
            "grid.alpha": 0.8,
            "lines.linewidth": 1.5,
            "legend.frameon": False,
            "legend.handlelength": 2.0,
            "legend.columnspacing": 1.1,
            "legend.borderaxespad": 0.2,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 2.6,
            "ytick.major.size": 2.6,
            "xtick.major.pad": 2.0,
            "ytick.major.pad": 2.0,
            "axes.labelpad": 2.5,
        }
    )


def new_figure(height: float, *, nrows: int = 1, ncols: int = 1, **kwargs):
    """A constrained-layout figure of the standard width.

    Constrained layout fits the decorations inside a *fixed* canvas, which is
    what lets every figure keep the same saved width.
    """
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(FIG_WIDTH, height), layout="constrained", **kwargs
    )
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.03, hspace=0.03)
    return fig, axes


def save(fig: plt.Figure, path: Path | str, also_png: bool = True) -> None:
    """Write a figure as PDF (and optionally PNG) and close it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"))
    if also_png:
        fig.savefig(path.with_suffix(".png"))
    plt.close(fig)


def fitted_legend(ax: plt.Axes, **kwargs):
    """Draw a legend inside `ax` and warn if it overhangs the axes box.

    A legend wider than its panel spills over the neighbouring axis, which is
    easy to miss in a multi-panel figure and impossible to see in the vector
    output until it is on the page.
    """
    kwargs.setdefault("fontsize", FS["legend"])
    legend = ax.legend(**kwargs)
    fig = ax.get_figure()
    fig.canvas.draw()
    box = legend.get_window_extent()
    frame = ax.get_window_extent()
    overhang = max(frame.x0 - box.x0, box.x1 - frame.x1)
    if overhang > 1.0:
        warnings.warn(
            f"legend overhangs its axes by {overhang:.0f} px; "
            "reduce ncol, handlelength or columnspacing",
            stacklevel=2,
        )
    return legend


def panel_label(ax: plt.Axes, text: str, dx: float = -0.16, dy: float = 1.06) -> None:
    """Place a bold panel label in axes coordinates."""
    ax.text(
        dx,
        dy,
        text,
        transform=ax.transAxes,
        fontsize=FS["title"],
        fontweight="bold",
        va="top",
        ha="left",
    )


def panel_title(ax: plt.Axes, letter: str, text: str = "",
                size: float | None = None, pad: float = 5.0) -> None:
    """Left-aligned title carrying its own bold panel letter.

    Keeping the letter inside the title avoids the collisions that occur when a
    separate label is placed in axes coordinates next to a centred title.
    """
    label = rf"$\bf{{{letter}}}$" + (f"   {text}" if text else "")
    ax.set_title(label, loc="left", fontsize=size or FS["title"], pad=pad)
