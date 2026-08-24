"""Build the separate submission files Applied Mathematics and Computation asks for.

The journal wants three items uploaded alongside the manuscript source:

* highlights, as a separate editable file whose name contains "highlights";
* the declaration of competing interests, as a .doc/.docx file;
* the declaration of generative AI use (also carried inside the manuscript,
  in its own section before the reference list).

    python paper-amc/scripts/make_submission_files.py
"""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.shared import Pt

OUT = Path(__file__).resolve().parents[1]

TITLE = (
    "Delegation cascades in evolutionary games: stable computation with "
    "strategic depth"
)

HIGHLIGHTS = [
    "Strategic depth creates a liability shelter in evolutionary games",
    "Maximum-shifted O(Z) fixation sums stabilise the reduced Markov chain",
    "Joint erosion and attribution loss raise unsafe frequency from 0 to 0.322",
    "An attribution floor matches reachable depth-cap safety-payoff outcomes",
]

AUTHORS = (
    "Trung-Kiet Huynh, Dao Sy Duy Minh, Chi-Nguyen Tran, "
    "Nguyen Lam Phu Quy, Pham Phu Hoa"
)

COMPETING_INTERESTS = (
    "The authors declare that they have no known competing financial interests "
    "or personal relationships that could have appeared to influence the work "
    "reported in this paper."
)

FUNDING = (
    "This research did not receive any specific grant from funding agencies in "
    "the public, commercial, or not-for-profit sectors."
)

AI_DECLARATION = (
    "During the preparation of this work the authors used Anthropic Claude "
    "and OpenAI Codex to draft and copy-edit prose and to assist with refactoring "
    "analysis code. The authors reviewed and edited all outputs, reproduced the "
    "reported results from the released code, and take full responsibility for "
    "the content of the published article. No generative AI tool was used to "
    "create numerical data or figures."
)


def _document() -> Document:
    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(11)
    return doc


def write_highlights() -> None:
    doc = _document()
    doc.add_heading("Highlights", level=1)
    doc.add_paragraph(TITLE).runs[0].bold = True
    doc.add_paragraph(AUTHORS)
    for item in HIGHLIGHTS:
        assert len(item) <= 85, "highlight over 85 characters: %r" % item
        doc.add_paragraph(item, style="List Bullet")
    doc.save(OUT / "highlights.docx")

    (OUT / "highlights.txt").write_text(
        "Highlights\n\n%s\n\n" % TITLE
        + "\n".join("- " + h for h in HIGHLIGHTS)
        + "\n",
        encoding="utf-8",
    )


def write_declarations() -> None:
    doc = _document()
    doc.add_heading("Declaration of competing interest", level=1)
    doc.add_paragraph(TITLE).runs[0].italic = True
    doc.add_paragraph(COMPETING_INTERESTS)
    doc.add_heading("Funding", level=1)
    doc.add_paragraph(FUNDING)
    doc.add_heading(
        "Declaration of generative AI and AI-assisted technologies in the "
        "manuscript preparation process",
        level=1,
    )
    doc.add_paragraph(AI_DECLARATION)
    doc.save(OUT / "declaration_of_interest.docx")


def main() -> None:
    write_highlights()
    write_declarations()
    for name in ("highlights.docx", "highlights.txt", "declaration_of_interest.docx"):
        print(name, (OUT / name).stat().st_size, "bytes")


if __name__ == "__main__":
    main()
