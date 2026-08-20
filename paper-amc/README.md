# Submission package: Applied Mathematics and Computation

Elsevier CAS bundle, **single-column** layout (`cas-sc.cls`). This folder is
self-contained: the class, the style file and the icon set are vendored here, so
the manuscript compiles without the template archive.

## Build

```bash
cd paper-amc
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Output: `main.pdf`, **25 numbered manuscript pages** plus one unnumbered
Highlights page that `cas-sc.cls` typesets automatically from the `highlights`
environment. The journal asks that submissions longer than 25 manuscript pages
be justified by their content, so the manuscript is at, not over, that mark.

Two build warnings are expected and harmless. The `Overfull \hbox (117.08pt)`
at `\maketitle` is produced identically by Elsevier's own `cas-sc-sample.tex`
and is a class artefact. The four BibTeX `empty pages` warnings are arXiv
preprints and technical reports with no page range; `refs.bib` is kept
byte-identical to the audited bibliography of the parent study rather than
edited to silence them.

## What to upload to Editorial Manager

| Item | File |
|---|---|
| Manuscript source | `main.tex` |
| Bibliography | `refs.bib` |
| Table sources | `tab_*.tex` (all seven, `\input` by `main.tex`) |
| Class and style | `cas-sc.cls`, `cas-common.sty`, `thumbnails/` |
| Figures 2 to 8 | `figures/fig02_transmission.pdf` … `figures/fig08_robustness.pdf` |
| Highlights | `highlights.docx` (the word "highlights" is in the file name, as required) |
| Declaration of competing interests | `declaration_of_interest.docx` |
| Cover letter | `cover_letter.md`, after filling in the bracketed fields |
| Compiled PDF (reference only) | `main.pdf` |

Figure 1 is a TikZ schematic embedded in `main.tex`; the journal allows text
graphics to be embedded in the LaTeX source. The file names of Figures 2 to 8
carry their figure numbers, so `fig05_plane.pdf` is Figure 5. All figures are
vector PDF, which is the format the journal asks for line drawings.

## Before submitting

- [x] **Author block** in `main.tex` and in
      `scripts/make_submission_files.py`: three authors, one shared
      affiliation, real ORCIDs, Trung-Kiet Huynh corresponding. All three
      contributed equally, carried by the `\fnmark[1]`/`\fntext[1]` note. The
      journal does not allow authorship changes after acceptance, and does not
      allow additions, deletions or reordering after submission without an
      Authorship Change Request form.
- [ ] The corresponding-author e-mail is the institutional student address
      `23122039@student.hcmus.edu.vn`. Confirm it will still be readable when
      the article appears, or swap in a durable address.
- [ ] Check the **generative AI declaration** near the end of `main.tex`
      against what was actually used, and edit it. It is a required statement
      when AI tools were used in manuscript preparation and it is published
      with the article.
- [ ] Fill in the suggested reviewers in `cover_letter.md`, or delete that
      sentence.
- [ ] Confirm the **data availability** statement: it points at
      <https://github.com/trungkiet2005/delegation-cascade>, which must be
      public and must contain `paper-amc/scripts/` for the numerical benchmarks
      of Section 5 to be reproducible.

## Provenance of the numbers

Nothing in this folder recomputes the model. Every table is either produced by
the parent study's analysis scripts or by the benchmark script added for this
submission.

| File | Produced by |
|---|---|
| `tab_race.tex`, `tab_decomposition.tex` | `scripts/make_amc_tables.py` from `paper/tables_generated.tex` |
| `tab_regimes.tex`, `tab_floorcap.tex` | `scripts/make_amc_tables.py` from `paper/regimes_generated.tex` |
| `tab_kernelshape.tex` | `scripts/make_amc_tables.py` from `paper/robustness_generated.tex` |
| `tab_cost.tex`, `tab_verification.tex` | `scripts/run_numerics.py` |
| `numerics_key_numbers.json` | `scripts/run_numerics.py` |
| `figures/*.pdf` | copied from `paper/figures/`, unchanged |

`make_amc_tables.py` applies exactly two mechanical changes and touches no
number: it splits the multi-table generated files into one table per file, and
it rewrites `\begin{table}[t]` as `\begin{table}[pos=t]`, because the CAS
classes give `table` and `figure` a key-value optional argument rather than a
float specifier. Passing a bare `[t]` there silently produces an *empty* float
specifier, which is worth knowing if further floats are added.

Regenerate everything with:

```bash
python paper-amc/scripts/make_amc_tables.py        # from repository root
python paper-amc/scripts/run_numerics.py           # ~4 min, needs egttools + mpmath
python paper-amc/scripts/make_submission_files.py  # needs python-docx
```

## How this differs from `paper/`

`paper/main.tex` is the venue-neutral 35-page manuscript. This version is
rewritten for the journal's computational emphasis and its length guideline:

- a new Section 5, **Numerical method**, states the cost of the route the paper
  avoids (Table 2), gives the scheme as Algorithm 1 with its complexity, and
  verifies it in four independent checks (Table 3). None of this exists in the
  parent manuscript;
- the analytical results are consolidated into one section, with the audit
  placement and attribution floor propositions moved next to the shelter
  theorem they qualify;
- Related work is cut from seven paragraphs to three, and the citation list
  from the parent's 95 references to the 61 that carry an argument here.
  `refs.bib` holds the 96-entry union of both variants; a numeric style prints
  only what is cited;
- the depth-ceiling and evolutionary-process tables are folded into the text,
  leaving seven tables;
- the numbered results keep the same order as in `paper/main.tex`
  (Lemma 1, Propositions 2 and 3, Theorem 4, Proposition 5, Corollary 6,
  Propositions 7 and 8), so the two manuscripts can be read side by side.
