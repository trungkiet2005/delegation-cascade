# Submission package: Applied Mathematics and Computation

Elsevier CAS bundle, **single-column** layout (`cas-sc.cls`). This folder is
self-contained: the class, the style file and the icon set are vendored here, so
the manuscript compiles without the template archive.

## Build

```bash
cd paper-amc
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Output: `main.pdf`, **24 numbered manuscript pages** plus one unnumbered
Highlights page that `cas-sc.cls` typesets automatically from the `highlights`
environment (25 PDF pages in total). This stays within the journal's 25-page
scrutiny threshold without removing the proofs or numerical error bounds.

The build has no undefined citations or references. BibTeX reports `empty
pages` for four arXiv/technical-report entries with no page range. The CAS class
also emits a title-block `Overfull \hbox (117.08pt)` and empty-anchor warnings at
`\maketitle`; visual inspection confirms that the title, five-author block,
affiliation, e-mails and ORCIDs remain inside the page. Loading `float` before
the class prevents duplicate float destinations, while resetting `\sfdefault`
to `lmss` avoids Type 3 strategy-label fonts.

## What to upload to Editorial Manager

Upload the **whole `paper-amc` folder as one archive** under the item type
"LaTeX source files". Every graphic is referenced through a subdirectory
(`figures/`) and `cas-common.sty` hard-codes `thumbnails/cas-email.jpeg` for
`\ead`, so uploading the files individually flattens those paths and the build
dies inside `\maketitle`. The archive must contain:

| Item | File |
|---|---|
| Manuscript source | `main.tex` |
| Bibliography | `refs.bib` and `main.bbl` |
| Table sources | `tab_*.tex` (all seven, `\input` by `main.tex`) |
| Class and style | `cas-sc.cls`, `cas-common.sty`, `thumbnails/` |
| Figures 2 to 8 | `figures/fig02_transmission.pdf` … `figures/fig08_robustness.pdf` |

and these go up as separate items:

| Item | File |
|---|---|
| Highlights | `highlights.docx` (the word "highlights" is in the file name, as required) |
| Declaration of competing interest | `declaration_of_interest.docx` |
| Cover letter | `cover_letter.md`, after filling in the bracketed fields |
| Compiled PDF (reference only) | `main.pdf` |

`elsarticle-num-names.bst` is *not* vendored here, so the folder is
self-contained for the class but not for the bibliography style; `main.bbl` is
in the list for that reason.

Figure 1 is a TikZ schematic embedded in `main.tex`; the journal allows text
graphics to be embedded in the LaTeX source. The file names of Figures 2 to 8
carry their figure numbers, so `fig05_plane.pdf` is Figure 5. All figures are
PDF; Figures 2 and 5 embed a raster layer, because matplotlib emits `imshow`
panels as raster XObjects, and `savefig.dpi` is set so that they print above
Elsevier's 500 dpi minimum for combination art.

## Before submitting

- [x] **Author block** in `main.tex` and in
      `scripts/make_submission_files.py`: five authors, one shared full-address
      affiliation, real ORCIDs, Trung-Kiet Huynh corresponding. The first three
      contributed equally, carried by the `\fnmark[1]`/`\fntext[1]` note. The
      journal does not allow authorship changes after acceptance, and does not
      allow additions, deletions or reordering after submission without an
      Authorship Change Request form.
- [ ] The corresponding-author e-mail is the institutional student address
      `23122039@student.hcmus.edu.vn`. Confirm it will still be readable when
      the article appears, or swap in a durable address.
- [x] The **generative AI declaration** names Anthropic Claude and OpenAI Codex,
      describes prose editing and code-refactoring assistance, and states that
      generated numerical data and figures were not used. All authors must
      confirm this wording is factually complete before submission.
- [ ] Fill in the suggested reviewers in `cover_letter.md`, or delete that
      sentence.
- [x] The **data availability** URL
      <https://github.com/trungkiet2005/delegation-cascade> is public and the
      repository contains `paper-amc/scripts/` for reproducing the numerical
      benchmarks of Section 5 (verified 24 August 2026).

## Provenance of the numbers

Nothing in this folder recomputes the model. Every table is either produced by
the parent study's analysis scripts or by the benchmark script added for this
submission.

| File | Produced by |
|---|---|
| `tab_race.tex`, `tab_decomposition.tex` | `scripts/make_amc_tables.py` from `paper/tables_generated.tex` |
| `tab_regimes.tex`, `tab_floorcap.tex` | `scripts/make_amc_tables.py` from `paper/regimes_generated.tex` |
| `tab_kernelshape.tex` | `scripts/make_amc_tables.py` from `paper/robustness_generated.tex` |
| `tab_cost.tex`, `tab_verification.tex` | `scripts/make_amc_tables.py` from `numerics_generated.tex` |
| `numerics_generated.tex`, `numerics_key_numbers.json` | `scripts/run_numerics.py` |
| `figures/*.pdf` | copied from `paper/figures/`, unchanged |

`tab_cost.tex` and `tab_verification.tex` were once split from
`numerics_generated.tex` by hand, and went stale against their own generator as
a result; `make_amc_tables.py` now splits that file too, so the only correct
way to change them is to change `run_numerics.py` and re-run both scripts.

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
  avoids (Table 2), gives the scheme as Algorithm 1 with its complexity, bounds
  the horizon truncation a priori, and checks five different things in Table 3.
  None of this exists in the parent manuscript;
- the analytical results are consolidated into one section, with the check
  placement and attribution floor propositions moved next to the shelter
  theorem they qualify;
- Related work is cut from seven paragraphs to three, and the citation list
  from the parent's 95 references to the 62 that carry an argument here.
  `refs.bib` holds the union of both variants; a numeric style prints only what
  is cited;
- the depth-ceiling and evolutionary-process tables are folded into the text,
  leaving seven tables;
- **every numbered result is proved.** Lemma 1, Propositions 2 and 3,
  Theorem 4, Proposition 5, Corollary 6 and Propositions 7 and 8 keep the
  numbering of `paper/main.tex`, so the two manuscripts can be read side by
  side, and Proposition 9 is added at the end of Section 5.4. The parent
  manuscript leaves Proposition 2, Theorem 4, Proposition 5 and Corollary 6
  unproved and states Theorem 4 without hypotheses; this version supplies both,
  defines `m(d)` explicitly as the self-play harm, and records that the
  fixed-resident reading opens the shelter one hand-off earlier.
