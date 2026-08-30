# Submission package: Applied Mathematics and Computation

Elsevier CAS bundle, **single-column** layout (`cas-sc.cls`). The class, style,
bibliography style and icon set are vendored here, so the generated source
archive compiles without the template archive or a system copy of Elsevier's
BibTeX files.

## Build

```bash
cd paper-amc
pdflatex main && bibtex main && pdflatex main && pdflatex main
pdflatex supplement && bibtex supplement && pdflatex supplement && pdflatex supplement
```

Outputs: `main.pdf`, **21 manuscript pages**, and `supplement.pdf`, **6 pages**.
The main paper keeps the central theorem, numerical scheme and headline results;
the supplement carries an auxiliary proof, complete validation and robustness,
exact tables that duplicate graphical displays, and the standard fixation
derivation. The bibliography uses the CAS class defaults rather than tightened
leading. Highlights remain only in the required separate editable
`highlights.docx` rather than being duplicated in `main.tex`.

The build has no undefined citations or references. BibTeX reports `empty
pages` for four arXiv/technical-report entries with no page range. The CAS class
also emits a title-block `Overfull \hbox (117.08pt)` and empty-anchor warnings at
`\maketitle`; visual inspection confirms that the title, five-author block,
affiliation, e-mails and ORCIDs remain inside the page. Loading `float` before
the class prevents duplicate float destinations, while resetting `\sfdefault`
to `lmss` avoids Type 3 strategy-label fonts.

## What to upload to Editorial Manager

Run `python paper-amc/scripts/build_submission_archives.py` from the repository
root for a QA build. The reproducibility ZIP records the base Git commit and a
`git_worktree_dirty` flag in `ENVIRONMENT.json`, while its SHA-256 manifest
identifies the exact included bytes. After committing the exact tested state,
run `python paper-amc/scripts/build_submission_archives.py --require-clean` for
the final release; that command refuses to build from an uncommitted or unknown
Git state. The generated `submission-artifacts/` directory is intentionally
ignored by Git: its ZIPs and outer checksum files are release outputs, and
committing a checksum would change the commit embedded in the reproducibility
ZIP and hence change the checksum again. Upload
`submission-artifacts/delegation-cascade-amc-source.zip` under
the item type "LaTeX source files". Do not zip the whole working folder: it
contains build intermediates and internal submission notes. Every graphic is
referenced through a subdirectory (`figures/`) and `cas-common.sty` hard-codes
`thumbnails/cas-email.jpeg` for `\ead`, so uploading the source files
individually can flatten those paths and break `\maketitle`. The deterministic
source archive contains:

| Item | File |
|---|---|
| Manuscript source | `main.tex` |
| Supplement source | `supplement.tex` |
| Bibliography | `refs.bib`, `main.bbl` and `supplement.bbl` |
| Table sources | `tab_*.tex` (three used by `main.tex`, four by `supplement.tex`) |
| Class and style | `cas-sc.cls`, `cas-common.sty`, `elsarticle-num-names.bst`, `thumbnails/` |
| Figures 2 to 8 | `figures/fig02_transmission.pdf` … `figures/fig08_robustness.pdf` |

and these go up as separate items:

| Item | File |
|---|---|
| Highlights | `highlights.docx` (the word "highlights" is in the file name, as required) |
| Declaration of competing interest | `declaration_of_interest.docx` |
| Cover letter | `cover_letter.md`, after a final author/date check |
| Main compiled PDF (reference only) | `main.pdf` |
| Supplementary material | `supplement.pdf` |

Both `refs.bib` and the two resolved `.bbl` files are included. The latter
protect the submission against publisher-side BibTeX differences, while the
vendored `elsarticle-num-names.bst` keeps a clean rebuild possible.

Figure 1 is a TikZ schematic embedded in `main.tex`; the journal allows text
graphics to be embedded in the LaTeX source. The main paper uses Figures 2 to
7, while `fig08_robustness.pdf` is Figure S1 in the supplement. All figures are
PDF; Figures 2 and 5 embed a raster layer, because matplotlib emits `imshow`
panels as raster XObjects, and `savefig.dpi` is set so that they print above
Elsevier's 500 dpi minimum for combination art.

## Before submitting

- [x] **Author block** in `main.tex`, `CITATION.cff` and
      `scripts/make_submission_files.py`: five authors, one shared full-address
      affiliation, five validated ORCIDs, Trung-Kiet Huynh corresponding.
      Dao-Sy Duy-Minh's confirmed ORCID is `0009-0002-4501-2788`; exact portal
      name segments are in `SUBMISSION_NOTES.md`. The first three
      contributed equally, carried by the `\fnmark[1]`/`\fntext[1]` note. The
      journal does not allow authorship changes after acceptance, and does not
      allow additions, deletions or reordering after submission without an
      Authorship Change Request form.
- [x] All five authors confirmed the displayed names and order, CRediT roles,
      equal-contribution note, corresponding-author e-mail, declarations and
      companion disclosure on 24 August 2026.
- [x] No generative-AI declaration is included because AI use was limited to
      basic checks of grammar, spelling and punctuation, for which Elsevier does
      not require a disclosure statement.
- [x] Four optional reviewer suggestions, institutional addresses and a
      preliminary conflict screen are recorded in `SUBMISSION_NOTES.md`, not in
      the manuscript or cover-letter body. Authors must perform the final
      relationship check before using them.
- [x] The exact tested snapshot is identified by the `v1.0.0` tag and GitHub
      release. Its deterministic reproducibility ZIP records the tagged commit
      and a clean worktree. The release provides an immutable versioned record;
      no DOI is claimed.
- [x] The cover letter discloses the two related companion manuscripts as
      separate submissions in the same coordinated submission round and
      explains the common benchmark layer. It also
      discloses the public non-peer-reviewed development preprint and identifies
      it as not being a version of record.

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
| `figures/*.pdf` | copied directly from `results/figures/` by `make_amc_tables.py` |

`tab_cost.tex` and `tab_verification.tex` were once split from
`numerics_generated.tex` by hand, and went stale against their own generator as
a result; `make_amc_tables.py` now splits that file too, so the only correct
way to change them is to change `run_numerics.py` and re-run both scripts.

For tables, `make_amc_tables.py` applies exactly two mechanical changes and
touches no number: it splits the multi-table generated files into one table per
file, and it rewrites `\begin{table}[t]` as `\begin{table}[pos=t]`, because the CAS
classes give `table` and `figure` a key-value optional argument rather than a
float specifier. Passing a bare `[t]` there silently produces an *empty* float
specifier, which is worth knowing if further floats are added. It also stages
the seven current figures directly from `results/figures/`.

Regenerate everything with:

```bash
python scripts/run_analysis.py
python scripts/run_robustness.py
python scripts/make_figures.py
python scripts/build_paper.py --no-compile
python paper-amc/scripts/run_numerics.py            # ~4 min, needs egttools + mpmath
python paper-amc/scripts/make_amc_tables.py
python paper-amc/scripts/make_submission_files.py  # needs python-docx

cd paper-amc
pdflatex main && bibtex main && pdflatex main && pdflatex main
pdflatex supplement && bibtex supplement && pdflatex supplement && pdflatex supplement
cd ..
python paper-amc/scripts/build_submission_archives.py --require-clean
```

Run the analysis and robustness stages before the figure and table stages; the
later scripts intentionally consume those generated results. Re-run the
archive builder only after both `main.bbl` and `supplement.bbl` are current.

## How this differs from `paper/`

`paper/main.tex` is the venue-neutral 35-page manuscript. This version is
rewritten for the journal's computational emphasis and its length guideline:

- a new Section 5, **Numerical method**, states the cost of the route the paper
  avoids (Table 2), gives the scheme as Algorithm 1 with its complexity, bounds
  the horizon truncation a priori, and summarises five validation checks whose
  complete results are Supplementary Table S1. None of this exists in the
  parent manuscript;
- the analytical results are consolidated into one section, with the check
  placement and attribution floor propositions moved next to the shelter
  theorem they qualify;
- Related work is cut from seven paragraphs to three, and the citation list
  from the parent's 95 references to the 63 that carry an argument here,
  including the reproducibility-software record.
  `refs.bib` holds the union of both variants; a numeric style prints only what
  is cited;
- the main paper keeps three non-duplicative tables. Four exact validation and
  result tables, plus the full robustness figure, are in the supplement;
- **every numbered result is proved in the main paper or supplement.** Lemma 1,
  whose elementary proof is Supplementary Section S1, Propositions 2 and 3,
  Theorem 4, Proposition 5, Corollary 6 and Propositions 7 and 8 keep the
  numbering of `paper/main.tex`, so the two manuscripts can be read side by
  side, and Proposition 9 is added at the end of Section 5.4. The parent
  manuscript leaves Proposition 2, Theorem 4, Proposition 5 and Corollary 6
  unproved and states Theorem 4 without hypotheses; this version supplies both,
  defines `m(d)` explicitly as the self-play harm, and records that the
  fixed-resident reading opens the shelter one hand-off earlier.
