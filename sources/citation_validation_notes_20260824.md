# Citation-validation notes (24 August 2026)

The `citation-management` validator parsed all 97 entries in
`paper-amc/refs.bib`. The manuscript cites 62 unique keys; all 62 are defined,
and BibTeX emits only those cited entries in the numerical reference list.
There are no undefined citations or duplicate citation keys.

The automated report in `citation_validation_20260824.json` labels three DOI
lookups as failures. These are resolver false negatives, not invalid metadata:

- `10.1145/3630106.3658948` is printed in the official FAccT 2024 paper,
  *Visibility into AI Agents*:
  https://facctconference.org/static/papers24/facct24-63.pdf
- `10.1093/imanum/draa038` is confirmed by the Oxford Academic article page for
  *Accurately computing the log-sum-exp and softmax functions*:
  https://academic.oup.com/imajna/article/41/4/2311/5893596
- `10.1145/3593013.3594073` is printed in the University of Cambridge
  repository copy of *Understanding accountability in algorithmic supply
  chains*:
  https://api.repository.cam.ac.uk/server/api/core/bitstreams/c67e48b1-8e01-402b-a562-fd31945414f9/content

The report's suggested duplicate between `lior2020respondeat` and
`shumailov2024collapse` is also a parser false positive: their titles, authors,
venues and years are different. The earlier full metadata and claim-fit audit
is retained at `paper/citation_audit_2026-08-19.md`.

BibTeX reports four empty-page warnings for conference records whose official
metadata do not use conventional page ranges (`wu2023autogen`,
`alemohammad2023mad`, `perez2025telephone`, and `cemri2025mast`). These are
warnings rather than missing cited works; each entry includes its conference
and an arXiv identifier or DOI where available.
