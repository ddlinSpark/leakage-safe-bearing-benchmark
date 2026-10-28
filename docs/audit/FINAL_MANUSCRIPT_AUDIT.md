# Final Manuscript Audit

## Verdict

Numerical and manuscript consistency: **PASS**.

The independent audit reports 0 mismatches across
18 tables and 23 figures. The
rendered document contains 30 pages.

## Source files

- Final manuscript: `docs/paper/Leakage_Safe_Bearing_Benchmark_SCI_final.docx`
- Final SHA-256: `5e0c69fb6ef555f17b74dd0adea670752153178fc2137906ba4f397736be4605`
- Pre-revision backup:
  `docs/paper/archive/Leakage_Safe_Bearing_Benchmark_SCI_pre_statistics_revision.docx`
- Backup SHA-256: `4eb5db8ec8a2119cda50bff40caa2049cc0e171debeac04afff82f1c9cfd1d29`
- Independent audit JSON:
  `artifacts/local/verify_manuscript_numbers.json`
- Figure hash manifest:
  `artifacts/local/figure_hashes.json`
- Page renders:
  `artifacts/local/render/final_revision/`

## Terminology scan

- No forbidden or legacy term was found.

The artifact uses "10% balanced budget", "50% balanced budget", and
"Full balanced budget" for the label-budget conditions. The displayed model
name is TFN; the historical machine-readable result key remains `tfnet`.

## Contextual keyword review

These words are not forbidden by themselves. Each match was checked in context.

### `first`
- paragraph 16: ...Fold-first statistical inference that resamples operating-condition domains before stochastic s...
- paragraph 73: ...ter bootstrap: operating-condition folds are resampled first, and seeds are then resampled within each selected fold. Model differences remain pa...
### `outperform`
- paragraph 78: ...4.1 RQ1: Do time-frequency representations outperform raw waveform models under leakage-safe cross-domain evaluation?...

The only `outperform` occurrence is the RQ1 question, which asks whether
time-frequency representations outperform a raw waveform model. It does not
assert that they do.

## p-value formatting

Observed forms:

- `p = 0.006`
- `p = 0.125`
- `p = 0.189.`
- `p = 0.250`
- `p = 0.875`
- `p = 1.000`

No malformed form such as `p=<` remains.

## Table and figure consistency

- Table captions present: 1 through 18.
- Figure captions present: 1 through 18 and S1 through S5.
- Table 6 is generated from `results/hierarchical_statistics.csv`.
- Table 10 displays TFN while retaining the historical `tfnet` result key.
- Reference [22] includes DOI `10.1016/j.ymssp.2023.110952`.
- Figure hashes: 48 of 48 PNG and PDF files
  match the publication hash manifest.
- All 30 rendered pages were visually inspected; no clipping, overlap, missing
  glyph, or broken table was found.
- Accessibility audit: high=0,
  medium=0, low=0.

## Release-readiness

**BLOCKED for anonymous public release until the privacy items are resolved.**

The remaining blockers are privacy-related, not numerical:

- 14 of 18 Git commits carry non-anonymous
  author email metadata.
- 320 of 320 retained canonical run configurations
  contain a workstation hostname field.
- The repository is public while the manuscript is labeled for double-blind
  review.

Resolving these items changes repository history or publication visibility.
They require an explicit decision and should not be altered silently during a
statistical revision.
