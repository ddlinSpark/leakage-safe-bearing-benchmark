# Final Manuscript Audit

## Verdict

Numerical and manuscript consistency: **PASS**.

The independent audit reports 0 mismatches across
13 tables and 12 figures. The
rendered document contains 21 pages.

## Source files

- Final manuscript: `docs/paper/Leakage_Safe_Bearing_Benchmark_SCI_final.docx`
- Final SHA-256: `9f6cce3d7d430edfcef9e17b5c53330ce6a6f556ba36e96a32582beded19a864`
- Pre-revision backup:
  `docs/paper/archive/Leakage_Safe_Bearing_Benchmark_SCI_pre_statistics_revision.docx`
- Backup SHA-256: `4eb5db8ec8a2119cda50bff40caa2049cc0e171debeac04afff82f1c9cfd1d29`
- Independent audit JSON:
  `artifacts/local/pre_submission_audit_statistics_revision.json`
- Page renders:
  `artifacts/local/render/statistics_revision_final/`

## Terminology scan

- No forbidden or legacy term was found.

The artifact uses "10% balanced budget", "50% balanced budget", and
"Full balanced budget" for the label-budget conditions. The displayed model
name is TFN; the historical machine-readable result key remains `tfnet`.

## Contextual keyword review

These words are not forbidden by themselves. Each match was checked in context.

### `first`
- paragraph 70: ...ter bootstrap: operating-condition folds are resampled first, and seeds are then resampled within each selected fold. Model differences remain pa...
### `outperform`
- paragraph 74: ...4.1 RQ1: Do time-frequency representations outperform raw waveform models under leakage-safe cross-domain evaluation?...

The only `outperform` occurrence is the RQ1 question, which asks whether
time-frequency representations outperform a raw waveform model. It does not
assert that they do.

## p-value formatting

Observed forms:

- `p = 0.006`
- `p = 0.125`
- `p = 0.189`
- `p = 0.250`
- `p = 0.875`
- `p = 1.000`

No malformed form such as `p=<` remains.

## Table and figure consistency

- Table captions present: 1 through 13.
- Figure captions present: 1 through 12.
- Table 6 is generated from `results/hierarchical_statistics.csv`.
- Table 10 displays TFN while retaining the historical `tfnet` result key.
- Reference [22] includes DOI `10.1016/j.ymssp.2023.110952`.
- Figure regeneration: 12 of 12 PNG files were
  byte-identical to the publication copies.
- All 21 rendered pages were visually inspected; no clipping, overlap, missing
  glyph, or broken table was found.
- Accessibility audit: high=0,
  medium=0, low=0.

## Release-readiness

**BLOCKED for anonymous public release until the privacy items are resolved.**

The remaining blockers are privacy-related, not numerical:

- 14 of 16 Git commits carry non-anonymous
  author email metadata.
- 320 of 320 retained canonical run configurations
  contain a workstation hostname field.
- The repository is public while the manuscript is labeled for double-blind
  review.

Resolving these items changes repository history or publication visibility.
They require an explicit decision and should not be altered silently during a
statistical revision.
