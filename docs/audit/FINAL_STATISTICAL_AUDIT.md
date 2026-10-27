# Final Statistical Audit

## Data preservation

- Canonical result rows: 320
- External-baseline rows: 40
- Canonical raw-results SHA-256: `a047b0f9530f07643ef2c5c57f395a15abe045c7e8640a10119c273e46fbeb10`
- External raw-results SHA-256: `c92f2b4a447dbc13f581981d15d76e97ca3382ab024a905f4fbccd6ac1e5b09a`
- Backup: `docs/paper/archive/Leakage_Safe_Bearing_Benchmark_SCI_pre_statistics_revision.docx`; SHA-256 `4eb5db8ec8a2119cda50bff40caa2049cc0e171debeac04afff82f1c9cfd1d29`.
- Revised manuscript: `docs/paper/Leakage_Safe_Bearing_Benchmark_SCI_final.docx`; SHA-256 `9f6cce3d7d430edfcef9e17b5c53330ce6a6f556ba36e96a32582beded19a864`.

No experimental row was rerun, deleted, relabeled by seed performance, or
filtered by outcome.

## Before revision

The manuscript described 20 matched fold-seed evaluations and used a t interval
plus a two-sided Wilcoxon signed-rank test across those 20 cells. The
fold-within-seed dependence was not represented in the inferential analysis.
Those values remain available in `results/statistical_tests.csv` as sensitivity
analyses.

## After revision

The manuscript now distinguishes descriptive variability from domain-level
inference. Primary paired intervals use a deterministic two-stage cluster
bootstrap with operating-condition folds as clusters and seeds nested within
folds. The exact fold sign-flip test is reported with its minimum attainable
two-sided p-value of 0.125. The report is generated from all 20 matched
fold-seed pairs per comparison; no average-of-seed summary replaces the raw
pairs.

## Changed conclusions

- `main:raw_cnn:gru_only`: unclustered significance True, hierarchical interval excludes zero False; the primary conclusion was changed conservatively.
- `main:gru_only:cwt_gru`: unclustered significance True, hierarchical interval excludes zero False; the primary conclusion was changed conservatively.
- `external:cwt_gru:drsn`: unclustered significance True, hierarchical interval excludes zero False; the primary conclusion was changed conservatively.

## Preserved conclusions

- Raw 1D-CNN versus CWT 2D-CNN still has a hierarchical interval that includes
  zero.
- Raw 1D-CNN versus CWT-GRU still has a hierarchical interval that includes
  zero.
- CWT-GRU without SE versus full CWT-GRU still has an interval that includes
  zero; the numerical direction is retained without an SE penalty claim.
- Self-training at 50% balanced budget and Full balanced budget still has no
  stable evidence of improvement.
- DRSN and TFN adaptations have lower mean macro-F1 than Raw 1D-CNN under the
  same protocol; the four-fold p-values remain explicitly coarse.

## Reproduction

```powershell
$env:PYTHONPATH = "src"
python -m spindle_cnn.analysis.hierarchical `
  --results-dir results `
  --external-dir results/sota `
  --output results/hierarchical_statistics.csv `
  --bootstrap-samples 20000
python scripts/build_statistical_reports.py
```

The comparison table and all numerical statements in the revised manuscript
are loaded from `results/hierarchical_statistics.csv` or the retained canonical
result files by the manuscript builder.
