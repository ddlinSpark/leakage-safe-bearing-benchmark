# Final Statistical Audit

## Data preservation

- Canonical result rows: 320
- External-baseline rows: 40
- Canonical raw-results LF-normalized SHA-256: `fc8336c7d44cba399c8b0a2356cc378d80a36c4e368769ff922a3e41f55ab5bf`
- External raw-results LF-normalized SHA-256: `193bf7ac9a00d8980eb3c5ed807db6ac12a33dc1657f2a6d1eb0c95c58b7881e`
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

## No-SE interval direction check

The published hierarchical row is defined as full CWT-GRU minus CWT-GRU without
SE. Its point estimate is `-0.027` and its interval is `[-0.073, +0.022]`.
Expressed as no-SE minus full, the point estimate is `+0.027` and the interval
is `[-0.022, +0.073]`. The sign reversal was checked directly from all 20 paired
fold-seed rows with the recorded bootstrap seed. The estimate is not a
transcription error and lies inside the sign-reversed percentile interval.

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
