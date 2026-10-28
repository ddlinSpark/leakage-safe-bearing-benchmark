# Consistency Report

## Verdict

**PASS**. 18 of 18 checks passed.

The check reads canonical CSV/JSON results and runs the independent DOCX/table audit. It never rewrites an experimental value.

## Checks

| Check | Status | Evidence |
| --- | --- | --- |
| Canonical run counts | PASS | 320 rows: {'main': 140, 'threshold': 100, 'label_efficiency': 80} |
| External adaptation counts | PASS | 40 rows |
| Consolidated experiment registry | PASS | 820 rows |
| Canonical result catalog | PASS | 13 files |
| Frozen split catalog | PASS | 4 files |
| Unique canonical run identifiers | PASS | canonical=320, external=40 |
| Completed canonical runs | PASS | all retained canonical and external rows report success |
| Clean canonical provenance | PASS | no canonical run was produced from a dirty worktree |
| Main model matrix | PASS | 7 methods x 4 folds x 5 seeds = 140 matched runs |
| External model matrix | PASS | 2 adaptations x 4 folds x 5 seeds = 40 matched runs |
| Macro-F1 and accuracy aggregation | PASS | all seven main methods are present |
| README key findings | PASS | seven result rows match canonical means and sample SDs |
| paper_metrics model means | PASS | main model means match raw_runs.csv exactly |
| paper_metrics external means | PASS | DRSN and TFN adaptation means match raw_runs.csv exactly |
| Fold-first hierarchical inference | PASS | 19 comparisons retain 4 folds, 5 seeds, 20 pairs, 20000 bootstrap samples, ordered CIs, and p >= 0.125 |
| Parameter-count registry | PASS | cwt_cnn=884564, cwt_gru=1101012, cwt_gru_no_se=1086244, cwt_gru_selftrain=1101012, cwt_gru_temp=1101012, gru_only=118276, raw_cnn=244740 |
| Seed and fold registry | PASS | seeds=[42, 123, 2024, 2025, 3407], folds=[0, 1, 2, 3] |
| Manuscript and table consistency | PASS | independent DOCX/table audit: 0 mismatches |

## Mismatches

- None.

## Source precedence

`results/raw_runs.csv` and the retained per-run predictions remain the numerical sources of truth. The DOCX is checked against those files and is never used to overwrite them.
