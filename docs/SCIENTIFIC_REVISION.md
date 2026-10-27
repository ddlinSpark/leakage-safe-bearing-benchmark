# Scientific Revision Record

## Problems in the original manuscript

1. The 10-class test labels were generated from engineering thresholds, and the
   calibrated self-training model also generated replacement labels. Reporting
   96% against these labels measures circular agreement rather than verified
   fault-classification accuracy.
2. The manuscript states that all three data splits contain 22 stations. This
   means that the same station-level operating conditions occur in training and
   testing even when windows are separated by time.
3. The manuscript calls 12,600 windows non-overlapping, while the underlying
   pipeline uses a stride of 250 for a window length of 1000. Adjacent windows
   therefore overlap by 75%.
4. Thirty-four versions were compared, but the paper does not distinguish
   exploratory model selection from confirmatory evaluation or report
   seed-level uncertainty.
5. Unsupported introduction claims about the share of spindle failures and the
   reduction in downtime are not traceable to a cited source.
6. Calibration is described as a cure for label error. Standard calibration
   changes probability estimates; it does not create missing class information.
   The large gain is evidence that the original labels and evaluation protocol
   were inconsistent, not evidence of a generally superior calibrator.
7. The source document has equation clipping, a shifted algorithm block,
   malformed table cells, split result tables, and large blank pages.

## Revised research question

The revision asks whether time-frequency-temporal fusion and calibrated
self-training improve fault diagnosis under a protocol that uses verified labels,
disjoint operating conditions, and unseen bearing units.

The study reports supervised raw-waveform and CWT baselines under an identical
split. It also reports label-efficiency results at 10%, 25%, 50%, and 100% of
the training labels. Self-training may only use the unlabeled training portion;
test labels are never involved.

## Reporting changes

- Use mean, standard deviation, 95% confidence intervals, and paired tests over
  repeated seeds and held-out conditions.
- Report macro-F1 as the primary metric because class prevalence differs.
- Report expected calibration error before and after temperature scaling.
- Distinguish exploratory observations from confirmatory results.
- Add limitations for dataset scope, sensor differences, and the gap between
  bearing benchmarks and complete machine-tool spindle systems.
