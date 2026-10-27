# Public Benchmark Protocol

## Paderborn primary experiment

- Dataset: Paderborn University Bearing Data Center.
- Signals: `vibration_1`, `phase_current_1`, `phase_current_2`.
- Sampling rate: 64 kHz.
- Classes: healthy, outer-ring, inner-ring, and multiple damage.
- Training bearings: `K001-K004`, `KA01/KA04/KA08`,
  `KI01/KI04/KI08`, and `KB23/KB24`.
- Unseen test bearings: `K005`, `KA15`, `KI14`, `KB27`.
- Conditions: `N09_M07_F10`, `N15_M01_F10`, `N15_M07_F04`,
  `N15_M07_F10`.
- Windows: 8192 samples, stride 8192.
- Measurements: files 1 and 2 for every selected bearing and condition.

For each of four folds, one operating condition is the test domain, a second
condition is used for validation, and the other two are used for training. The
test bearing codes are excluded from every training and validation partition.
The result is a bearing-code-disjoint and operating-condition-disjoint test.

## Repeated design

The canonical config uses `4 folds x 5 seeds`, or 20 runs per method under the
Full balanced budget.
The seeds are `42, 123, 2024, 2025, 3407`.

The unified runner produces:

```text
Raw 1D-CNN
GRU-only
CWT 2D-CNN
CWT-GRU without SE
CWT-GRU with SE
CWT-GRU with SE + temperature scaling
CWT-GRU with SE + temperature scaling + self-training
```

Temperature scaling reuses the same trained CWT-GRU weights; it is an evaluation
condition rather than a new architecture.

## Label efficiency

The `10% balanced budget`, `50% balanced budget`, and `Full balanced budget`
settings refer to a
class-balanced label budget. At each fraction, the runner samples the same
number of windows from every class. The full budget contains the minimum
available count across classes and may be smaller than the total training
partition. Exact counts are stored per run.

## Threshold sensitivity

At a 50% balanced budget, self-training is evaluated at confidence thresholds
`0.30–0.70` with the same 4 folds and 5 seeds. Candidate coverage, pseudo-label
accuracy, test Macro-F1, test accuracy, and calibration error are recorded.

Pseudo-label accuracy is computed after selection for audit only. It is never
used to train, tune, or select a threshold for the confirmatory run.

## External same-protocol adaptations

DRSN and TFN are evaluated with the same windows, four folds, five seeds,
labels, optimizer, epoch limit, early-stopping rule, validation split, test
split, and metrics. Their results are stored separately under `results/sota/`.

- DRSN: residual shrinkage with adaptive channel-wise soft thresholding, adapted
  to the three-channel four-class task.
- TFN: the public TFN comparison backbone, adapted to three input channels
  and four classes.

These are adaptations, not exact reproductions of the original datasets,
dimensionalities, or training recipes. MTFC-T and DRSN-Transformer are excluded
from numerical comparison because sufficiently complete implementations were
not available in the project materials.

## CWRU pilot

CWRU is retained as a saturation check, not as the primary evidence. Its
leave-one-load-out experiment uses 2048-sample non-overlapping windows. CWRU
does not provide the same physical-bearing grouping as the Paderborn benchmark,
so CWRU results must not be described as bearing-code-disjoint.
