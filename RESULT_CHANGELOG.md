# Result Changelog

## Canonical results

No canonical result row, seed, fold, split manifest, metric, or manuscript table value was changed in this revision.

- Canonical rows preserved: 320
- External adaptation rows preserved: 40
- Canonical raw result SHA-256 after revision: `a047b0f9530f07643ef2c5c57f395a15abe045c7e8640a10119c273e46fbeb10`
- External raw result SHA-256 after revision: `c92f2b4a447dbc13f581981d15d76e97ca3382ab024a905f4fbccd6ac1e5b09a`

## Why the primary values did not change

The audit established that the stored split claims are correct, the window ranges do not overlap, held-out bearings never enter fitting, and all saved logits reproduce the recorded metrics. No implementation defect justified a replacement of the primary matrix.

## Added evidence

- Raw source windows and exact measurement filenames were restored and shown to reproduce the canonical cache elementwise.
- Damage-origin and label manifests were added.
- Domain-shift, label-granularity, calibration, baseline-tuning, training-budget, CWT, normalization, and self-training sensitivity outputs were added as separate result families.
- The no-SE paired estimate and hierarchical interval were independently reproduced.
- A resumed 30-epoch run briefly created four duplicate fold-seed rows because the output-path configuration changed between attempts. All attempts are retained in `raw_runs_all_attempts.csv`; the 80 unique rows used for analysis had identical duplicate metrics and were selected by latest configuration timestamp, not by outcome.

## Manuscript changes

- The title now foregrounds cross-domain benchmarking and lists time-frequency fusion, calibration, and self-training as evaluated components rather than as a proposed new architecture.
- The abstract and contributions emphasize protocol, fold-aware inference, configuration-specific claims, and joint-shift evidence.
- Tables 14-18 and Figures 13-18 were added for the sensitivity and diagnostic analyses. Supplementary Figures S1-S5 retain additional fold and audit views.

## Selection boundary

No sensitivity setting was moved into the primary configuration after observing held-out test performance. No run was deleted or filtered by outcome.
