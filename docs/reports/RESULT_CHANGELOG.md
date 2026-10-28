# Result Changelog

## Canonical results

No canonical result row, seed, fold, split manifest, metric, or manuscript table value was changed in this revision.

- Canonical rows preserved: 320
- External adaptation rows preserved: 40
- Canonical raw result SHA-256 after revision: `fc8336c7d44cba399c8b0a2356cc378d80a36c4e368769ff922a3e41f55ab5bf`
- External raw result SHA-256 after revision: `193bf7ac9a00d8980eb3c5ed807db6ac12a33dc1657f2a6d1eb0c95c58b7881e`

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
