# Manuscript and GitHub Consistency Report

Date: 2026-10-28

## Verdict

**PASS.** Manuscript Table 5, README Main Results, and the canonical result
files agree for all seven method conditions. The independent manuscript audit
reports zero mismatches.

## Main results

Values are mean ± sample standard deviation over four folds and five seeds,
with 20 matched evaluations per condition.

| Method | Macro-F1 | Accuracy | Runs |
| --- | ---: | ---: | ---: |
| Raw 1D-CNN | 0.289 ± 0.104 | 0.392 ± 0.094 | 20 |
| GRU only | 0.221 ± 0.030 | 0.242 ± 0.025 | 20 |
| CWT 2D-CNN | 0.291 ± 0.086 | 0.399 ± 0.081 | 20 |
| CWT-GRU without SE | 0.307 ± 0.103 | 0.408 ± 0.097 | 20 |
| CWT-GRU | 0.280 ± 0.107 | 0.383 ± 0.092 | 20 |
| CWT-GRU + temperature scaling | 0.280 ± 0.107 | 0.383 ± 0.092 | 20 |
| CWT-GRU + temperature scaling + self-training | 0.290 ± 0.116 | 0.387 ± 0.099 | 20 |

## Generalization gap

| Evidence location | Validation Macro-F1 | Held-out test Macro-F1 | Status |
| --- | ---: | ---: | --- |
| Abstract | 0.797–0.853 | 0.280–0.307 | PASS |
| Table 13 | 0.797–0.853 | 0.280–0.307 | PASS |
| Figure 12 | Four principal models | Four principal models | PASS |
| README Generalization Gap | 0.797–0.853 | 0.280–0.307 | PASS |

## Canonical registries

- Canonical evaluations: 320
- Main evaluations: 140
- Label-efficiency evaluations: 80
- Threshold-sensitivity evaluations: 100
- External same-protocol evaluations: 40
- Consolidated registry: 820 rows
- Folds: 0, 1, 2, 3
- Seeds: 42, 123, 2024, 2025, 3407

## Checks

| Check | Result |
| --- | --- |
| Manuscript Table 5 versus independent recomputation | PASS |
| README Main Results versus canonical means and sample SDs | PASS |
| Seven model conditions represented | PASS |
| Run counts equal 20 per condition | PASS |
| Independent DOCX/table audit mismatches | 0 |
| Result consistency checks | 18 of 18 PASS |
| Canonical CSV modification | None |

No canonical result file was changed. The manuscript and README were updated to
the existing canonical values.
