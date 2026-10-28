# Paper Revision Report

## Revision objective

The revision strengthens the evidence chain without changing the primary split,
seeds, window construction, four-class definition, or canonical result values.
The paper is framed as a leakage-safe cross-domain benchmark and negative-results
study, not as a new high-scoring architecture.

## Item-by-item changes

| Problem | Evidence | Change | Manuscript location |
| --- | --- | --- | --- |
| The leakage-safe claim needed direct source-level verification. | All 16 selected archives match recorded SHA-256 values; 127 readable source files reproduce the 3,939 canonical windows after standardization with zero maximum difference. | Added a source-window audit and state that one known truncated KA08 file is excluded by both constructions. | Section 3.2; `AUDIT_REPORT.md`. |
| The validation-test gap was described mainly as joint domain shift. | Raw vibration RMS, kurtosis, crest factor, spectral centroid, and spectral entropy differ across held-out conditions, with fold-dependent direction. | Added raw-signal diagnostics and an explicit non-causal interpretation. | Section 7.6; Table 14; Figure 13. |
| Damage origin was mentioned only as a limitation. | All held-out fault codes have real lifetime damage; fitting single-location faults mix origins and fitting multiple-damage faults are real. | Added a fold-condition origin manifest and retained the sentence that bearing identity, condition, and damage origin change together. | Section 7.6; Discussion 8.1; Conclusion 9. |
| Four-class failure could be mistaken for complete loss of fault detection. | Post-hoc healthy-versus-fault Macro-F1 is higher than four-class Macro-F1 for the principal models, while three-class performance remains lower than binary. | Added a post-hoc label-granularity sensitivity without changing the primary task. | Section 7.6; Table 15; Figure 14. |
| The magnitude-CWT negative result could be overgeneralized or attributed to one arbitrary setting. | Scale count, resolution, maximum normalization, log magnitude, and real/imaginary representations were evaluated under matched fold-seed keys. The 32-scale setting improved over the 48-scale baseline, but its interval versus Raw 1D-CNN still included zero and only one of four fold means favored it over Raw. | Added full CWT preprocessing controls, retained the primary configuration, and report the scale-specific result without post-hoc promotion. | Sections 3.8, 7.7, and 8.2; Table 16; Figure 15. |
| Window-wise normalization could remove transferable amplitude information. | Training-partition channel statistics were applied to validation and test partitions and CWT maps were rebuilt. Raw 1D-CNN fell by 0.170 Macro-F1 with an interval of -0.231 to -0.094; CWT-GRU changed by +0.020 with an interval that included zero. | Added normalization A/B sensitivity while preserving the primary per-window z-score protocol. | Sections 3.8 and 7.7; `results/normalization_sensitivity_summary.csv`; Appendix Figure S5. |
| Fixed-threshold self-training could be criticized as an intentionally weak baseline. | Coverage-controlled selection achieved mean pseudo-label accuracy 0.987 and coverage 50.1%, but its 50% budget difference from supervised training was +0.034 with an interval of -0.020 to +0.096. At the Full budget it selected no multiple-damage pseudo-label in 4 of 20 runs. | Added per-class top-50% pseudo-labeling and class-conditional selection diagnostics. | Section 7.7; Table 17; Figure 16; Discussion 8.4. |
| DRSN and TFN low scores could be attributed to an unsuitable default learning rate. | A second predeclared learning-rate candidate was selected per fold using validation Macro-F1 only. | Added a finite baseline tuning audit covering Raw, CWT, DRSN, and TFN; all test outcomes were retained. | Section 7.7; Table 18. |
| A 12-epoch budget could underfit slower architectures. | The existing histories and a 30-epoch opportunity were analyzed with validation-only selection. No 30-epoch run selected the final epoch; the latest selected epoch was 19. Validation improved in 53 of 80 matched checks, but every test interval for the 30-versus-12 difference included zero. | Added training-budget sensitivity without replacing the predeclared 12-epoch primary matrix. | Sections 3.8 and 7.7; Table 18; Appendix Figure S4. |
| Aggregate reliability diagrams could hide fold-specific calibration failure. | Per-run temperatures and per-fold reliability bins were recomputed from validation-fitted temperatures and saved logits. | Added per-fold reliability, temperature distributions, and validity checks for saved test probabilities. | Sections 7.3 and 8.5; Figure 8; Appendix Figures S1-S2. |
| Feature clusters were not inspected. | CWT-GRU embeddings had negative domain silhouette across folds, low held-out class silhouette (0.149-0.411), and higher same-class than same-bearing or same-condition nearest-neighbour agreement. | Added PCA/t-SNE and neighbour diagnostics, with test labels used only for colouring. The result shows partial class organization but weak held-out class cohesion, not a clean class-versus-domain separation. | Section 7.7; Figures 17-18. |
| The paper could overstate comparison fairness or universal failure. | Canonical outcomes are unchanged; sensitivity settings do not replace the primary configuration; external models remain adaptations. | Rewrote the title, abstract, contributions, discussion, conclusion, and limitations around evidence boundaries. | Title; Abstract; Sections 1.1, 8, and 9. |

## Claim controls

- The tested bearing codes remain K005, KA15, KI14, and KB27.
- Test bearing and condition data remain absent from fitting, validation,
  temperature fitting, pseudo-label selection, and per-run model selection.
- The 320 canonical rows and 40 external rows are preserved.
- No seed or fold is removed because of its test outcome.
- Four independent operating-condition folds remain the inferential unit.
- Results are stated for the evaluated configurations, not for entire method
  families.

## Revision status

- Leakage audit: **VERIFIED**
- Split-manifest reproduction: **VERIFIED**
- Canonical numerical audit: **VERIFIED**
- Raw source reconstruction: **VERIFIED**
- Sensitivity analyses: retained as exploratory when the Git worktree was dirty
- Claims: limited to the protocol and evaluated configurations
