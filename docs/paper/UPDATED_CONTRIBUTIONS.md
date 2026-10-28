# Updated Contributions

The revised Section 1.1 contains the following four contributions.

1. **A leakage-safe protocol that applies simultaneous physical-bearing and
   operating-condition holdout to non-overlapping windows and audited raw-source
   identities.**

   The cache is reconstructed from official archives and re-standardized
   elementwise from the restored source windows. The audit verifies that test
   bearings and conditions never enter training or validation.

2. **A protocol-controlled comparison of representation, architecture, and
   training-procedure choices on identical fold-seed keys.**

   Raw waveform, recurrent, magnitude-CWT, fusion, SE, temperature scaling,
   fixed-threshold self-training, and coverage-controlled self-training are
   compared under the same split. DRSN and TFN are separately described as
   same-protocol adaptations.

3. **Fold-first statistical inference for repeated seeds within shared
   operating-condition domains.**

   Paired intervals resample folds before seeds. The exact sign-flip test and
   the four-domain inferential limit are reported alongside descriptive
   fold-wise wins and losses.

4. **A cross-domain evidence analysis that separates claims supported by the
   experiment from unisolated causal factors.**

   Raw-signal statistics, damage-origin manifests, label-granularity controls,
   calibration, training audits, and penultimate feature visualizations are
   reported. The design supports a statement about joint bearing, condition,
   and damage-origin change, not a claim that one factor caused the gap.

The wording avoids claiming a new general architecture, a new universal
benchmark, or state-of-the-art superiority.
