# Updated Contributions

The revised Section 1.1 contains the following three contributions.

1. **A joint comparison that distinguishes representation and training-
   procedure effects from split-induced inflation by evaluating raw waveform,
   time-frequency, recurrent fusion, attention, temperature scaling, and
   self-training under simultaneous bearing-identity and operating-condition
   holdout.**

   Prior leakage-aware work has primarily exposed inflated scores or audited
   split construction. This study does not claim to discover leakage. It
   evaluates several representation and training choices under one common
   grouped protocol while withholding target labels from fitting, calibration,
   pseudo-label creation, and model selection.

2. **A protocol-controlled external comparison in which representative DRSN
   and TFN adaptations are retrained on the same windows, folds, seeds, and
   metrics rather than ranked with numbers imported from incompatible published
   splits.**

   The contribution is the controlled comparison boundary. The external models
   are adaptations and are not presented as exact reproductions of their
   original experimental settings.

3. **An evidence report that retains negative and statistically non-significant
   outcomes, reports fold-aware uncertainty and the validation-to-test gap, and
   separates calibration reliability from classification accuracy.**

   This contribution emphasizes reporting practice and evidence quality rather
   than claiming that architectural complexity or self-training is inherently
   beneficial.

The wording avoids claiming a new general architecture, a new universal
benchmark, or state-of-the-art superiority.
