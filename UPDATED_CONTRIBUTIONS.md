# Updated Contributions

The revised Section 1.1 contains the following three contributions.

1. **A leakage-safe comparison that extends split auditing by jointly
   evaluating waveform and time-frequency representations, recurrent fusion,
   SE attention, temperature scaling, and self-training under simultaneous
   unseen-bearing and unseen-condition holdout.**

   Earlier leakage-aware work primarily exposes inflated scores or audits
   split construction. This contribution concerns the behavior of several
   representation and training choices under one common grouped protocol.

2. **A same-protocol external-baseline evaluation that retrains DRSN and TFNet
   adaptations with the same windows, folds, seeds, balanced label budget, and
   metrics as the internal models.**

   The contribution is the controlled comparison boundary. The external models
   are adaptations and are not presented as exact reproductions of their
   original experimental settings.

3. **An uncertainty-aware evidence set that retains negative and statistically
   non-significant outcomes, separates calibration reliability from
   classification accuracy, and quantifies the validation-to-test gap under
   joint domain shift.**

   This contribution emphasizes reporting practice and evidence quality rather
   than claiming that architectural complexity or self-training is inherently
   beneficial.

The wording avoids claiming a new general architecture, a new universal
benchmark, or state-of-the-art superiority.
