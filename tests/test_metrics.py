from __future__ import annotations

import numpy as np

from spindle_cnn.evaluation.metrics import metrics_from_logits


def test_perfect_classifier_has_unit_accuracy_and_macro_f1() -> None:
    labels = np.asarray([0, 1, 2, 3, 0, 1, 2, 3])
    logits = np.full((len(labels), 4), -10.0, dtype=np.float32)
    logits[np.arange(len(labels)), labels] = 10.0
    metrics = metrics_from_logits(logits, labels)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["ece"] < 1e-8
