"""Classification and probability-quality metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)

from .calibration import probabilities_from_logits


def expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    error = 0.0
    for lower, upper in zip(bins[:-1], bins[1:], strict=True):
        mask = (confidence > lower) & (confidence <= upper)
        if not np.any(mask):
            continue
        accuracy = (predicted[mask] == labels[mask]).mean()
        error += mask.mean() * abs(accuracy - confidence[mask].mean())
    return float(error)


def _multiclass_brier(probabilities: np.ndarray, labels: np.ndarray) -> float:
    num_classes = probabilities.shape[1]
    scores = []
    for class_id in range(num_classes):
        target = (labels == class_id).astype(np.int64)
        scores.append(
            brier_score_loss(
                target,
                probabilities[:, class_id],
                pos_label=1,
            )
        )
    return float(np.mean(scores))


def metrics_from_logits(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    temperature: float = 1.0,
) -> dict[str, float]:
    probabilities = probabilities_from_logits(logits, temperature)
    predicted = probabilities.argmax(axis=1)
    num_classes = logits.shape[1]
    all_labels = np.arange(num_classes)
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "macro_f1": float(
            f1_score(labels, predicted, labels=all_labels, average="macro", zero_division=0)
        ),
        "macro_precision": float(
            precision_score(
                labels,
                predicted,
                labels=all_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                labels,
                predicted,
                labels=all_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "ece": expected_calibration_error(probabilities, labels),
        "nll": float(log_loss(labels, probabilities, labels=all_labels)),
        "brier": _multiclass_brier(probabilities, labels),
    }


def calibration_bin_table(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    n_bins: int = 15,
) -> list[dict[str, float | int]]:
    """Return reliability-diagram bins for CSV generation."""

    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    rows: list[dict[str, float | int]] = []
    for bin_id, (lower, upper) in enumerate(
        zip(
            np.linspace(0.0, 1.0, n_bins + 1)[:-1],
            np.linspace(0.0, 1.0, n_bins + 1)[1:],
            strict=True,
        )
    ):
        mask = (confidence > lower) & (confidence <= upper)
        rows.append(
            {
                "bin": bin_id,
                "lower": float(lower),
                "upper": float(upper),
                "count": int(mask.sum()),
                "mean_confidence": float(confidence[mask].mean()) if mask.any() else np.nan,
                "accuracy": float((predicted[mask] == labels[mask]).mean())
                if mask.any()
                else np.nan,
            }
        )
    return rows
