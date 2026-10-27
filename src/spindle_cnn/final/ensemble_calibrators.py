"""Ensemble v2 (CWT only) + v3 (CWT+GRU) calibrated logits via simple average."""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..paths import OUTPUTS_DIR


CLASS_NAMES = [
    "Normal",
    "Motor1_Current_Over",
    "Motor1_Temperature_Over",
    "Motor1_Torque_Over",
    "Motor2_Current_Over",
    "Motor2_Temperature_Over",
    "Motor2_Torque_Over",
    "OilMistTemperature_High",
    "Motor1_Speed_Abnormal",
    "Motor2_Speed_Abnormal",
]


def main() -> None:
    v2_dir = OUTPUTS_DIR / "self_train_v2_calibrated"
    v3_dir = OUTPUTS_DIR / "self_train_v3_gru_calibrated"
    out_dir = OUTPUTS_DIR / "ensemble_v2_v3_gru"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load calibrated logits
    v2_logits = np.load(v2_dir / "calibrated_test_logits.npy")
    v3_logits = np.load(v3_dir / "calibrated_test_logits.npy")
    print(f"v2 logits: {v2_logits.shape}, v3 logits: {v3_logits.shape}")

    # Load true labels from test_predictions.csv
    v2_preds = pd.read_csv(v2_dir / "test_predictions.csv")
    y_true = v2_preds["true_label"].to_numpy(dtype=np.int64)

    # Average logits
    avg_logits = (v2_logits + v3_logits) / 2.0
    y_pred = np.argmax(avg_logits, axis=1).astype(np.int64)

    # Metrics
    all_labels = np.arange(len(CLASS_NAMES))
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, labels=all_labels, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, labels=all_labels, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=all_labels, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, labels=all_labels, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, labels=all_labels, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, labels=all_labels, average="weighted", zero_division=0)),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    # Classification report
    report = classification_report(
        y_true, y_pred, labels=np.arange(len(CLASS_NAMES)),
        target_names=CLASS_NAMES, zero_division=0, output_dict=True,
    )
    pd.DataFrame(report).transpose().to_csv(out_dir / "classification_report.csv", index=True)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(CLASS_NAMES)))
    cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
    cm_df.to_csv(out_dir / "confusion_matrix.csv", index=True)

    # Per-class comparison
    print("\n===== Ensemble vs Individual =====")
    print(f"{'Class':<28} {'v2 F1':>8} {'v3 F1':>8} {'Ens F1':>8}")
    print("-" * 56)

    v2_report = classification_report(
        y_true, np.argmax(v2_logits, axis=1), labels=np.arange(len(CLASS_NAMES)),
        target_names=CLASS_NAMES, zero_division=0, output_dict=True,
    )
    v3_report = classification_report(
        y_true, np.argmax(v3_logits, axis=1), labels=np.arange(len(CLASS_NAMES)),
        target_names=CLASS_NAMES, zero_division=0, output_dict=True,
    )

    for cls_name in CLASS_NAMES:
        v2_f1 = v2_report[cls_name]["f1-score"]
        v3_f1 = v3_report[cls_name]["f1-score"]
        ens_f1 = report[cls_name]["f1-score"]
        best = "*" if ens_f1 > max(v2_f1, v3_f1) else ""
        print(f"{cls_name:<28} {v2_f1:>8.4f} {v3_f1:>8.4f} {ens_f1:>8.4f} {best}")

    print("-" * 56)
    print(f"{'Macro Avg':<28} {v2_report['macro avg']['f1-score']:>8.4f} {v3_report['macro avg']['f1-score']:>8.4f} {metrics['f1_macro']:>8.4f}")
    print(f"{'Accuracy':<28} {float(np.mean(y_true == np.argmax(v2_logits, axis=1))):>8.4f} {float(np.mean(y_true == np.argmax(v3_logits, axis=1))):>8.4f} {metrics['accuracy']:>8.4f}")

    # Save predictions
    pred_df = v2_preds[["true_label", "true_label_name"]].copy()
    pred_df["pred_label"] = y_pred
    pred_df["pred_label_name"] = [CLASS_NAMES[int(v)] for v in y_pred]
    pred_df.to_csv(out_dir / "test_predictions.csv", index=False)

    print(f"\nEnsemble results saved to {out_dir}")


if __name__ == "__main__":
    main()
