"""
evaluate_redchili.py — metrics and evaluation for the Red Chili model.

Outputs in model_training/redchili/outputs/:
    confusion_matrix.png
    classification_report.txt
    eval_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay  # noqa: E402

from data_prep_redchili import OUTPUTS_DIR, CLASS_NAMES, get_datasets_redchili  # noqa: E402


def collect_predictions(model, ds) -> tuple[np.ndarray, np.ndarray]:
    ys_true, preds = [], []
    for imgs, targets in ds:
        out = model(imgs, training=False)
        preds.append(out.numpy())
        ys_true.append(targets.numpy())
    return np.concatenate(ys_true), np.concatenate(preds)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(OUTPUTS_DIR / "redchili_model.keras"))
    args = ap.parse_args()

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    model = tf.keras.models.load_model(args.model)

    datasets, _, _ = get_datasets_redchili(cache=True)
    test_ds = datasets["test"]

    y_true, probs = collect_predictions(model, test_ds)
    y_pred = probs.argmax(axis=1)

    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASS_NAMES)))
    report = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, digits=3, zero_division=0)

    print("\n=== Red Chili Model Evaluation ===")
    print(report)

    disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
    disp.plot(cmap="Reds", values_format="d")
    plt.title("Red Chili Verdict Confusion Matrix")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "confusion_matrix.png", dpi=120)
    plt.close()

    (OUTPUTS_DIR / "classification_report.txt").write_text(report)

    accuracy = float((y_true == y_pred).mean())
    summary = {
        "accuracy": accuracy,
        "confusion_matrix": cm.tolist(),
        "classes": CLASS_NAMES,
        "total_test_samples": len(y_true),
    }
    (OUTPUTS_DIR / "eval_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[evaluate_redchili] Saved evaluation outputs to {OUTPUTS_DIR}")


if __name__ == "__main__":
    main()
