"""
evaluate.py — metrics + the domain-gap stress test.

Produces in outputs/:
    confusion_matrix.png     3-class verdict confusion matrix (test split)
    classification_report.txt per-class precision/recall/F1
    pct_metrics.json         adulteration-% bin accuracy + MAE (labeled files only)
    domain_gap_stress.json   accuracy when heavy phone-like degradation is applied

The stress test applies HEAVY augmentation to the held-out test set to estimate how
much accuracy we lose between lab conditions and messy real-world phone shots.
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
from sklearn.metrics import (classification_report, confusion_matrix,  # noqa: E402
                             ConfusionMatrixDisplay)

from data_prep import (OUTPUTS_DIR, PCT_BINS, CLASS_NAMES,  # noqa: E402
                       get_datasets, make_dataset)

CONF_THRESHOLD_NOTE = (
    "App-side fallback: predictions below 70% confidence return 'Inconclusive' instead "
    "of guessing. This trades recall for trust — see docs/architecture.md."
)


def collect_predictions(model, ds) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ys_true_cls, ys_true_bin, preds = [], [], []
    for imgs, targets in ds:
        out = model(imgs, training=False)
        preds.append(np.concatenate([out["verdict"].numpy(), out["pct_bins"].numpy()], axis=1))
        ys_true_cls.append(targets["verdict"].numpy())
        ys_true_bin.append(targets["pct_bins"].numpy())
    pred = np.concatenate(preds)
    return (np.concatenate(ys_true_cls), np.concatenate(ys_true_bin),
            pred[:, :len(CLASS_NAMES)], pred[:, len(CLASS_NAMES):])


def evaluate_verdict(y_true, y_pred, title: str, out_png: Path | None = None) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASS_NAMES)))
    report = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, digits=3, zero_division=0)

    print(f"\n=== {title} ===")
    print(report)

    if out_png:
        disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
        disp.plot(cmap="Blues", values_format="d")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_png, dpi=120)
        plt.close()

    return {
        "confusion_matrix": cm.tolist(),
        "report": report,
        "accuracy": float((y_true == y_pred).mean()),
    }


def expected_pct(pct_probs: np.ndarray) -> np.ndarray:
    """Probability-weighted bin value — gives users a smoother % than argmax."""
    return pct_probs @ np.asarray(PCT_BINS, dtype=np.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(OUTPUTS_DIR / "tirat_model.keras"))
    args = ap.parse_args()

    OUTPUTS_DIR.mkdir(exist_ok=True)
    model = tf.keras.models.load_model(args.model)

    datasets, _, _, _ = get_datasets(cache=True)
    test_ds = datasets["test"]

    y_cls, y_bin, verdict_probs, pct_probs = collect_predictions(model, test_ds)
    y_pred = verdict_probs.argmax(axis=1)
    confidence = verdict_probs.max(axis=1)

    clean = evaluate_verdict(y_cls, y_pred, "Clean lab-condition test set",
                             OUTPUTS_DIR / "confusion_matrix.png")
    (OUTPUTS_DIR / "classification_report.txt").write_text(clean["report"])

    # ---- Percentage-head metrics: only over samples with REAL level labels ----
    labeled = y_bin >= 0
    if labeled.any():
        bin_pred = pct_probs[labeled].argmax(axis=1)
        bin_acc = float((bin_pred == y_bin[labeled]).mean())
        mae = float(np.abs(expected_pct(pct_probs[labeled]) -
                           np.asarray(PCT_BINS)[y_bin[labeled]]).mean())
        pct_metrics = {"binned_accuracy": bin_acc, "mae_percentage_points": mae,
                       "n_labeled_samples": int(labeled.sum()),
                       "note": "Files lacking a filename % tag were excluded here."}
        print(f"\n=== Adulteration %% head ===\nbinned acc={bin_acc:.3f} "
              f"MAE={mae:.1f} pts over {int(labeled.sum())} labeled samples")
    else:
        pct_metrics = {"note": "No filenames carried a parseable % tag — fix PCT_PATTERN "
                               "in data_prep.py to enable percentage-head metrics."}
        print("\n[!] No labeled % samples found; skipping pct-head metrics.")
    (OUTPUTS_DIR / "pct_metrics.json").write_text(json.dumps(pct_metrics, indent=2))

    # ------------------------------------------------------------------
    # DOMAIN-GAP STRESS TEST
    #
    # We degrade the TEST set as if taken by a careless user: harsh exposure swings,
    # stronger rotation/blur, sensor noise, and lossy JPEG re-compression (phones save
    # JPEG; lab sets are usually PNG — the codec artifacts alone shift predictions).
    #
    # WHY this matters: this number is our cheapest honest estimate of field accuracy
    # BEFORE spending money on a pilot. But read it as OPTIMISTIC: background and
    # lighting geometry still match the lab. A true field estimate needs crowdsourced
    # photos re-verified by a lab sample (docs/architecture.md).
    # ------------------------------------------------------------------
    def stress(image: tf.Tensor) -> tf.Tensor:
        image = tf.image.random_flip_left_right(image)
        image = tf.keras.layers.RandomRotation(0.08)(image)
        image = tf.image.random_brightness(image, 0.35)
        image = tf.image.random_contrast(image, 0.65, 1.35)
        noise = tf.random.normal(tf.shape(image), stddev=8.0)   # ~ISO-800 grain
        image = image + noise
        image = tf.image.adjust_jpeg_quality(
            tf.clip_by_value(image, 0.0, 255.0), quality=45)
        return tf.cast(tf.clip_by_value(image, 0.0, 255.0), tf.float32)

    # Rebuild the test split as an un-batched dataset so per-image stress mapping is clean
    from data_prep import build_file_lists
    file_splits, _ = build_file_lists()
    stressed_ds = make_dataset(*file_splits["test"], augment=False,
                               shuffle=False).unbatch().map(
        lambda img, t: (stress(img), t)).batch(32).prefetch(tf.data.AUTOTUNE)

    y_cls_s, y_bin_s, vp_s, _ = collect_predictions(model, stressed_ds)
    stressed = evaluate_verdict(y_cls_s, vp_s.argmax(axis=1),
                                "Simulated phone-photo test set",
                                OUTPUTS_DIR / "confusion_matrix_stressed.png")

    # Confidence-threshold behaviour under stress: what fraction survives the 70% gate?
    kept = vp_s.max(axis=1) >= 0.70
    gated_acc = float((vp_s[kept].argmax(axis=1) == y_cls_s[kept]).mean()) if kept.any() else 0.0

    result = {
        "clean_accuracy": clean["accuracy"],
        "stressed_accuracy": stressed["accuracy"],
        "estimated_field_drop_points": round(100*(clean["accuracy"] - stressed["accuracy"]), 1),
        "stress_kept_fraction_at_0.70": float(kept.mean()),
        "stress_accuracy_after_0.70_gate": gated_acc,
        "note": CONF_THRESHOLD_NOTE,
    }
    (OUTPUTS_DIR / "domain_gap_stress.json").write_text(json.dumps(result, indent=2))
    print("\n=== Domain-gap stress ===")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
