"""
train_redchili.py — Red Chili model training (transfer learning with MobileNetV3Small).

Binary classification head:
    verdict : softmax over {pure, adulterated}

Training schedule:
    Phase 1 — frozen base, train head only
    Phase 2 — unfreeze top of base at 1e-5 LR
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import tensorflow as tf  # noqa: E402

from data_prep_redchili import OUTPUTS_DIR, CLASS_NAMES, get_datasets_redchili  # noqa: E402

INPUT_SIZE = 224
HEAD_UNITS = 128
DROPOUT = 0.3


def build_model() -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(INPUT_SIZE, INPUT_SIZE, 3), name="image")
    x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1.0)(inputs)  # -> [-1, 1]

    base = tf.keras.applications.MobileNetV3Small(
        include_top=False,
        weights="imagenet",
        input_shape=(INPUT_SIZE, INPUT_SIZE, 3),
        include_preprocessing=False,
        pooling=None,
    )
    base.trainable = False
    x = base(x, training=False)

    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dropout(DROPOUT, name="dropout")(x)
    x = tf.keras.layers.Dense(HEAD_UNITS, activation="relu", name="trunk")(x)

    verdict = tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax", name="verdict")(x)

    model = tf.keras.Model(inputs=inputs, outputs=verdict, name="redchili_mobilenetv3s")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def unfreeze_top(model: tf.keras.Model, n_layers: int = 40) -> None:
    model.get_layer("MobileNetV3Small").trainable = True
    backbone = model.get_layer("MobileNetV3Small")
    for layer in backbone.layers[:-n_layers]:
        layer.trainable = False
    for layer in backbone.layers[-n_layers:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True


def plot_curves(history: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for key in ("accuracy", "val_accuracy"):
        if key in history:
            axes[0].plot(history[key], label=key, marker=".")
    axes[0].set_title("Verdict accuracy")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    for key in ("loss", "val_loss"):
        if key in history:
            axes[1].plot(history[key], label=key, marker=".")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs-head", type=int, default=25, help="max epochs, phase 1")
    ap.add_argument("--epochs-ft", type=int, default=15, help="max epochs, phase 2")
    ap.add_argument("--skip-phase1", action="store_true",
                    help="skip Phase 1; load best_phase1.keras and run only Phase 2")
    args = ap.parse_args()

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_dir = OUTPUTS_DIR / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("[train_redchili] Loading datasets...")
    datasets, stats, class_names = get_datasets_redchili(cache=True)
    train_ds, val_ds, test_ds = datasets["train"], datasets["val"], datasets["test"]
    print(f"[train_redchili] {json.dumps(stats['sizes'])} — classes={class_names}")

    # Class weights: the corrected label mapping is imbalanced
    # (pure = C1_PWH+WH00 ~ 492 vs adulterated ~ 4735). Without weights the
    # model drifts to the majority class; for a screening tool a missed
    # adulterant (false 'pure') is the costlier error.
    per_class = stats["per_class"]
    n_cls = len(class_names)
    n_total = sum(per_class.values())
    class_weight = {i: n_total / (n_cls * per_class[class_names[i]]) for i in range(n_cls)}
    pretty = {class_names[k]: round(v, 3) for k, v in class_weight.items()}
    print(f"[train_redchili] per_class={per_class}")
    print(f"[train_redchili] class_weight={pretty}")

    es_kwargs = dict(monitor="val_accuracy", mode="max",
                     restore_best_weights=True, verbose=1)

    if args.skip_phase1:
        # Load the best Phase 1 checkpoint and run only Phase 2
        p1_path = ckpt_dir / "best_phase1.keras"
        print(f"[train_redchili] --skip-phase1: loading {p1_path}")
        model = tf.keras.models.load_model(p1_path)
        h1 = {}
        h1_time = 0.0
    else:
        # ---------------- Phase 1: heads on frozen backbone ----------------
        model = build_model()
        cbs1 = [
            tf.keras.callbacks.EarlyStopping(patience=5, **es_kwargs),
            tf.keras.callbacks.ModelCheckpoint(
                ckpt_dir / "best_phase1.keras", monitor="val_accuracy",
                mode="max", save_best_only=True, verbose=0),
            tf.keras.callbacks.CSVLogger(OUTPUTS_DIR / "log_phase1.csv"),
        ]
        t0 = time.time()
        h1 = model.fit(train_ds, validation_data=val_ds,
                       epochs=args.epochs_head, callbacks=cbs1,
                       class_weight=class_weight)
        h1_time = time.time() - t0
        print(f"[train_redchili] Phase 1 done in {h1_time/60:.1f} min")

    # ---------------- Phase 2: fine-tune top of backbone ----------------
    unfreeze_top(model, n_layers=40)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    cbs2 = [
        tf.keras.callbacks.EarlyStopping(patience=4, **es_kwargs),
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_dir / "best_phase2.keras", monitor="val_accuracy",
            mode="max", save_best_only=True, verbose=0),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                             patience=2, min_lr=1e-6),
        tf.keras.callbacks.CSVLogger(OUTPUTS_DIR / "log_phase2.csv"),
    ]
    t0 = time.time()
    h2 = model.fit(train_ds, validation_data=val_ds,
                   epochs=args.epochs_ft, callbacks=cbs2,
                   class_weight=class_weight)
    print(f"[train_redchili] Phase 2 done in {(time.time()-t0)/60:.1f} min")

    # ---------------- Artifacts ----------------
    if isinstance(h1, dict) and not h1:
        merged = dict(h2.history)
    else:
        merged = {k: list(h1.history[k]) + list(h2.history.get(k, [])) for k in h1.history}
    plot_curves(merged, OUTPUTS_DIR / "training_curves.png")

    final_path = OUTPUTS_DIR / "redchili_model.keras"
    model.save(final_path)

    best_val_acc = max(merged["val_accuracy"])
    summary = {
        "classes": class_names,
        "input": {"size": INPUT_SIZE, "dtype": "float32", "range": "0..255 (normalized inside graph)"},
        "best_val_verdict_accuracy": round(float(best_val_acc), 4),
        "epochs_phase1": len(h1.history["loss"]) if not isinstance(h1, dict) else 0,
        "epochs_phase2": len(h2.history["loss"]),
        "dataset_sizes": stats["sizes"],
    }
    (OUTPUTS_DIR / "training_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[train_redchili] Saved {final_path.name}, curves + summary to {OUTPUTS_DIR}")
    print(f"[train_redchili] Best val verdict accuracy: {best_val_acc:.3f}")


if __name__ == "__main__":
    main()
