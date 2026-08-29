"""
train.py — Tirat model training (transfer learning with MobileNetV3Small).

Two heads on a shared trunk:
    verdict   : softmax over {pure, wood, gypsum}
    pct_bins  : softmax over 7 bins [10,15,20,25,30,35,40] % w/w

WHY MobileNetV3Small:
    - Target device is a mid-range Android phone. V3Small is ~2.5M params,
      runs real-time on CPU with the TFLite CPU/GPU delegates, and its
      squeeze-and-excite blocks are good at fine texture cues (fiber strands vs
      crystalline specks) that distinguish adulterants.
    - Transfer learning from ImageNet is essential: ~5k photos is far too little
      data to learn low-level visual features from scratch.

Training schedule:
    Phase 1 — frozen base, train heads only (stabilizes fresh head weights)
    Phase 2 — unfreeze top of base at tiny LR (adapt features to flour texture)
Both phases use early stopping + checkpointing on val verdict accuracy.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (matplotlib import after Agg)
import tensorflow as tf  # noqa: E402

from data_prep import OUTPUTS_DIR, PCT_BINS, CLASS_NAMES, get_datasets  # noqa: E402

INPUT_SIZE = 224
HEAD_UNITS = 128        # shared trunk width before both heads
DROPOUT = 0.3
PCT_LOSS_WEIGHT = 0.5   # pct head is secondary info; don't let it fight the verdict head


def build_model() -> tf.keras.Model:
    """MobileNetV3Small backbone + shared dense trunk + two softmax heads.

    WHY one normalization layer INSIDE the graph:
        The app feeds raw 0..255 float pixels (simplest possible mobile pipeline).
        Baking Rescaling into the model means training, evaluation, and TFLite all
        share identical preprocessing — no train/serve skew from re-implementing it
        twice (once in Python, once in TypeScript).
    """
    inputs = tf.keras.Input(shape=(INPUT_SIZE, INPUT_SIZE, 3), name="image")
    x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1.0)(inputs)  # -> [-1, 1]

    base = tf.keras.applications.MobileNetV3Small(
        include_top=False,
        weights="imagenet",
        input_shape=(INPUT_SIZE, INPUT_SIZE, 3),
        include_preprocessing=False,   # we normalize explicitly above
        pooling=None,
    )
    base.trainable = False             # phase 1: frozen feature extractor
    x = base(x, training=False)

    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dropout(DROPOUT, name="dropout")(x)
    x = tf.keras.layers.Dense(HEAD_UNITS, activation="relu", name="trunk")(x)

    verdict = tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax", name="verdict")(x)
    pct = tf.keras.layers.Dense(len(PCT_BINS), activation="softmax", name="pct_bins")(x)

    model = tf.keras.Model(inputs=inputs, outputs={"verdict": verdict, "pct_bins": pct},
                           name="tirat_mobilenetv3s")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss={
            "verdict": "sparse_categorical_crossentropy",
            "pct_bins": "sparse_categorical_crossentropy",
        },
        loss_weights={"verdict": 1.0, "pct_bins": PCT_LOSS_WEIGHT},
        metrics={"verdict": ["accuracy"], "pct_bins": ["accuracy"]},
    )
    return model


def unfreeze_top(model: tf.keras.Model, n_layers: int = 40) -> None:
    """Unfreeze the last `n_layers` of the backbone for fine-tuning.

    WHY not the whole base:
        Early MobileNet layers learn generic edges/colors that transfer perfectly;
        only the later blocks need to adapt to flour texture. Also, BatchNorm
        statistics computed on ImageNet would be destroyed by tiny-data fine-tuning,
        so BN layers stay frozen even where convs unfreeze — standard practice for
        small datasets.
    """
    model.get_layer("mobilenetv3small").trainable = True
    backbone = model.get_layer("mobilenetv3small")
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
        axes[0].plot(history[key], label=key, marker=".")
    axes[0].set_title("Verdict accuracy")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    total_loss = [a + PCT_LOSS_WEIGHT * b for a, b in
                  zip(history["loss"], history["pct_bins_loss"])]
    total_val_loss = [a + PCT_LOSS_WEIGHT * b for a, b in
                      zip(history["val_loss"], history["val_pct_bins_loss"])]
    axes[1].plot(total_loss, label="train (weighted total)", marker=".")
    axes[1].plot(total_val_loss, label="val (weighted total)", marker=".")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs-head", type=int, default=30, help="max epochs, phase 1")
    ap.add_argument("--epochs-ft", type=int, default=20, help="max epochs, phase 2")
    args = ap.parse_args()

    OUTPUTS_DIR.mkdir(exist_ok=True)
    ckpt_dir = OUTPUTS_DIR / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    print("[train] Loading datasets (see data_prep.py for split/augmentation policy)...")
    datasets, stats, class_names, pct_bins = get_datasets(cache=True)
    train_ds, val_ds, test_ds = datasets["train"], datasets["val"], datasets["test"]
    print(f"[train] {json.dumps(stats['sizes'])} — classes={class_names}, bins={pct_bins}")

    # DOMAIN-GAP note: everything we can do about lab-vs-phone differences at THIS stage
    # is augmentation (already baked into train_ds) plus a strict capture protocol in the
    # app (flat frame, plain background, flash on) that pulls user photos toward lab
    # conditions. Real-world validation still requires field photos — see docs/.

    es_kwargs = dict(monitor="val_verdict_accuracy", mode="max",
                     restore_best_weights=True, verbose=1)

    # ---------------- Phase 1: heads on frozen backbone ----------------
    model = build_model()
    cbs1 = [
        tf.keras.callbacks.EarlyStopping(patience=6, **es_kwargs),
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_dir / "best_phase1.keras", monitor="val_verdict_accuracy",
            mode="max", save_best_only=True, verbose=0),
        tf.keras.callbacks.CSVLogger(OUTPUTS_DIR / "log_phase1.csv"),
    ]
    t0 = time.time()
    h1 = model.fit(train_ds, validation_data=val_ds,
                   epochs=args.epochs_head, callbacks=cbs1)
    print(f"[train] Phase 1 done in {(time.time()-t0)/60:.1f} min")

    # ---------------- Phase 2: fine-tune top of backbone ----------------
    unfreeze_top(model, n_layers=40)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),   # tiny LR: don't wreck pretrained weights
        loss={"verdict": "sparse_categorical_crossentropy",
              "pct_bins": "sparse_categorical_crossentropy"},
        loss_weights={"verdict": 1.0, "pct_bins": PCT_LOSS_WEIGHT},
        metrics={"verdict": ["accuracy"], "pct_bins": ["accuracy"]},
    )
    cbs2 = [
        tf.keras.callbacks.EarlyStopping(patience=4, **es_kwargs),
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_dir / "best_phase2.keras", monitor="val_verdict_accuracy",
            mode="max", save_best_only=True, verbose=0),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                             patience=2, min_lr=1e-6),
        tf.keras.callbacks.CSVLogger(OUTPUTS_DIR / "log_phase2.csv"),
    ]
    t0 = time.time()
    h2 = model.fit(train_ds, validation_data=val_ds,
                   epochs=args.epochs_ft, callbacks=cbs2)
    print(f"[train] Phase 2 done in {(time.time()-t0)/60:.1f} min")

    # ---------------- Artifacts ----------------
    merged = {k: list(h1.history[k]) + list(h2.history.get(k, [])) for k in h1.history}
    plot_curves(merged, OUTPUTS_DIR / "training_curves.png")

    final_path = OUTPUTS_DIR / "tirat_model.keras"
    model.save(final_path)

    best_val_acc = max(merged["val_verdict_accuracy"])
    summary = {
        "classes": class_names,
        "pct_bins": pct_bins,
        "input": {"size": INPUT_SIZE, "dtype": "float32", "range": "0..255 (normalized inside graph)"},
        "best_val_verdict_accuracy": round(float(best_val_acc), 4),
        "final_val_pct_accuracy": round(float(merged["val_pct_bins_accuracy"][-1]), 4),
        "epochs_phase1": len(h1.history["loss"]),
        "epochs_phase2": len(h2.history["loss"]),
        "dataset_sizes": stats["sizes"],
    }
    (OUTPUTS_DIR / "training_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[train] Saved {final_path.name}, curves + summary to {OUTPUTS_DIR}")
    print(f"[train] Best val verdict accuracy: {best_val_acc:.3f}")
    print("[train] Next: python evaluate.py, then python convert_tflite.py")


if __name__ == "__main__":
    main()
