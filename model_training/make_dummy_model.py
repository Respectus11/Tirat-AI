"""
make_dummy_model.py — generate a VALID but RANDOM-WEIGHTS .tflite placeholder.

Purpose: let the mobile app build, load a real TFLite file, and run the complete
capture -> inference -> history flow BEFORE the real dataset exists or training runs.
Predictions are meaningless noise by design — this is plumbing, not science.

Run once after `pip install -r requirements.txt`:

    python make_dummy_model.py

Writes:
    ../app/assets/models/tirat_model.tflite   (same filename the real model uses,
                                               same IO shapes, same internal
                                               normalization => identical app code path)
    ../app/assets/models/labels.json
"""

from __future__ import annotations

import json
from pathlib import Path

import tensorflow as tf

from data_prep import CLASS_NAMES, PCT_BINS

INPUT_SIZE = 224
APP_MODELS_DIR = Path(__file__).resolve().parent.parent / "app" / "assets" / "models"


def build_dummy() -> tf.keras.Model:
    """Tiny convnet with the EXACT same interface as the real model.

    Deliberately NOT MobileNetV3: that would trigger an ImageNet weight download,
    which we avoid here (offline-first, zero-surprise scaffolding).
    Includes the same Rescaling(1/127.5, offset=-1) layer so the mobile app's
    preprocessing contract (raw 0..255 floats) holds for dummy AND real models.
    """
    inputs = tf.keras.Input(shape=(INPUT_SIZE, INPUT_SIZE, 3), name="image")
    x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1.0)(inputs)
    x = tf.keras.layers.Conv2D(8, 5, strides=2, activation="relu")(x)
    x = tf.keras.layers.Conv2D(16, 3, strides=2, activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    verdict = tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax", name="verdict")(x)
    pct = tf.keras.layers.Dense(len(PCT_BINS), activation="softmax", name="pct_bins")(x)
    return tf.keras.Model(inputs, {"verdict": verdict, "pct_bins": pct})


def main() -> None:
    APP_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model = build_dummy()
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]   # dynamic-range quant: small & valid
    tflite_bytes = converter.convert()

    out_path = APP_MODELS_DIR / "tirat_model.tflite"
    out_path.write_bytes(tflite_bytes)

    labels = {
        "classes": CLASS_NAMES,
        "pct_bins": PCT_BINS,
        "input": {"size": INPUT_SIZE, "dtype": "float32", "range": "0..255",
                  "note": "normalization happens inside the model"},
        "outputs": {
            "verdict": {"shape": [1, len(CLASS_NAMES)], "activation": "softmax"},
            "pct_bins": {"shape": [1, len(PCT_BINS)], "activation": "softmax"},
        },
        "confidence_threshold": 0.70,
        "dummy": True,
    }
    (APP_MODELS_DIR / "labels.json").write_text(json.dumps(labels, indent=2))

    print(f"[dummy] Wrote {out_path} ({out_path.stat().st_size/1024:.0f} KB) + labels.json")
    print("[dummy] WARNING: weights are random. Verdicts are NOISE until you train")
    print("[dummy] the real model (see NOTE.md) and run convert_tflite.py.")


if __name__ == "__main__":
    main()
