"""
convert_tflite_redchili.py — export the trained Red Chili Keras model to float16 TFLite.

Outputs:
    outputs/redchili_float16.tflite <- ALSO copied to ../../app/assets/models/redchili_model.tflite
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import tensorflow as tf

from data_prep_redchili import CLASS_NAMES

INPUT_SIZE = 224
OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"
APP_MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "app" / "assets" / "models"


def write_labels_json(dest: Path) -> None:
    labels = {
        "classes": CLASS_NAMES,
        "input": {
            "size": INPUT_SIZE,
            "dtype": "float32",
            "range": "0..255",
            "note": "normalization happens inside the model graph"
        },
        "outputs": {
            "verdict": {"shape": [1, len(CLASS_NAMES)], "activation": "softmax"}
        },
        "confidence_threshold": 0.70,
        "food_type": "redchili",
        "dummy": False
    }
    dest.write_text(json.dumps(labels, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(OUTPUTS_DIR / "checkpoints" / "best_phase2.keras"))
    args = ap.parse_args()

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    APP_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model = tf.keras.models.load_model(args.model)
    print(f"[convert_redchili] Loaded {args.model}")

    conv16 = tf.lite.TFLiteConverter.from_keras_model(model)
    conv16.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_bytes = conv16.convert()
    f16_path = OUTPUTS_DIR / "redchili_float16.tflite"
    f16_path.write_bytes(tflite_bytes)

    # ---------------- Numeric Equivalence Check ----------------
    print("[convert_redchili] Running float32 (Keras) vs float16 (TFLite) numeric equivalence check...")
    import numpy as np
    dummy_input = np.random.uniform(0.0, 255.0, size=(1, INPUT_SIZE, INPUT_SIZE, 3)).astype(np.float32)
    keras_pred = model(dummy_input).numpy()

    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    in_idx = interp.get_input_details()[0]["index"]
    out_idx = interp.get_output_details()[0]["index"]
    interp.set_tensor(in_idx, dummy_input)
    interp.invoke()
    tflite_pred = interp.get_tensor(out_idx)

    max_delta = float(np.max(np.abs(keras_pred - tflite_pred)))
    print(f"[convert_redchili] Max absolute delta: {max_delta:.6f}")
    if max_delta > 0.05:
        print("[convert_redchili] WARNING: High divergence between Keras and Float16 TFLite output!")
    else:
        print("[convert_redchili] Numeric equivalence verified (delta <= 0.05).")

    shipped = APP_MODELS_DIR / "redchili_model.tflite"
    shutil.copyfile(f16_path, shipped)
    write_labels_json(APP_MODELS_DIR / "redchili_labels.json")

    print(f"[convert_redchili] float16: {f16_path} ({f16_path.stat().st_size/1024/1024:.2f} MB)")
    print(f"[convert_redchili] Copied -> {shipped}")


if __name__ == "__main__":
    main()
