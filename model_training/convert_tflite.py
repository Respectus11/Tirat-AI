"""
convert_tflite.py — export the trained Keras model to quantized TFLite for mobile.

Outputs:
    outputs/tirat_float16.tflite   <- ALSO copied to ../app/assets/models/tirat_model.tflite
    outputs/tirat_int8.tflite      <- kept for future optimization work only

WHY float16 is what we ship:
    float16 weights halve model size (~4-5 MB) while keeping FLOAT32 inputs/outputs.
    That means the app can feed normalized floats directly — one simple code path.
    Full int8 quantization shrinks further (~1.2 MB) and runs faster on some DSPs,
    but forces uint8 input/output tensors, i.e. a second preprocessing pipeline in
    TypeScript plus calibration risk. Ship float16 now; benchmark int8 later.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import tensorflow as tf

from data_prep import CLASS_NAMES, PCT_BINS, get_datasets

INPUT_SIZE = 224
OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"
APP_MODELS_DIR = Path(__file__).resolve().parent.parent / "app" / "assets" / "models"
N_CALIBRATION_BATCHES = 120   # ~3800 images at batch 32: enough to span activation ranges


def representative_dataset():
    """Sample of training images used to calibrate int8 activation ranges.

    Uses RAW 0..255 float pixels because normalization lives INSIDE the graph —
    calibration must see exactly what inference will see (no train/serve skew).
    """
    datasets, _, _, _ = get_datasets(cache=False)
    for imgs, _ in datasets["train"].take(N_CALIBRATION_BATCHES):
        yield [tf.cast(imgs, tf.float32)]


def write_labels_json(dest: Path) -> None:
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
        "dummy": False,
    }
    dest.write_text(json.dumps(labels, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(OUTPUTS_DIR / "tirat_model.keras"))
    ap.add_argument("--skip-int8", action="store_true",
                    help="skip int8 export when no dataset is present")
    args = ap.parse_args()

    OUTPUTS_DIR.mkdir(exist_ok=True)
    APP_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model = tf.keras.models.load_model(args.model)
    print(f"[convert] Loaded {args.model}")

    # ---------------- float16 (ships to the app) ----------------
    conv16 = tf.lite.TFLiteConverter.from_keras_model(model)
    conv16.optimizations = [tf.lite.Optimize.DEFAULT]
    conv16.target_spec.supported_types = [tf.float16]
    f16_path = OUTPUTS_DIR / "tirat_float16.tflite"
    f16_path.write_bytes(conv16.convert())

    shipped = APP_MODELS_DIR / "tirat_model.tflite"
    shutil.copyfile(f16_path, shipped)
    write_labels_json(APP_MODELS_DIR / "labels.json")

    print(f"[convert] float16: {f16_path} ({f16_path.stat().st_size/1024/1024:.2f} MB)")
    print(f"[convert] Copied -> {shipped} (this file is bundled into the app)")

    # ---------------- full int8 (experimental artifact) ----------------
    if args.skip_int8:
        print("[convert] Skipping int8 (--skip-int8).")
        return
    try:
        conv8 = tf.lite.TFLiteConverter.from_keras_model(model)
        conv8.optimizations = [tf.lite.Optimize.DEFAULT]
        conv8.representative_dataset = representative_dataset
        conv8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        # NOTE: uint8 IO on purpose — documents that adopting this variant requires
        # changes to the app's preprocessing pipeline (see module docstring).
        conv8.inference_input_type = tf.uint8
        conv8.inference_output_type = tf.uint8
        i8_path = OUTPUTS_DIR / "tirat_int8.tflite"
        i8_path.write_bytes(conv8.convert())
        print(f"[convert] int8: {i8_path} ({i8_path.stat().st_size/1024/1024:.2f} MB)"
              " — NOT shipped; benchmark before switching.")
    except Exception as exc:  # noqa: BLE001 — conversion is best-effort scaffolding
        print(f"[convert] int8 export failed ({exc}). "
              "Float16 was still exported successfully.")

    # ---------------- optional TFLite metadata (tflite-support) ----------------
    # Embeds class labels into the model file itself so third-party tooling can read
    # them. Nice-to-have; guarded so its absence never blocks the pipeline.
    # TODO: verify exact MetadataWriter API against the installed tflite-support
    # version when enabling this — the API shifts between releases.
    try:
        import tflite_support  # type: ignore  # noqa: F401

        txt = OUTPUTS_DIR / "labels.txt"
        txt.write_text("\n".join(CLASS_NAMES))
        print(f"[convert] labels.txt written ({txt}) — wire up MetadataWriter "
              "(see TODO above) if you want metadata embedded in the .tflite.")
    except ImportError:
        print("[convert] tflite-support not installed — skipping metadata "
              "(optional; see requirements.txt).")


if __name__ == "__main__":
    main()
