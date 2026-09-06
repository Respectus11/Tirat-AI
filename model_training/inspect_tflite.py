"""inspect_tflite.py — print input/output tensor shapes of shipped TFLite models.

Phase 0 diagnostic: confirms whether the shipped redchili_model.tflite really has
a 2-class output (matching redchili_labels.json) or a stale 16-class output head,
which would make the app's tensor-matching fall back to the wrong buffer.

Run:  .venv/Scripts/python.exe inspect_tflite.py
"""
from __future__ import annotations

import json
from pathlib import Path

import tensorflow as tf  # noqa: F401  (interpreter lives here)

APP_MODELS = Path(__file__).resolve().parent.parent / "app" / "assets" / "models"

for name in ("redchili_model.tflite", "tirat_model.tflite"):
    path = APP_MODELS / name
    print(f"\n=== {name} ({path.stat().st_size:,} bytes) ===")
    interp = tf.lite.Interpreter(model_path=str(path))
    interp.allocate_tensors()
    for d in interp.get_input_details():
        print(f"  IN  {d['name']}: shape={d['shape'].tolist()} dtype={d['dtype'].__name__} "
              f"quant={d.get('quantization')}")
    for d in interp.get_output_details():
        print(f"  OUT {d['name']}: shape={d['shape'].tolist()} dtype={d['dtype'].__name__} "
              f"quant={d.get('quantization')}")
    # Byte-length the app would compute for each output buffer (float32 => 4 bytes/elem)
    for d in interp.get_output_details():
        n = 1
        for dim in d["shape"]:
            n *= int(dim)
        print(f"      -> byteLength if float32: {n * 4}")
