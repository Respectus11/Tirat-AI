"""
data_prep_redchili.py — dataset loading, splitting, and phone-camera augmentation
for the Red Chili Powder adulteration model.

Parallel implementation to data_prep.py for Teff (does NOT touch teff files).

Expected layout:
    model_training/data_redchili/raw/<class_folders>/*.jpg (or .png/.jpeg)

Subfolders are auto-detected. Classes with 'pure' or 'control' (case-insensitive)
are mapped to the 'pure' verdict class (index 0). All other folders map to 'adulterated' (index 1).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data_redchili" / "raw"
OUTPUTS_DIR = SCRIPT_DIR / "outputs"

IMG_SIZE = 224
BATCH_SIZE = 32
SEED = 42
TRAIN_FRAC = 0.80
VAL_FRAC = 0.10

CLASS_NAMES = ["pure", "adulterated"]
AUTOTUNE = tf.data.AUTOTUNE


class SlightBlur(tf.keras.layers.Layer):
    """Mild variable gaussian-ish blur via depthwise convolution.
    Same logic as Teff data_prep.py for phone-camera consistency.
    """

    def __init__(self, max_strength: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.max_strength = max_strength

    def call(self, inputs, training=None):
        identity = tf.constant([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        smooth = tf.fill([3, 3], 1.0 / 9.0)
        alpha = tf.random.uniform([], 0.0, self.max_strength)
        kernel = (1.0 - alpha) * identity + alpha * smooth
        kernel = tf.reshape(kernel, [3, 3, 1, 1])
        kernel = tf.tile(kernel, [1, 1, tf.shape(inputs)[-1], 1])
        return tf.nn.depthwise_conv2d(inputs, kernel, strides=[1, 1, 1, 1], padding="SAME")


def make_augmenter(strong: bool = False):
    """Augmentation mirroring teff pipeline (rotation, brightness/contrast, blur)."""
    rot = 0.04 if not strong else 0.08
    bright = 0.20 if not strong else 0.35
    contrast = 0.20 if not strong else 0.35
    blur = SlightBlur(0.5 if not strong else 0.8)

    def augment(image: tf.Tensor) -> tf.Tensor:
        image = tf.image.random_flip_left_right(image)
        image = tf.keras.layers.RandomRotation(rot)(image)
        image = tf.keras.layers.RandomBrightness(bright)(image)
        image = tf.image.random_contrast(image, 1 - contrast, 1 + contrast)
        image = blur(image)
        return tf.clip_by_value(image, 0.0, 255.0)

    return augment


def _load_and_preprocess(path: tf.Tensor) -> tf.Tensor:
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    return tf.cast(img, tf.float32)


def make_dataset(paths, cls_labels, *, augment=False, shuffle=False,
                 batch_size=BATCH_SIZE, cache=True) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices(
        (np.asarray(paths), np.asarray(cls_labels, dtype=np.int32))
    )

    def _map(path, cls):
        img = _load_and_preprocess(path)
        if augment:
            img = make_augmenter()(img)
        return img, cls

    ds = ds.map(_map, num_parallel_calls=AUTOTUNE)
    if cache:
        ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(1024, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


def build_file_lists(data_dir: Path | None = None):
    data_dir = data_dir or DATA_DIR
    paths, cls_labels, raw_folder_labels = [], [], []

    if not data_dir.exists() or not any(data_dir.iterdir()):
        raise SystemExit(
            f"\n[!] Red Chili dataset folder not found at {data_dir}\n"
            "    Please place the unzipped dataset subfolders inside model_training/data_redchili/raw/\n"
        )

    subfolders = [p for p in sorted(data_dir.iterdir()) if p.is_dir()]
    if not subfolders:
        raise SystemExit(f"[!] No subdirectories found in '{data_dir}'.")

    raw_classes = {}
    for folder in subfolders:
        name = folder.name.lower()
        if "pure" in name or "control" in name:
            target_cls = 0  # pure
        else:
            target_cls = 1  # adulterated

        raw_classes[folder.name] = CLASS_NAMES[target_cls]

        files = sorted(
            p for p in folder.rglob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and not p.name.startswith(".")
        )
        for p in files:
            paths.append(str(p))
            cls_labels.append(target_cls)
            raw_folder_labels.append(folder.name)

    total = len(paths)
    print(f"[data_prep_redchili] Found {total} images across {len(subfolders)} raw folders.")
    print(f"[data_prep_redchili] Folder mapping: {json.dumps(raw_classes, indent=2)}")

    indices = np.arange(total)
    train_idx, hold_idx = train_test_split(
        indices, train_size=TRAIN_FRAC, random_state=SEED,
        stratify=cls_labels,
    )
    rel_val = VAL_FRAC / (VAL_FRAC + (1 - TRAIN_FRAC - VAL_FRAC))
    val_idx, test_idx = train_test_split(
        hold_idx, train_size=rel_val, random_state=SEED,
        stratify=[cls_labels[i] for i in hold_idx],
    )

    def take(idxs):
        return (
            [paths[i] for i in idxs],
            [cls_labels[i] for i in idxs],
        )

    splits = {
        "train": take(train_idx),
        "val": take(val_idx),
        "test": take(test_idx),
    }
    stats = {
        "total": total,
        "raw_folder_counts": Counter(raw_folder_labels),
        "per_class": {CLASS_NAMES[i]: cls_labels.count(i) for i in range(len(CLASS_NAMES))},
        "sizes": {k: len(v[0]) for k, v in splits.items()},
    }
    return splits, stats


def get_datasets_redchili(cache: bool = True):
    splits, stats = build_file_lists()
    datasets = {
        name: make_dataset(p, c, augment=(name == "train"), shuffle=(name == "train"), cache=cache)
        for name, (p, c) in splits.items()
    }
    return datasets, stats, CLASS_NAMES


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data prep check for Red Chili")
    args = parser.parse_args()
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        _, stats, _ = build_file_lists()
        (OUTPUTS_DIR / "dataset_summary.json").write_text(json.dumps(stats, indent=2))
        print(json.dumps(stats, indent=2))
    except SystemExit as e:
        print(e)
