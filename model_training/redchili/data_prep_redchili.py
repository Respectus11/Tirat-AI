"""
data_prep_redchili.py — dataset loading, splitting, and phone-camera augmentation
for the Red Chili Powder adulteration model.

Parallel implementation to data_prep.py for Teff (does NOT touch teff files).

Expected layout:
    model_training/data_redchili/raw/<class_folders>/*.jpg (or .png/.jpeg)

Subfolders are auto-detected.

Dataset: "Red Chilli Adulteration Digital Image Dataset (DS-WH-1)".
  - DS I:  C1_PWH  = Category 1, Pure WH-chili      -> pure
           C2_AWH  = Category 2, Adulterated WH     -> adulterated
  - DS II: WH00    = 0% adulteration                -> pure
           WHBM/WHGM/WHRB/WHWS _{5,10,15}           -> adulterated

Verified visually + via dataset_structure.txt: C2_AWH is the ADULTERATED class
(brighter, different texture than C1_PWH). Mapping it to 'pure' poisons ~20% of
the labels (the bug that produced the coin-flip 49.7% model).
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
PURE_FOLDERS = ("C1_PWH", "WH00")  # only these are pure; everything else is adulterated
AUTOTUNE = tf.data.AUTOTUNE


class SlightBlur(tf.keras.layers.Layer):
    """Mild variable gaussian-ish blur via depthwise convolution.
    Same logic as Teff data_prep.py for phone-camera consistency.
    """

    def __init__(self, max_strength: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.max_strength = max_strength

    def call(self, inputs, training=None):
        if len(inputs.shape) == 3:
            inputs = tf.expand_dims(inputs, 0)
            out = self._apply_blur(inputs)
            return tf.squeeze(out, 0)
        return self._apply_blur(inputs)

    def _apply_blur(self, inputs):
        identity = tf.constant([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        smooth = tf.fill([3, 3], 1.0 / 9.0)
        alpha = tf.random.uniform([], 0.0, self.max_strength)
        kernel = (1.0 - alpha) * identity + alpha * smooth
        kernel = tf.reshape(kernel, [3, 3, 1, 1])
        kernel = tf.tile(kernel, [1, 1, tf.shape(inputs)[-1], 1])
        return tf.nn.depthwise_conv2d(inputs, kernel, strides=[1, 1, 1, 1], padding="SAME")


def _resize_short_side(img: tf.Tensor) -> tf.Tensor:
    shape = tf.shape(img)
    h, w = tf.cast(shape[0], tf.float32), tf.cast(shape[1], tf.float32)
    scale = IMG_SIZE / tf.minimum(h, w)
    new_h = tf.cast(tf.round(h * scale), tf.int32)
    new_w = tf.cast(tf.round(w * scale), tf.int32)
    return tf.image.resize(img, [new_h, new_w])


def _crop_center_224(img: tf.Tensor) -> tf.Tensor:
    shape = tf.shape(img)
    h, w = shape[0], shape[1]
    top = (h - IMG_SIZE) // 2
    left = (w - IMG_SIZE) // 2
    return img[top:top + IMG_SIZE, left:left + IMG_SIZE]


def _load_and_preprocess(path: tf.Tensor) -> tf.Tensor:
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = _resize_short_side(img)
    img = _crop_center_224(img)
    return tf.cast(img, tf.float32)


class Augmenter:
    def __init__(self, strong: bool = False):
        self.rot = 0.04 if not strong else 0.08
        self.bright = 0.20 if not strong else 0.35
        self.contrast = 0.20 if not strong else 0.35
        self.blur = SlightBlur(0.5 if not strong else 0.8)
        self.flip = tf.keras.layers.RandomFlip("horizontal")
        self.rot_layer = tf.keras.layers.RandomRotation(self.rot)
        self.bright_layer = tf.keras.layers.RandomBrightness(self.bright)

    def __call__(self, image: tf.Tensor) -> tf.Tensor:
        image = self.flip(image)
        image = self.rot_layer(image)
        image = self.bright_layer(image)
        image = tf.image.random_contrast(image, 1 - self.contrast, 1 + self.contrast)
        image = self.blur(image)
        return tf.clip_by_value(image, 0.0, 255.0)


def make_augmenter(strong: bool = False):
    return Augmenter(strong)


def make_dataset(paths, cls_labels, *, augment=False, shuffle=False,
                 batch_size=BATCH_SIZE, cache=True) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices(
        (np.asarray(paths), np.asarray(cls_labels, dtype=np.int32))
    )

    aug_fn = make_augmenter() if augment else None

    def _map(path, cls):
        img = _load_and_preprocess(path)
        if aug_fn is not None:
            img = aug_fn(img)
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

    pure_paths, pure_folders = [], []
    adulterated_dict = {}
    raw_classes = {}

    for folder in subfolders:
        name = folder.name
        is_pure = name.startswith(PURE_FOLDERS)
        target_cls = 0 if is_pure else 1
        raw_classes[name] = CLASS_NAMES[target_cls]

        files = sorted(
            p for p in folder.rglob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            and not p.name.startswith(".")
            and p.stat().st_size > 0
        )
        if is_pure:
            for p in files:
                pure_paths.append(str(p))
                pure_folders.append(name)
        else:
            adulterated_dict[name] = [str(p) for p in files]

    n_pure = len(pure_paths)
    print(f"[data_prep_redchili] Found {n_pure} pure images ({Counter(pure_folders)})")
    
    # Balance 1:1: sample equally across all adulterated classes to match n_pure
    adulterated_paths, adulterated_folders = [], []
    n_adulterated_folders = len(adulterated_dict)
    rng = np.random.RandomState(SEED)
    
    # Distribute samples as evenly as possible across all adulterant folders
    per_folder = n_pure // n_adulterated_folders
    remainder = n_pure % n_adulterated_folders
    
    for idx, (f_name, f_files) in enumerate(sorted(adulterated_dict.items())):
        sample_count = per_folder + (1 if idx < remainder else 0)
        chosen = rng.choice(f_files, size=min(sample_count, len(f_files)), replace=False)
        for p in chosen:
            adulterated_paths.append(p)
            adulterated_folders.append(f_name)

    print(f"[data_prep_redchili] Sampled {len(adulterated_paths)} adulterated images across {n_adulterated_folders} folders for 1:1 balance")

    paths = pure_paths + adulterated_paths
    cls_labels = [0] * len(pure_paths) + [1] * len(adulterated_paths)
    raw_folder_labels = pure_folders + adulterated_folders

    total = len(paths)
    print(f"[data_prep_redchili] Total dataset: {total} images (Pure: {len(pure_paths)}, Adulterated: {len(adulterated_paths)})")
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
        _, stats = build_file_lists()
        (OUTPUTS_DIR / "dataset_summary.json").write_text(json.dumps(stats, indent=2))
        print(json.dumps(stats, indent=2))
    except SystemExit as e:
        print(e)
