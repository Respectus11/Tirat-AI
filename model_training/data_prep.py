"""
data_prep.py — dataset loading, splitting, augmentation, and caching for Tirat.

Expected layout (see NOTE.md):
    data/pure/*.jpg
    data/adulterated_wood/*.jpg
    data/adulterated_gypsum/*.jpg

Outputs a dict of tf.data.Dataset objects (train/val/test) consumed by
train.py / evaluate.py, plus dataset statistics.

WHY so much care in this file:
--------------------------------------------------------------
DOMAIN GAP is the #1 technical risk of this project. Our training images come
from a controlled lab setup; users will photograph flour on kitchen tables with
random phones, lighting, and backgrounds. The augmentation below deliberately
simulates *phone-like* conditions (rotation from hand tilt, brightness/contrast
jitter from auto-exposure, blur from missed focus) — but augmentation can only
narrow the gap, never close it. See docs/architecture.md for the field-data
plan that actually closes it.
--------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Configuration — single source of truth for the whole pipeline
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
OUTPUTS_DIR = SCRIPT_DIR / "outputs"

IMG_SIZE = 224          # MobileNetV3Small native input
BATCH_SIZE = 32
SEED = 42               # fixed seed => reproducible splits between runs
TRAIN_FRAC = 0.80
VAL_FRAC = 0.10         # test gets the remaining 0.10

# Fixed class order. This EXACT order becomes labels.json in the app — never
# reorder without re-exporting the model + labels together.
CLASS_DIRS = ["pure", "adulterated_wood", "adulterated_gypsum"]
CLASS_NAMES = ["pure", "wood", "gypsum"]  # short names shipped to the app UI

# Published-research bins: 10-40% w/w in 5% steps => 7 classes.
PCT_BINS = [10, 15, 20, 25, 30, 35, 40]

# Extracts e.g. "sample_25.jpg" -> 25, "wood_30pct_003.png" -> 30.
# ADJUST to your dataset's naming convention BEFORE training (see NOTE.md).
PCT_PATTERN = r"(?:^|[^0-9])(\d{1,2})\s*%?"

AUTOTUNE = tf.data.AUTOTUNE


def extract_pct_bin(filename: str) -> int:
    """Return index into PCT_BINS parsed from filename, or -1 if absent/out of range.

    -1 is used as an explicit 'unknown' marker downstream: those samples still train
    the verdict head but receive weak fallback pct labels (see build_file_lists).
    """
    m = re.search(PCT_PATTERN, Path(filename).stem)
    if not m:
        return -1
    value = int(m.group(1))
    if value in PCT_BINS:
        return PCT_BINS.index(value)
    return -1


class SlightBlur(tf.keras.layers.Layer):
    """Mild variable gaussian-ish blur via depthwise convolution.

    WHY custom: Keras has no RandomBlur layer, and real phone shots are often
    slightly out of focus. We mix an identity kernel with a small smoothing
    kernel using a random alpha per call, so some batches stay sharp and others
    get softly blurred — mimicking focus variance instead of applying uniform
    softness (which would just teach the model blurry = normal).
    """

    def __init__(self, max_strength: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.max_strength = max_strength

    def call(self, inputs, training=None):
        identity = tf.constant([[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]])
        smooth = tf.fill([3, 3], 1.0 / 9.0)
        alpha = tf.random.uniform([], 0.0, self.max_strength)
        kernel = (1.0 - alpha) * identity + alpha * smooth
        kernel = tf.reshape(kernel, [3, 3, 1, 1])
        kernel = tf.tile(kernel, [1, 1, tf.shape(inputs)[-1], 1])
        return tf.nn.depthwise_conv2d(inputs, kernel, strides=[1, 1, 1, 1], padding="SAME")


def make_augmenter(strong: bool = False):
    """Build the per-image augmentation function applied ONLY to training data.

    strong=False -> training-time augmentation (mild, keeps class signal intact)
    strong=True  -> evaluate.py's simulated-phone-photo stress test (aggressive)

    Applied via dataset.map OUTSIDE the model graph on purpose: Keras Random*
    layers inside a SavedModel convert unpredictably to TFLite, and we want the
    exported graph to be pure inference.
    """
    rot = 0.04 if not strong else 0.08            # fraction of 2*pi (~14 deg vs ~29 deg)
    bright = 0.20 if not strong else 0.35         # auto-exposure swing
    contrast = 0.20 if not strong else 0.35
    blur = SlightBlur(0.5 if not strong else 0.8)

    def augment(image: tf.Tensor) -> tf.Tensor:
        image = tf.image.random_flip_left_right(image)   # cheap free variation
        image = tf.keras.layers.RandomRotation(rot)(image)
        image = tf.keras.layers.RandomBrightness(bright)(image)
        image = tf.image.random_contrast(image, 1 - contrast, 1 + contrast)
        image = blur(image)
        return tf.clip_by_value(image, 0.0, 255.0)

    return augment


def _load_and_preprocess(path: tf.Tensor) -> tf.Tensor:
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = _resize_short_side(img)
    img = _crop_center_224(img)
    return tf.cast(img, tf.float32)


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


def make_dataset(paths, cls_labels, pct_labels, *, augment=False, shuffle=False,
                 batch_size=BATCH_SIZE, cache=True) -> tf.data.Dataset:
    """Assemble a tf.data.Dataset yielding (image, {'verdict': int32, 'pct_bins': int32})."""
    ds = tf.data.Dataset.from_tensor_slices(
        (np.asarray(paths), np.asarray(cls_labels, dtype=np.int32),
         np.asarray(pct_labels, dtype=np.int32))
    )

    def _map(path, cls, pct):
        img = _load_and_preprocess(path)
        if augment:
            img = make_augmenter()(img)
        return img, {"verdict": cls, "pct_bins": pct}

    ds = ds.map(_map, num_parallel_calls=AUTOTUNE)

    if cache:
        # In-memory caching AFTER decode+augment? No — cache() here caches the raw
        # decoded tensors before shuffling; augmentation runs per-epoch afterwards,
        # so each epoch sees fresh random crops/jitter while decode cost is paid once.
        ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(1024, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


def build_file_lists(data_dir: Path | None = None):
    """Scan folders, parse labels, do a STRATIFIED 80/10/10 split."""
    data_dir = data_dir or DATA_DIR
    paths, cls_labels, pct_labels = [], [], []

    if not data_dir.exists() or not any(data_dir.iterdir()):
        raise SystemExit(
            f"\n[!] No dataset found at {data_dir}\n"
            "    You must download it yourself — see NOTE.md for the source and layout.\n"
            "    Expected: data/pure, data/adulterated_wood, data/adulterated_gypsum\n"
        )

    missing_pct = 0
    for dir_name, cls_idx in zip(CLASS_DIRS, range(len(CLASS_DIRS))):
        folder = data_dir / dir_name
        files = sorted(
            p for p in folder.rglob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and not p.name.startswith(".")
        )
        if not files:
            raise SystemExit(f"[!] Folder '{folder}' is empty or missing.")
        bin_counts = Counter()
        for p in files:
            b = extract_pct_bin(p.name)
            bin_counts[b] += 1
            paths.append(str(p))
            cls_labels.append(cls_idx)
            pct_labels.append(b)
        n_fallback = sum(bin_counts[-1:])
        missing_pct += n_fallback

    # apply fallbacks
    by_cls_med = {}
    for idx, d in enumerate(CLASS_DIRS):
        bins = [b for c, b in zip(cls_labels, pct_labels) if c == idx and b >= 0]
        by_cls_med[idx] = PCT_BINS.index(int(np.median(bins))) if bins else 3  # default ~25%
    pct_labels_final = []
    for cls, b in zip(cls_labels, pct_labels):
        pct_labels_final.append(b if b >= 0 else by_cls_med[cls])

    total = len(paths)
    print(f"[data_prep] Found {total} images across {len(CLASS_DIRS)} classes.")
    print(f"[data_prep] Files WITHOUT parseable % level: {missing_pct} "
          f"({100.0 * missing_pct / total:.1f}%) — given weak class-median bin labels.")
    if missing_pct / total > 0.30:
        print("[!] WARNING: >30% of filenames lack a level tag. Fix PCT_PATTERN "
              "in data_prep.py before trusting the percentage head.")

    # Stratified split keeps every (class x pct-bin) cell proportionally represented.
    # With ~5000 images and 7 bins this matters: a naive shuffle could leave a whole
    # bin absent from val/test, making those metrics meaningless.
    indices = np.arange(total)
    train_idx, hold_idx = train_test_split(
        indices, train_size=TRAIN_FRAC, random_state=SEED,
        stratify=[f"{c}_{max(b,0)}" for c, b in zip(cls_labels, pct_labels_final)],
    )
    rel_val = VAL_FRAC / (VAL_FRAC + (1 - TRAIN_FRAC - VAL_FRAC))  # 0.5
    val_idx, test_idx = train_test_split(
        hold_idx, train_size=rel_val, random_state=SEED,
        stratify=[f"{c}_{max(b,0)}" for c, b in
                  [(cls_labels[i], pct_labels_final[i]) for i in hold_idx]],
    )

    def take(idxs):
        return (
            [paths[i] for i in idxs],
            [cls_labels[i] for i in idxs],
            [pct_labels_final[i] for i in idxs],
        )

    splits = {
        "train": take(train_idx),
        "val": take(val_idx),
        "test": take(test_idx),
    }
    stats = {
        "total": total,
        "per_class": {CLASS_DIRS[i]: cls_labels.count(i) for i in range(len(CLASS_DIRS))},
        "missing_pct_tags": missing_pct,
        "fallback_bins_by_class": {CLASS_DIRS[i]: PCT_BINS[by_cls_med[i]] for i in range(len(CLASS_DIRS))},
        "sizes": {k: len(v[0]) for k, v in splits.items()},
    }
    return splits, stats


def get_datasets(cache: bool = True):
    """Public API used by train.py / evaluate.py / convert_tflite.py."""
    splits, stats = build_file_lists()
    datasets = {
        name: make_dataset(p, c, b, augment=(name == "train"), shuffle=(name == "train"), cache=cache)
        for name, (p, c, b) in splits.items()
    }
    return datasets, stats, CLASS_NAMES, PCT_BINS


def preview_augmentations(out_path: Path = OUTPUTS_DIR / "augmentation_preview.png"):
    """Save one grid: original vs augmented variants of a single training image."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    splits, _ = build_file_lists()
    paths = splits["train"][0]
    img = _load_and_preprocess(tf.constant(paths[0]))
    aug = make_augmenter()

    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    axes[0][0].imshow(img.numpy().astype(np.uint8))
    axes[0][0].set_title("original")
    for ax in axes.flat[1:]:
        ax.imshow(aug(img).numpy().astype(np.uint8))
        ax.set_title("augmented")
    for ax in axes.flat:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"[data_prep] Augmentation preview saved -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset sanity check for Tirat")
    parser.add_argument("--preview", action="store_true",
                        help="also render an augmentation preview grid")
    args = parser.parse_args()

    _, stats, *_ = build_file_lists()
    OUTPUTS_DIR.mkdir(exist_ok=True)
    (OUTPUTS_DIR / "dataset_summary.json").write_text(json.dumps(stats, indent=2))

    print(json.dumps(stats, indent=2))
    print("\nSplit ratios:", {k: round(n / stats["total"], 3) for k, n in stats["sizes"].items()})
    if args.preview:
        preview_augmentations()
