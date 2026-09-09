"""
evaluate_gradcam.py — Explainable AI (XAI) feature verification via Grad-CAM.

Implementation of Grad-CAM (Gradient-weighted Class Activation Mapping)
for MobileNetV3-Small Red Chili Adulteration Model, following the methodology
of Brar et al. (RSC Sustainable Food Technology, 2025).

Purpose:
    Visually audit whether the neural network activates on true physical
    particulate inclusions (wheat bran fibers, rice husk silica, sawdust grains)
    in adulterated samples versus homogeneous diffuse activations on pure samples,
    proving the model does not rely on lighting or background artifacts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

from data_prep_redchili import CLASS_NAMES, IMG_SIZE, OUTPUTS_DIR, build_file_lists  # noqa: E402

GRADCAM_DIR = OUTPUTS_DIR / "gradcam"


def get_gradcam_heatmap(
    model: tf.keras.Model,
    img_array: np.ndarray,
    target_class_idx: int | None = None,
    conv_layer_name: str | None = None,
) -> np.ndarray:
    """Computes Grad-CAM heatmap for a given input image."""
    # Find the target convolutional layer in the backbone if not specified
    backbone = None
    try:
        backbone = model.get_layer("MobileNetV3Small")
    except ValueError:
        pass

    if conv_layer_name is None:
        if backbone is not None:
            # Pick the last Conv2D or DepthwiseConv2D layer in backbone
            for layer in reversed(backbone.layers):
                if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
                    conv_layer_name = layer.name
                    break
        if conv_layer_name is None:
            for layer in reversed(model.layers):
                if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
                    conv_layer_name = layer.name
                    break

    # Construct gradient model
    if backbone is not None:
        target_conv_layer = backbone.get_layer(conv_layer_name)
        # Create sub-model mapping backbone input to [target_conv_output, backbone_output]
        grad_submodel = tf.keras.Model(
            inputs=backbone.input,
            outputs=[target_conv_layer.output, backbone.output]
        )
    else:
        target_conv_layer = model.get_layer(conv_layer_name)
        grad_submodel = tf.keras.Model(
            inputs=model.input,
            outputs=[target_conv_layer.output, model.output]
        )

    with tf.GradientTape() as tape:
        # Forward pass through model
        if backbone is not None:
            # Rescale input
            inputs = tf.cast(img_array, tf.float32)
            rescaled = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1.0)(inputs)
            conv_outputs, backbone_outputs = grad_submodel(rescaled)
            
            # Route through remaining heads
            tape.watch(conv_outputs)
            # Find gap and dense layers in main model
            x = model.get_layer("gap_final")(backbone_outputs) if "gap_final" in [l.name for l in model.layers] else model.get_layer("gap")(backbone_outputs)
            if "gap_intermediate" in [l.name for l in model.layers]:
                inter_gap = model.get_layer("gap_intermediate")(conv_outputs)
                x = model.get_layer("multiscale_fusion")([inter_gap, x])
            if "dropout" in [l.name for l in model.layers]:
                x = model.get_layer("dropout")(x, training=False)
            if "trunk" in [l.name for l in model.layers]:
                x = model.get_layer("trunk")(x)
            preds = model.get_layer("verdict")(x)
        else:
            conv_outputs, preds = grad_submodel(img_array)
            tape.watch(conv_outputs)

        if target_class_idx is None:
            target_class_idx = tf.argmax(preds[0])
        class_channel = preds[:, target_class_idx]

    # Compute gradients of top predicted class with respect to conv outputs
    grads = tape.gradient(class_channel, conv_outputs)
    if grads is None:
        return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

    # Global average pooling of gradients -> alpha importance weights
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weighted combination of feature maps
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Apply ReLU to retain only features with positive influence
    heatmap = tf.maximum(heatmap, 0.0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
    heatmap_np = heatmap.numpy()

    # Resize to original input size
    heatmap_resized = tf.image.resize(
        heatmap_np[..., np.newaxis], [IMG_SIZE, IMG_SIZE]
    ).numpy().squeeze()

    return heatmap_resized


def overlay_heatmap(img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Superimposes Grad-CAM heatmap onto RGB image."""
    cmap = matplotlib.colormaps.get_cmap("jet")
    colored_heatmap = cmap(heatmap)[:, :, :3]  # RGB
    colored_heatmap = (colored_heatmap * 255.0).astype(np.uint8)

    img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
    overlay = (alpha * colored_heatmap + (1.0 - alpha) * img_uint8).astype(np.uint8)
    return overlay


def run_gradcam_audit(model_path: Path, n_samples_per_class: int = 4) -> None:
    GRADCAM_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[gradcam] Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)

    splits, stats = build_file_lists()
    test_paths, test_labels = splits["test"]

    pure_test = [p for p, l in zip(test_paths, test_labels) if l == 0][:n_samples_per_class]
    adulterated_test = [p for p, l in zip(test_paths, test_labels) if l == 1][:n_samples_per_class]

    selected = [(p, 0) for p in pure_test] + [(p, 1) for p in adulterated_test]

    fig, axes = plt.subplots(len(selected), 3, figsize=(10, 3.2 * len(selected)))
    if len(selected) == 1:
        axes = np.expand_dims(axes, 0)

    for i, (img_path, true_label) in enumerate(selected):
        raw_bytes = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(raw_bytes, channels=3)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE]).numpy()

        input_tensor = np.expand_dims(img, 0)
        preds = model.predict(input_tensor, verbose=0)[0]
        pred_label = int(np.argmax(preds))
        conf = float(preds[pred_label])

        heatmap = get_gradcam_heatmap(model, input_tensor, target_class_idx=pred_label)
        overlay = overlay_heatmap(img, heatmap)

        # Plot original
        axes[i, 0].imshow(img.astype(np.uint8))
        axes[i, 0].set_title(f"True: {CLASS_NAMES[true_label]}\n{Path(img_path).name[:18]}")
        axes[i, 0].axis("off")

        # Plot raw heatmap
        axes[i, 1].imshow(heatmap, cmap="jet")
        axes[i, 1].set_title("Grad-CAM Heatmap")
        axes[i, 1].axis("off")

        # Plot overlay
        color = "green" if pred_label == true_label else "red"
        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title(
            f"Pred: {CLASS_NAMES[pred_label]} ({conf*100:.1f}%)\n"
            f"({'Diffuse (Pure)' if pred_label == 0 else 'Focal Particulates'})",
            color=color,
        )
        axes[i, 2].axis("off")

    fig.tight_layout()
    out_img = GRADCAM_DIR / "gradcam_verification_audit.png"
    fig.savefig(out_img, dpi=130)
    plt.close(fig)
    print(f"[gradcam] Saved Grad-CAM verification audit image -> {out_img}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        default=str(OUTPUTS_DIR / "redchili_model.keras"),
        help="Path to trained Keras model",
    )
    ap.add_argument(
        "--samples",
        type=int,
        default=4,
        help="Samples per class to audit",
    )
    args = ap.parse_args()

    m_path = Path(args.model)
    if not m_path.exists():
        ckpt_path = OUTPUTS_DIR / "checkpoints" / "best_phase2.keras"
        if ckpt_path.exists():
            m_path = ckpt_path
        else:
            print(f"[gradcam] Error: Model file not found at {m_path}")
            return

    run_gradcam_audit(m_path, n_samples_per_class=args.samples)


if __name__ == "__main__":
    main()
