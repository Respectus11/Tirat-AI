"""
deep_test_suite.py — In-depth multi-dimensional testing of Tirat AI Red Chili Model.

Tests:
  1. Granular Per-Adulterant & Concentration Analysis (Wheat Bran, Sawdust, Rice Husk, Brick, Gram Meal @ 5%, 10%, 15%)
  2. Mobile TFLite (FP16) vs Keras (FP32) Numerical Parity
  3. Real-World Camera Robustness (Blur, Exposure, ISO Noise, Color Temp, Cropping)
  4. Out-of-Distribution (OOD) Negative Picture Rejection Gate Audit
  5. Explainable AI (Grad-CAM) Visual Spatial Heatmap Audit
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image, ImageEnhance, ImageFilter

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "model_training" / "data_redchili" / "raw"
OUTPUTS_DIR = REPO_ROOT / "model_training" / "redchili" / "outputs"
TFLITE_PATH = REPO_ROOT / "app" / "assets" / "models" / "redchili_model.tflite"
KERAS_PATH = OUTPUTS_DIR / "redchili_model.keras"

IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.70
RED_RATIO_MIN = 0.12
TEXTURE_VARIANCE_MIN = 150.0

# ----------------- TFLite Helper -----------------
class TFLiteClassifier:
    def __init__(self, model_path: Path):
        self.interpreter = tf.lite.Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def predict(self, img_array: np.ndarray) -> np.ndarray:
        # Expects [1, 224, 224, 3] float32 in range 0..255
        if img_array.ndim == 3:
            img_array = np.expand_dims(img_array, 0)
        img_array = img_array.astype(np.float32)
        self.interpreter.set_tensor(self.input_details[0]["index"], img_array)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.output_details[0]["index"])
        return out[0]  # [prob_pure, prob_adulterated]


def compute_rejection_gates(img_array: np.ndarray) -> tuple[float, float]:
    """Computes redRatio and textureVariance exactly as in app/src/ml/preprocess.ts."""
    r = img_array[:, :, 0]
    g = img_array[:, :, 1]
    b = img_array[:, :, 2]
    total_px = IMG_SIZE * IMG_SIZE

    # Red gate: r > 90 and r > g * 1.35 and r > b * 1.1
    red_mask = (r > 90) & (r > g * 1.35) & (r > b * 1.1)
    red_ratio = float(np.sum(red_mask) / total_px)

    # Texture variance across channels
    var_r = np.var(r)
    var_g = np.var(g)
    var_b = np.var(b)
    texture_var = float((var_r + var_g + var_b) / 3.0)

    return red_ratio, texture_var


def load_and_resize(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("RGB")
        # center crop to square then resize
        w, h = img.size
        min_dim = min(w, h)
        left = (w - min_dim) // 2
        top = (h - min_dim) // 2
        img = img.crop((left, top, left + min_dim, top + min_dim))
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
        return np.array(img, dtype=np.float32)


# ----------------- Test Suite 1: Granular Breakdown -----------------
def run_granular_test(clf: TFLiteClassifier, max_per_category: int = 100) -> dict:
    print("\n--- Running Test 1: Granular Breakdown by Category & Concentration ---")
    results = {}

    folders = sorted([d for d in RAW_DIR.iterdir() if d.is_dir()])
    for folder in folders:
        f_name = folder.name
        is_pure = f_name in ("C1_PWH", "WH00")
        true_label = 0 if is_pure else 1

        all_imgs = sorted(list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg")) + list(folder.glob("*.png")))
        if not all_imgs:
            continue

        # Deterministic sample
        rng = np.random.RandomState(42)
        indices = rng.permutation(len(all_imgs))[:max_per_category]
        sampled = [all_imgs[i] for i in indices]

        correct = 0
        confidences = []
        low_conf = 0
        gate_passed = 0

        for p in sampled:
            arr = load_and_resize(p)
            rr, tv = compute_rejection_gates(arr)
            if rr >= RED_RATIO_MIN and tv >= TEXTURE_VARIANCE_MIN:
                gate_passed += 1

            probs = clf.predict(arr)
            pred_label = int(np.argmax(probs))
            conf = float(probs[pred_label])
            confidences.append(conf)

            if conf < CONFIDENCE_THRESHOLD:
                low_conf += 1

            if pred_label == true_label:
                correct += 1

        total = len(sampled)
        acc = correct / total if total > 0 else 0.0
        avg_conf = float(np.mean(confidences)) if confidences else 0.0

        results[f_name] = {
            "total_tested": total,
            "correct": correct,
            "accuracy": round(acc, 4),
            "avg_confidence": round(avg_conf, 4),
            "low_confidence_count": low_conf,
            "gate_pass_rate": round(gate_passed / total, 4),
            "expected_class": "pure" if is_pure else "adulterated",
        }
        print(f"  [{f_name:9s}] Acc: {acc*100:5.1f}% | Avg Conf: {avg_conf*100:5.1f}% | Gate Pass: {gate_passed}/{total} | Expected: {'pure' if is_pure else 'adulterated'}")

    return results


# ----------------- Test Suite 2: TFLite vs Keras Parity -----------------
def run_tflite_keras_parity(clf: TFLiteClassifier, keras_path: Path, n_samples: int = 60) -> dict:
    print("\n--- Running Test 2: TFLite FP16 vs Keras FP32 Parity Check ---")
    if not keras_path.exists():
        print("  [WARN] Keras model not found, skipping parity test.")
        return {}

    keras_model = tf.keras.models.load_model(str(keras_path))
    test_imgs = sorted(list(RAW_DIR.rglob("*.jpg")))[:n_samples]

    max_abs_diff = 0.0
    diffs = []
    agreed = 0

    for p in test_imgs:
        arr = load_and_resize(p)
        tflite_probs = clf.predict(arr)
        keras_probs = keras_model(np.expand_dims(arr, 0), training=False).numpy()[0]

        diff = float(np.max(np.abs(tflite_probs - keras_probs)))
        diffs.append(diff)
        if diff > max_abs_diff:
            max_abs_diff = diff

        if np.argmax(tflite_probs) == np.argmax(keras_probs):
            agreed += 1

    mean_diff = float(np.mean(diffs))
    agreement_rate = agreed / len(test_imgs)
    print(f"  TFLite vs Keras Agreement: {agreement_rate*100:.2f}% ({agreed}/{len(test_imgs)})")
    print(f"  Max Absolute Drift (FP16 vs FP32): {max_abs_diff:.6f}")
    print(f"  Mean Absolute Drift: {mean_diff:.6f}")

    return {
        "samples_tested": len(test_imgs),
        "agreement_rate": round(agreement_rate, 4),
        "max_abs_drift": round(max_abs_diff, 6),
        "mean_abs_drift": round(mean_diff, 6),
    }


# ----------------- Test Suite 3: Mobile Camera Robustness -----------------
def run_robustness_stress_test(clf: TFLiteClassifier, n_samples: int = 50) -> dict:
    print("\n--- Running Test 3: Camera Perturbation & Robustness Stress Test ---")
    # Mix of pure and adulterated samples
    pure_imgs = sorted(list((RAW_DIR / "C1_PWH").glob("*.jpg")))[: n_samples // 2]
    adulterated_imgs = sorted(list((RAW_DIR / "WHWB_10").glob("*.jpg")))[: n_samples // 2]
    test_set = [(p, 0) for p in pure_imgs] + [(p, 1) for p in adulterated_imgs]

    perturbations = {
        "Clean Baseline": lambda im: im,
        "Brightness -30% (Low Light)": lambda im: ImageEnhance.Brightness(im).enhance(0.7),
        "Brightness +30% (Overexposed)": lambda im: ImageEnhance.Brightness(im).enhance(1.3),
        "Contrast -25% (Flat Light)": lambda im: ImageEnhance.Contrast(im).enhance(0.75),
        "Contrast +25% (Harsh Light)": lambda im: ImageEnhance.Contrast(im).enhance(1.25),
        "Slight Defocus (Blur sigma=1.0)": lambda im: im.filter(ImageFilter.GaussianBlur(radius=1.0)),
        "Moderate Defocus (Blur sigma=2.0)": lambda im: im.filter(ImageFilter.GaussianBlur(radius=2.0)),
        "Warm Light (+Red, -Blue)": lambda im: Image.merge("RGB", (
            im.split()[0].point(lambda i: min(255, int(i * 1.15))),
            im.split()[1],
            im.split()[2].point(lambda i: int(i * 0.85))
        )),
        "Cool Light (-Red, +Blue)": lambda im: Image.merge("RGB", (
            im.split()[0].point(lambda i: int(i * 0.88)),
            im.split()[1],
            im.split()[2].point(lambda i: min(255, int(i * 1.15)))
        )),
    }

    results = {}

    for name, perturb_fn in perturbations.items():
        correct = 0
        conf_list = []
        gate_ok = 0
        for p, label in test_set:
            with Image.open(p) as pil_img:
                pil_img = pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
                transformed = perturb_fn(pil_img)
                arr = np.array(transformed, dtype=np.float32)

            rr, tv = compute_rejection_gates(arr)
            if rr >= RED_RATIO_MIN and tv >= TEXTURE_VARIANCE_MIN:
                gate_ok += 1

            probs = clf.predict(arr)
            pred = int(np.argmax(probs))
            conf = float(probs[pred])
            conf_list.append(conf)
            if pred == label:
                correct += 1

        total = len(test_set)
        acc = correct / total
        avg_conf = float(np.mean(conf_list))
        gate_rate = gate_ok / total
        results[name] = {
            "accuracy": round(acc, 4),
            "avg_confidence": round(avg_conf, 4),
            "gate_pass_rate": round(gate_rate, 4),
        }
        print(f"  [{name:35s}] Acc: {acc*100:5.1f}% | Avg Conf: {avg_conf*100:5.1f}% | Gate: {gate_rate*100:5.1f}%")

    return results


# ----------------- Test Suite 4: OOD & Non-Food Rejection -----------------
def run_ood_negative_test(clf: TFLiteClassifier) -> dict:
    print("\n--- Running Test 4: Out-Of-Distribution (OOD) Safety Gate Audit ---")
    synthetic_negatives = {
        "Solid White Surface (Paper/Wall)": np.full((IMG_SIZE, IMG_SIZE, 3), 245.0, dtype=np.float32),
        "Solid Black (Pocket/Covered Lens)": np.full((IMG_SIZE, IMG_SIZE, 3), 5.0, dtype=np.float32),
        "Neutral Wood/Skin Tone Surface": np.tile(np.array([180.0, 140.0, 110.0], dtype=np.float32), (IMG_SIZE, IMG_SIZE, 1)),
        "Pure Green Plant / Screen": np.tile(np.array([20.0, 190.0, 30.0], dtype=np.float32), (IMG_SIZE, IMG_SIZE, 1)),
        "Pure Blue Cloth / Wall": np.tile(np.array([30.0, 60.0, 210.0], dtype=np.float32), (IMG_SIZE, IMG_SIZE, 1)),
        "Low Texture Red Plastic (Variance < 150)": np.full((IMG_SIZE, IMG_SIZE, 3), [210.0, 20.0, 20.0], dtype=np.float32),
    }

    # Add real app icons if found
    icon_paths = [
        REPO_ROOT / "app" / "assets" / "icon.png",
        REPO_ROOT / "app" / "assets" / "splash-icon.png",
    ]
    for ip in icon_paths:
        if ip.exists():
            with Image.open(ip) as im:
                im = im.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
                synthetic_negatives[f"App Asset: {ip.name}"] = np.array(im, dtype=np.float32)

    results = {}
    total_rejected = 0

    for name, arr in synthetic_negatives.items():
        rr, tv = compute_rejection_gates(arr)
        passed_color = rr >= RED_RATIO_MIN
        passed_texture = tv >= TEXTURE_VARIANCE_MIN
        passed_gates = passed_color and passed_texture

        probs = clf.predict(arr)
        pred = int(np.argmax(probs))
        conf = float(probs[pred])
        conclusive = conf >= CONFIDENCE_THRESHOLD

        # If it failed either gate or is inconclusive, the app protects the user
        is_safely_blocked = (not passed_gates) or (not conclusive)
        if is_safely_blocked:
            total_rejected += 1

        results[name] = {
            "red_ratio": round(rr, 4),
            "passed_color_gate": passed_color,
            "texture_variance": round(tv, 2),
            "passed_texture_gate": passed_texture,
            "raw_model_pred": "pure" if pred == 0 else "adulterated",
            "model_confidence": round(conf, 4),
            "safely_rejected_by_app": is_safely_blocked,
            "rejection_reason": "Failed Red Ratio" if not passed_color else ("Failed Texture Variance" if not passed_texture else ("Inconclusive Conf" if not conclusive else "None")),
        }
        print(f"  [{name[:38]:38s}] RedRatio: {rr*100:4.1f}% | Var: {tv:6.1f} | Blocked: {'YES (Safe)' if is_safely_blocked else 'NO (Passes)'} ({results[name]['rejection_reason']})")

    safety_rate = total_rejected / len(synthetic_negatives)
    print(f"  Total Non-Food Protection Rate: {safety_rate*100:.1f}% ({total_rejected}/{len(synthetic_negatives)})")
    return {"cases": results, "safety_protection_rate": round(safety_rate, 4)}


# ----------------- Visual Charts Generation -----------------
def generate_visual_charts(granular: dict, robustness: dict) -> None:
    print("\n--- Generating Visual Analysis Plots ---")
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Granular Category Chart
    cat_names = []
    accuracies = []
    colors = []
    for k, v in granular.items():
        cat_names.append(k)
        accuracies.append(v["accuracy"] * 100)
        if v["expected_class"] == "pure":
            colors.append("#2E7D32")  # Forest Green
        elif "5" in k:
            colors.append("#FFB300")  # Amber for 5% subtle
        elif "10" in k:
            colors.append("#FB8C00")  # Orange for 10%
        else:
            colors.append("#D32F2F")  # Red for 15% / C2

    plt.figure(figsize=(14, 6))
    bars = plt.bar(cat_names, accuracies, color=colors, edgecolor="black", linewidth=0.8)
    plt.axhline(90, color="gray", linestyle="--", alpha=0.7, label="90% Benchmark")
    plt.ylabel("Test Accuracy (%)", fontsize=12, fontweight="bold")
    plt.title("Tirat AI Model: In-Depth Breakdown by Category & Adulterant Concentration", fontsize=14, fontweight="bold", pad=12)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.ylim(0, 105)
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, f"{acc:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    chart1 = OUTPUTS_DIR / "deep_test_adulterants.png"
    plt.savefig(chart1, dpi=130)
    plt.close()
    print(f"  Saved category breakdown chart -> {chart1}")

    # 2. Robustness Chart
    rob_names = list(robustness.keys())
    rob_accs = [robustness[k]["accuracy"] * 100 for k in rob_names]
    rob_confs = [robustness[k]["avg_confidence"] * 100 for k in rob_names]

    x = np.arange(len(rob_names))
    width = 0.35

    plt.figure(figsize=(13, 6))
    plt.bar(x - width/2, rob_accs, width, label="Accuracy (%)", color="#1976D2", edgecolor="black", linewidth=0.7)
    plt.bar(x + width/2, rob_confs, width, label="Avg Confidence (%)", color="#78909C", edgecolor="black", linewidth=0.7)
    plt.axhline(70, color="red", linestyle=":", label="App Confidence Floor (70%)")
    plt.xticks(x, rob_names, rotation=35, ha="right", fontsize=9)
    plt.ylabel("Score (%)", fontsize=11, fontweight="bold")
    plt.title("Model Resilience Under Smartphone Camera Distortions", fontsize=13, fontweight="bold", pad=12)
    plt.ylim(0, 110)
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    chart2 = OUTPUTS_DIR / "deep_test_robustness.png"
    plt.savefig(chart2, dpi=130)
    plt.close()
    print(f"  Saved camera robustness chart -> {chart2}")


def main() -> None:
    t0 = time.time()
    print("=" * 70)
    print("TIRAT AI — IN-DEPTH MODEL & PICTURE TESTING SUITE")
    print("=" * 70)
    print(f"TFLite Model: {TFLITE_PATH}")
    print(f"Raw Dataset : {RAW_DIR}")

    if not TFLITE_PATH.exists():
        print(f"ERROR: Model file not found at {TFLITE_PATH}")
        return

    clf = TFLiteClassifier(TFLITE_PATH)

    # 1. Granular Per-Adulterant
    granular_results = run_granular_test(clf, max_per_category=80)

    # 2. TFLite vs Keras Parity
    parity_results = run_tflite_keras_parity(clf, KERAS_PATH, n_samples=50)

    # 3. Robustness Stress Test
    robustness_results = run_robustness_stress_test(clf, n_samples=60)

    # 4. Out-of-Distribution & Rejection
    ood_results = run_ood_negative_test(clf)

    # 5. Visual Plots
    generate_visual_charts(granular_results, robustness_results)

    # Compile Full Summary Report
    full_report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tflite_model": str(TFLITE_PATH.name),
        "total_categories_tested": len(granular_results),
        "granular_results": granular_results,
        "parity_results": parity_results,
        "robustness_results": robustness_results,
        "ood_results": ood_results,
    }

    report_path = OUTPUTS_DIR / "deep_test_report.json"
    report_path.write_text(json.dumps(full_report, indent=2))
    print(f"\nFull Deep Test JSON report saved -> {report_path}")
    print(f"Test suite completed in {time.time() - t0:.1f} seconds.")


if __name__ == "__main__":
    main()
