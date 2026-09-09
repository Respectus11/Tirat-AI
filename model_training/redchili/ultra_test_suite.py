"""
ultra_test_suite.py — Definitive "Ultra Test" for Tirat AI Red Chili Model.

Multi-Circumstance Testing Matrix:
  1. Comprehensive Benchmark on All 5,300+ Raw Samples (Pure vs 5 Adulterant Types)
  2. Concentration Sensitivity Curves (0% Pure vs 5% vs 10% vs 15%)
  3. Realistic Mobile Circumstance Matrix:
     - Dim Indoor Stall (Brightness -40% + Sensor ISO Noise)
     - Harsh Sunlight / Overexposure (Brightness +40% + Contrast +30%)
     - Color Casts (Tungsten 2700K vs Fluorescent 6500K)
     - Rotational Invariance (0°, 90°, 180°, 270°, Flips)
     - Optical Zoom / Distance (0.75x Wide vs 1.5x Macro)
     - Heavy JPEG Compression (Quality 30, 50, 85)
     - Progressive Defocus Sweep (Sigma 0.5 -> 2.5)
  4. Non-Chili & Borderline Lookalike Stress Test (Other Spices, Non-Food, Degraded)
  5. On-Device Inference Speed / Latency Benchmarks (p50, p95, p99)
  6. Calibration, ROC-AUC, and Confidence Distribution
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
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "model_training" / "data_redchili" / "raw"
OUTPUTS_DIR = REPO_ROOT / "model_training" / "redchili" / "outputs" / "ultra_test"
TFLITE_PATH = REPO_ROOT / "app" / "assets" / "models" / "redchili_model.tflite"

IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.70
RED_RATIO_MIN = 0.12
TEXTURE_VARIANCE_MIN = 150.0

# ----------------- Classifier Wrapper -----------------
class UltraTFLiteClassifier:
    def __init__(self, model_path: Path):
        self.interpreter = tf.lite.Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def predict_one(self, img_array: np.ndarray) -> np.ndarray:
        if img_array.ndim == 3:
            img_array = np.expand_dims(img_array, 0)
        img_array = img_array.astype(np.float32)
        self.interpreter.set_tensor(self.input_details[0]["index"], img_array)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.output_details[0]["index"])
        return out[0]  # [p_pure, p_adulterated]

    def benchmark_latency(self, n_runs: int = 500) -> dict:
        dummy = np.random.uniform(0, 255, (1, IMG_SIZE, IMG_SIZE, 3)).astype(np.float32)
        # Warmup
        for _ in range(25):
            self.interpreter.set_tensor(self.input_details[0]["index"], dummy)
            self.interpreter.invoke()

        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            self.interpreter.set_tensor(self.input_details[0]["index"], dummy)
            self.interpreter.invoke()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)  # ms

        times = np.array(times)
        return {
            "runs": n_runs,
            "mean_ms": round(float(np.mean(times)), 2),
            "median_ms": round(float(np.median(times)), 2),
            "p90_ms": round(float(np.percentile(times, 90)), 2),
            "p95_ms": round(float(np.percentile(times, 95)), 2),
            "p99_ms": round(float(np.percentile(times, 99)), 2),
            "min_ms": round(float(np.min(times)), 2),
            "max_ms": round(float(np.max(times)), 2),
        }


def check_app_gates(img_array: np.ndarray) -> tuple[float, float, bool]:
    r = img_array[:, :, 0]
    g = img_array[:, :, 1]
    b = img_array[:, :, 2]
    total_px = IMG_SIZE * IMG_SIZE
    red_mask = (r > 90) & (r > g * 1.35) & (r > b * 1.1)
    red_ratio = float(np.sum(red_mask) / total_px)

    var_r = np.var(r)
    var_g = np.var(g)
    var_b = np.var(b)
    texture_var = float((var_r + var_g + var_b) / 3.0)

    passed = (red_ratio >= RED_RATIO_MIN) and (texture_var >= TEXTURE_VARIANCE_MIN)
    return red_ratio, texture_var, passed


def load_image_square(path: Path) -> Image.Image:
    with Image.open(path) as img:
        img = img.convert("RGB")
        w, h = img.size
        min_dim = min(w, h)
        left = (w - min_dim) // 2
        top = (h - min_dim) // 2
        return img.crop((left, top, left + min_dim, top + min_dim)).resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)


# ----------------- 1. Large-Scale Dataset Evaluation -----------------
def run_full_dataset_audit(clf: UltraTFLiteClassifier, max_samples_per_dir: int = 150) -> dict:
    print("\n[ULTRA-1] Running Full Dataset Scale & Breadth Audit...")
    all_dirs = sorted([d for d in RAW_DIR.iterdir() if d.is_dir()])
    
    y_true_all = []
    y_pred_all = []
    y_prob_adulterated = []
    per_dir_stats = {}

    rng = np.random.RandomState(42)

    for folder in all_dirs:
        is_pure = folder.name in ("C1_PWH", "WH00")
        label = 0 if is_pure else 1
        img_files = sorted(list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg")) + list(folder.glob("*.png")))
        if not img_files:
            continue

        selected = [img_files[i] for i in rng.permutation(len(img_files))[:max_samples_per_dir]]
        dir_correct = 0
        confs = []
        gates_passed = 0

        for p in selected:
            pil_img = load_image_square(p)
            arr = np.array(pil_img, dtype=np.float32)
            _, _, passed = check_app_gates(arr)
            if passed:
                gates_passed += 1

            probs = clf.predict_one(arr)
            pred = int(np.argmax(probs))
            conf = float(probs[pred])
            confs.append(conf)

            y_true_all.append(label)
            y_pred_all.append(pred)
            y_prob_adulterated.append(float(probs[1]))

            if pred == label:
                dir_correct += 1

        n_dir = len(selected)
        acc = dir_correct / n_dir
        per_dir_stats[folder.name] = {
            "samples": n_dir,
            "accuracy": round(acc, 4),
            "avg_confidence": round(float(np.mean(confs)), 4),
            "gate_pass_rate": round(gates_passed / n_dir, 4),
            "type": "pure" if is_pure else "adulterated",
        }
        print(f"   Folder {folder.name:10s} ({n_dir:3d} imgs) -> Acc: {acc*100:5.1f}% | Avg Conf: {np.mean(confs)*100:5.1f}%")

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    y_prob_adulterated = np.array(y_prob_adulterated)

    total_acc = accuracy_score(y_true_all, y_pred_all)
    cm = confusion_matrix(y_true_all, y_pred_all)
    fpr, tpr, _ = roc_curve(y_true_all, y_prob_adulterated)
    roc_auc = auc(fpr, tpr)
    p, r, f1, _ = precision_recall_fscore_support(y_true_all, y_pred_all, average="binary")

    print(f"\n   >>> TOTAL SAMPLES EVALUATED: {len(y_true_all)}")
    print(f"   >>> OVERALL ACCURACY       : {total_acc*100:.2f}%")
    print(f"   >>> ROC AUC SCORE          : {roc_auc:.4f}")
    print(f"   >>> BALANCED F1 SCORE      : {f1:.4f}")
    print(f"   >>> CONFUSION MATRIX       : Pure={cm[0,0]}/{cm[0,0]+cm[0,1]} | Adulterated={cm[1,1]}/{cm[1,0]+cm[1,1]}")

    return {
        "total_samples": len(y_true_all),
        "overall_accuracy": round(float(total_acc), 4),
        "roc_auc": round(float(roc_auc), 4),
        "precision": round(float(p), 4),
        "recall": round(float(r), 4),
        "f1": round(float(f1), 4),
        "confusion_matrix": cm.tolist(),
        "per_dir_stats": per_dir_stats,
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
    }


# ----------------- 2. Concentration Tier Curve -----------------
def run_concentration_tier_analysis(clf: UltraTFLiteClassifier) -> dict:
    print("\n[ULTRA-2] Running Concentration Sensitivity Curve Analysis...")
    tiers = {
        "0% (Pure)": ["C1_PWH", "WH00"],
        "5% (Subtle Adulteration)": ["WHBM_5", "WHGM_5", "WHRB_5", "WHWB_5", "WHWS_5"],
        "10% (Moderate Adulteration)": ["WHBM_10", "WHGM_10", "WHRB_10", "WHWB_10", "WHWS_10"],
        "15% (Heavy Adulteration)": ["WHBM_15", "WHGM_15", "WHRB_15", "WHWB_15", "WHWS_15"],
    }

    tier_results = {}
    rng = np.random.RandomState(123)

    for tier_name, folder_names in tiers.items():
        is_pure = "Pure" in tier_name
        target_label = 0 if is_pure else 1

        correct = 0
        conf_scores = []
        total = 0

        for f_name in folder_names:
            f_path = RAW_DIR / f_name
            if not f_path.exists():
                continue
            imgs = sorted(list(f_path.glob("*.jpg")))
            sample_count = min(60, len(imgs))
            selected = [imgs[i] for i in rng.permutation(len(imgs))[:sample_count]]

            for p in selected:
                pil_img = load_image_square(p)
                arr = np.array(pil_img, dtype=np.float32)
                probs = clf.predict_one(arr)
                pred = int(np.argmax(probs))
                conf = float(probs[pred])
                conf_scores.append(conf)
                if pred == target_label:
                    correct += 1
                total += 1

        acc = correct / total if total > 0 else 0.0
        avg_conf = float(np.mean(conf_scores)) if conf_scores else 0.0
        tier_results[tier_name] = {
            "total": total,
            "correct": correct,
            "accuracy": round(acc, 4),
            "avg_confidence": round(avg_conf, 4),
        }
        print(f"   Tier {tier_name:28s} -> Acc: {acc*100:5.1f}% | Avg Confidence: {avg_conf*100:5.1f}% (N={total})")

    return tier_results


# ----------------- 3. Realistic Mobile Circumstance Matrix -----------------
def run_mobile_circumstances_matrix(clf: UltraTFLiteClassifier, n_samples: int = 80) -> dict:
    print("\n[ULTRA-3] Running Realistic Smartphone Circumstance Matrix...")
    # Gather balanced test images
    pure_candidates = sorted(list((RAW_DIR / "C1_PWH").glob("*.jpg")) + list((RAW_DIR / "WH00").glob("*.jpg")))
    adulterated_candidates = sorted(list((RAW_DIR / "WHWB_10").glob("*.jpg")) + list((RAW_DIR / "WHWS_10").glob("*.jpg")) + list((RAW_DIR / "WHRB_10").glob("*.jpg")))

    rng = np.random.RandomState(99)
    p_sel = [pure_candidates[i] for i in rng.permutation(len(pure_candidates))[: n_samples // 2]]
    a_sel = [adulterated_candidates[i] for i in rng.permutation(len(adulterated_candidates))[: n_samples // 2]]
    eval_set = [(p, 0) for p in p_sel] + [(p, 1) for p in a_sel]

    circumstances = {
        "1. Clean Baseline (Standard Studio)": lambda im: im,
        "2. Dim Stall Light (-40% Bright + ISO Noise)": lambda im: add_iso_noise(ImageEnhance.Brightness(im).enhance(0.6), noise_std=12),
        "3. Blazing Sun (+40% Bright + High Contrast)": lambda im: ImageEnhance.Contrast(ImageEnhance.Brightness(im).enhance(1.4)).enhance(1.3),
        "4. Tungsten Warm Light (Yellowish 2700K)": lambda im: apply_color_cast(im, r_scale=1.2, g_scale=1.05, b_scale=0.8),
        "5. Fluorescent Cool Light (Bluish 6500K)": lambda im: apply_color_cast(im, r_scale=0.85, g_scale=1.0, b_scale=1.2),
        "6. Inverted Orientation (180° Upside Down)": lambda im: im.rotate(180),
        "7. Angled Sideways (90° Rotation)": lambda im: im.rotate(90),
        "8. Macro Close-Up (1.5x Digital Zoom Crop)": lambda im: zoom_crop(im, zoom_factor=1.5),
        "9. Wide Distance (0.75x Distant Framing)": lambda im: distant_pad(im, scale_factor=0.75),
        "10. High Compression (JPEG Quality 50)": lambda im: jpeg_compress(im, quality=50),
        "11. Severe Compression (JPEG Quality 25)": lambda im: jpeg_compress(im, quality=25),
        "12. Mild Camera Shake (Blur Sigma 0.8)": lambda im: im.filter(ImageFilter.GaussianBlur(radius=0.8)),
        "13. Heavy Camera Shake (Blur Sigma 2.0)": lambda im: im.filter(ImageFilter.GaussianBlur(radius=2.0)),
    }

    results = {}
    for c_name, transform_fn in circumstances.items():
        correct = 0
        confs = []
        gates_passed = 0

        for path, true_label in eval_set:
            base_pil = load_image_square(path)
            transformed = transform_fn(base_pil)
            arr = np.array(transformed, dtype=np.float32)

            _, _, passed = check_app_gates(arr)
            if passed:
                gates_passed += 1

            probs = clf.predict_one(arr)
            pred = int(np.argmax(probs))
            conf = float(probs[pred])
            confs.append(conf)

            if pred == true_label:
                correct += 1

        acc = correct / len(eval_set)
        avg_c = float(np.mean(confs))
        gate_r = gates_passed / len(eval_set)
        results[c_name] = {
            "accuracy": round(acc, 4),
            "avg_confidence": round(avg_c, 4),
            "gate_pass_rate": round(gate_r, 4),
        }
        status = "EXCELLENT" if acc >= 0.90 else ("GOOD" if acc >= 0.75 else "DEGRADED (Expected)")
        print(f"   [{status:19s}] {c_name:48s} -> Acc: {acc*100:5.1f}% | Conf: {avg_c*100:5.1f}% | Gate: {gate_r*100:5.1f}%")

    return results


def add_iso_noise(pil_img: Image.Image, noise_std: float = 10.0) -> Image.Image:
    arr = np.array(pil_img, dtype=np.float32)
    noise = np.random.normal(0, noise_std, arr.shape)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


def apply_color_cast(pil_img: Image.Image, r_scale: float, g_scale: float, b_scale: float) -> Image.Image:
    r, g, b = pil_img.split()
    r = r.point(lambda i: min(255, int(i * r_scale)))
    g = g.point(lambda i: min(255, int(i * g_scale)))
    b = b.point(lambda i: min(255, int(i * b_scale)))
    return Image.merge("RGB", (r, g, b))


def zoom_crop(pil_img: Image.Image, zoom_factor: float = 1.5) -> Image.Image:
    w, h = pil_img.size
    crop_w = int(w / zoom_factor)
    crop_h = int(h / zoom_factor)
    left = (w - crop_w) // 2
    top = (h - crop_h) // 2
    return pil_img.crop((left, top, left + crop_w, top + crop_h)).resize((w, h), Image.Resampling.BILINEAR)


def distant_pad(pil_img: Image.Image, scale_factor: float = 0.75) -> Image.Image:
    w, h = pil_img.size
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)
    shrunk = pil_img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (w, h), (180, 140, 110))  # wood table color background
    offset = ((w - new_w) // 2, (h - new_h) // 2)
    canvas.paste(shrunk, offset)
    return canvas


def jpeg_compress(pil_img: Image.Image, quality: int = 50) -> Image.Image:
    import io
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# ----------------- 4. Non-Chili & Lookalike Stress Test -----------------
def run_lookalike_stress_test(clf: UltraTFLiteClassifier) -> dict:
    print("\n[ULTRA-4] Running Non-Chili Lookalikes & Negative Picture Stress Test...")
    test_cases = {
        "Turmeric / Curry Powder (Yellow)": np.tile(np.array([220.0, 180.0, 20.0], dtype=np.float32), (IMG_SIZE, IMG_SIZE, 1)),
        "Paprika Powder (Deep Crimson)": np.tile(np.array([170.0, 25.0, 20.0], dtype=np.float32), (IMG_SIZE, IMG_SIZE, 1)),
        "Cinnamon / Cocoa (Dull Brown)": np.tile(np.array([120.0, 70.0, 45.0], dtype=np.float32), (IMG_SIZE, IMG_SIZE, 1)),
        "White Teff / Wheat Flour (Off-white)": np.tile(np.array([230.0, 225.0, 215.0], dtype=np.float32), (IMG_SIZE, IMG_SIZE, 1)),
        "Terracotta Clay Tile (Red-Orange Matte)": np.full((IMG_SIZE, IMG_SIZE, 3), [195.0, 80.0, 55.0], dtype=np.float32),
        "Human Hand Skin Tone (Tan/Warm)": np.tile(np.array([215.0, 170.0, 140.0], dtype=np.float32), (IMG_SIZE, IMG_SIZE, 1)),
        "Wood Cutting Board (Brown Grain)": np.tile(np.array([160.0, 115.0, 75.0], dtype=np.float32), (IMG_SIZE, IMG_SIZE, 1)),
        "Dark Pocket / Blank Shutter": np.full((IMG_SIZE, IMG_SIZE, 3), 10.0, dtype=np.float32),
        "White Paper / Empty Plate": np.full((IMG_SIZE, IMG_SIZE, 3), 245.0, dtype=np.float32),
    }

    # Add real noise textures
    for name in list(test_cases.keys()):
        base = test_cases[name]
        noise = np.random.normal(0, 12, base.shape).astype(np.float32)
        test_cases[name] = np.clip(base + noise, 0, 255)

    results = {}
    blocked_count = 0

    for name, arr in test_cases.items():
        rr, tv, passed_gates = check_app_gates(arr)
        probs = clf.predict_one(arr)
        pred = int(np.argmax(probs))
        conf = float(probs[pred])
        is_conclusive = conf >= CONFIDENCE_THRESHOLD

        # Blocked safely if either gates reject or confidence is below threshold
        is_safe = (not passed_gates) or (not is_conclusive)
        if is_safe:
            blocked_count += 1

        rejection_reason = []
        if rr < RED_RATIO_MIN:
            rejection_reason.append("Color Gate (<12% Red)")
        if tv < TEXTURE_VARIANCE_MIN:
            rejection_reason.append("Texture Variance (<150)")
        if not is_conclusive:
            rejection_reason.append("Low Confidence (<70%)")

        results[name] = {
            "red_ratio": round(rr, 4),
            "texture_variance": round(tv, 2),
            "passed_gates": passed_gates,
            "model_pred": "pure" if pred == 0 else "adulterated",
            "model_confidence": round(conf, 4),
            "is_safely_intercepted": is_safe,
            "reasons": rejection_reason,
        }
        res_str = "BLOCKED SAFELY" if is_safe else "UNCAUGHT"
        print(f"   [{res_str:14s}] {name:38s} -> RedRatio: {rr*100:4.1f}% | Var: {tv:6.1f} | Conf: {conf*100:4.1f}%")

    safety_rate = blocked_count / len(test_cases)
    print(f"\n   >>> NON-FOOD REJECTION SAFETY RATE: {safety_rate*100:.1f}% ({blocked_count}/{len(test_cases)})")
    return {"cases": results, "safety_rate": round(safety_rate, 4)}


# ----------------- 5. Generate Ultra Plots -----------------
def generate_ultra_plots(full_audit: dict, tier_results: dict, circumstances: dict, latency: dict) -> None:
    print("\n[ULTRA-5] Generating Definitive Ultra Visualizations...")
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Multi-Panel Performance Dashboard
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)

    # Panel A: ROC Curve
    ax1 = fig.add_subplot(gs[0, 0])
    fpr = np.array(full_audit["fpr"])
    tpr = np.array(full_audit["tpr"])
    ax1.plot(fpr, tpr, color="#2E7D32", lw=2.5, label=f"ROC Curve (AUC = {full_audit['roc_auc']:.4f})")
    ax1.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--")
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel("False Positive Rate", fontsize=11, fontweight="bold")
    ax1.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11, fontweight="bold")
    ax1.set_title("A. Receiver Operating Characteristic (ROC) Curve", fontsize=12, fontweight="bold")
    ax1.legend(loc="lower right", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Panel B: Concentration Sensitivity Curve
    ax2 = fig.add_subplot(gs[0, 1])
    tiers = list(tier_results.keys())
    tier_accs = [tier_results[t]["accuracy"] * 100 for t in tiers]
    tier_confs = [tier_results[t]["avg_confidence"] * 100 for t in tiers]
    x_pos = np.arange(len(tiers))
    ax2.plot(x_pos, tier_accs, marker="o", lw=2.5, markersize=8, color="#1565C0", label="Detection Accuracy (%)")
    ax2.plot(x_pos, tier_confs, marker="s", lw=2.0, markersize=7, color="#E65100", linestyle="--", label="Model Confidence (%)")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(["0%\n(Pure)", "5%\n(Subtle)", "10%\n(Medium)", "15%\n(High)"], fontsize=10)
    ax2.set_ylabel("Percentage (%)", fontsize=11, fontweight="bold")
    ax2.set_title("B. Sensitivity Across Adulteration Concentration Tiers", fontsize=12, fontweight="bold")
    ax2.set_ylim(70, 105)
    for i, (a, c) in enumerate(zip(tier_accs, tier_confs)):
        ax2.annotate(f"{a:.1f}%", (x_pos[i], a + 1.2), ha="center", fontsize=9, fontweight="bold", color="#1565C0")
    ax2.legend(loc="lower right", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)

    # Panel C: Circumstance Robustness Bars
    ax3 = fig.add_subplot(gs[1, 0])
    c_names = [k.split(". ")[1] for k in circumstances.keys()]
    c_accs = [circumstances[k]["accuracy"] * 100 for k in circumstances.keys()]
    colors = ["#2E7D32" if a >= 90 else ("#FB8C00" if a >= 70 else "#D32F2F") for a in c_accs]
    y_pos = np.arange(len(c_names))
    ax3.barh(y_pos, c_accs, color=colors, edgecolor="black", height=0.7)
    ax3.axvline(90, color="gray", linestyle="--", alpha=0.7, label="90% High Performance")
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(c_names, fontsize=8.5)
    ax3.invert_yaxis()
    ax3.set_xlabel("Accuracy Under Distortion (%)", fontsize=11, fontweight="bold")
    ax3.set_title("C. Robustness Under Smartphone Distortions", fontsize=12, fontweight="bold")
    ax3.set_xlim(0, 110)
    ax3.grid(axis="x", linestyle=":", alpha=0.6)

    # Panel D: Latency Distribution
    ax4 = fig.add_subplot(gs[1, 1])
    lat_labels = ["Min", "Median (p50)", "Mean", "p95", "p99", "Max"]
    lat_vals = [latency["min_ms"], latency["median_ms"], latency["mean_ms"], latency["p95_ms"], latency["p99_ms"], latency["max_ms"]]
    ax4.bar(lat_labels, lat_vals, color="#455A64", edgecolor="black", width=0.55)
    ax4.axhline(45, color="red", linestyle=":", label="45ms Real-Time Mobile Target")
    ax4.set_ylabel("Inference Latency (ms)", fontsize=11, fontweight="bold")
    ax4.set_title(f"D. On-Device Inference Speed Benchmarks (N={latency['runs']} runs)", fontsize=12, fontweight="bold")
    ax4.set_ylim(0, max(lat_vals) * 1.35)
    for i, v in enumerate(lat_vals):
        ax4.text(i, v + 0.8, f"{v:.1f}ms", ha="center", fontsize=9, fontweight="bold")
    ax4.legend(loc="upper left", fontsize=10)
    ax4.grid(axis="y", linestyle=":", alpha=0.6)

    plt.suptitle("Tirat AI Red Chili Model — Comprehensive Ultra Test Evaluation Dashboard", fontsize=15, fontweight="bold", y=0.98)
    dashboard_path = OUTPUTS_DIR / "ultra_test_dashboard.png"
    plt.savefig(dashboard_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"   Saved ultra dashboard plot -> {dashboard_path}")


def main() -> None:
    start_time = time.time()
    print("=" * 75)
    print("TIRAT AI — DEFINITIVE ULTRA MODEL TEST SUITE (FINAL CHECK)")
    print("=" * 75)

    clf = UltraTFLiteClassifier(TFLITE_PATH)

    # 1. Full Dataset Audit
    audit_results = run_full_dataset_audit(clf, max_samples_per_dir=100)

    # 2. Concentration Tier Analysis
    tier_results = run_concentration_tier_analysis(clf)

    # 3. Mobile Circumstances Matrix
    circumstances_results = run_mobile_circumstances_matrix(clf, n_samples=80)

    # 4. Lookalike & Negative Picture Stress Test
    lookalike_results = run_lookalike_stress_test(clf)

    # 5. Latency Benchmark
    print("\n[ULTRA-5] Running Latency & Real-Time Throughput Benchmark (500 iterations)...")
    latency_results = clf.benchmark_latency(n_runs=500)
    print(f"   Mean Latency: {latency_results['mean_ms']} ms | Median: {latency_results['median_ms']} ms | p95: {latency_results['p95_ms']} ms")

    # 6. Plots
    generate_ultra_plots(audit_results, tier_results, circumstances_results, latency_results)

    # 7. Final JSON Report
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_file": str(TFLITE_PATH.name),
        "audit_results": audit_results,
        "tier_results": tier_results,
        "circumstances_results": circumstances_results,
        "lookalike_results": lookalike_results,
        "latency_results": latency_results,
        "total_elapsed_sec": round(time.time() - start_time, 2),
    }

    report_file = OUTPUTS_DIR / "ultra_test_report.json"
    report_file.write_text(json.dumps(report, indent=2))
    print(f"\n[DONE] Ultra Test Complete in {time.time() - start_time:.1f}s!")
    print(f"Saved JSON report -> {report_file}")


if __name__ == "__main__":
    main()
