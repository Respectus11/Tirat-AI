"""
ethiopian_market_test.py — African & Ethiopian Market Conditions & Multi-Perspective Test Suite.

Simulates authentic Ethiopian food market realities (Merkato / Shola open-air markets, regional mills):
  1. African / Ethiopian Adulterant Profiles:
     - Wood sawdust ("ye-inchet fafuka")
     - Brick powder ("shekla duket" / "ye-deha debdeb")
     - Wheat bran ("ye-sinde gulo")
     - Rice husk / milling chaff
  2. Ethiopian Kitchen & Market Cross-Food Lookalikes:
     - Injera sourdough texture ("ayen")
     - Teff flour (white & brown teff)
     - Teff grain & harvested panicles
     - Ethiopian Turmeric ("Ird")
     - Highland Red Clay Soil ("Ye-key afer")
     - Roasted Barley ("Beso")
     - Traditional Woven Grain Basket ("Mesob" / "Sefed") background
     - Woven Polypropylene Grain Sack ("Madaberia") background
  3. Multi-Perspective Camera Geometries:
     - 15° Casual Oblique Angle
     - 30° Market Counter Slanted Perspective
     - 45° Steep In-Sack Perspective
     - Cast Smartphone Shadow (Harsh high-elevation solar contrast)
     - Hand / Finger Container Occlusion
     - High Altitude UV Solar Glare (Addis Ababa 2,400m midday sun)
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "model_training" / "data_redchili" / "raw"
APP_IMAGES_DIR = REPO_ROOT / "app" / "assets" / "images"
OUTPUTS_DIR = REPO_ROOT / "model_training" / "redchili" / "outputs" / "ethiopian_test"
TFLITE_PATH = REPO_ROOT / "app" / "assets" / "models" / "redchili_model.tflite"

IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.70
RED_RATIO_MIN = 0.12
TEXTURE_VARIANCE_MIN = 150.0

class TFLiteClassifier:
    def __init__(self, model_path: Path):
        self.interpreter = tf.lite.Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def predict(self, img_array: np.ndarray) -> np.ndarray:
        if img_array.ndim == 3:
            img_array = np.expand_dims(img_array, 0)
        img_array = img_array.astype(np.float32)
        self.interpreter.set_tensor(self.input_details[0]["index"], img_array)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.output_details[0]["index"])
        return out[0]  # [prob_pure, prob_adulterated]

def compute_gates(img_array: np.ndarray) -> tuple[float, float, bool]:
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

def load_and_center_crop(path: Path) -> Image.Image:
    with Image.open(path) as img:
        img = img.convert("RGB")
        w, h = img.size
        min_dim = min(w, h)
        left = (w - min_dim) // 2
        top = (h - min_dim) // 2
        return img.crop((left, top, left + min_dim, top + min_dim)).resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)

# ----------------- Perspective & Transform Helpers -----------------
def apply_perspective_tilt(pil_img: Image.Image, angle_deg: float) -> Image.Image:
    """Simulates oblique camera angles by applying trapezoidal perspective distortion."""
    w, h = pil_img.size
    rad = math.radians(angle_deg)
    # Shrink top width proportionally to angle
    inset = int(w * math.sin(rad) * 0.4)
    # Source points (square)
    src = [(0, 0), (w, 0), (w, h), (0, h)]
    # Target points (trapezoid)
    dst = [(inset, 0), (w - inset, 0), (w, h), (0, h)]

    # Compute projective transform matrix coefficients
    def find_coeffs(pa, pb):
        matrix = []
        for p1, p2 in zip(pa, pb):
            matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
            matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])
        A = np.matrix(matrix, dtype=float)
        B = np.array(pb).reshape(8)
        res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
        return np.array(res).reshape(8)

    coeffs = find_coeffs(dst, src)
    return pil_img.transform((w, h), Image.Transform.PERSPECTIVE, coeffs, Image.Resampling.BILINEAR)

def add_cast_phone_shadow(pil_img: Image.Image) -> Image.Image:
    """Simulates a user's smartphone shadow cast over the sample in harsh outdoor sunlight."""
    arr = np.array(pil_img, dtype=np.float32)
    h, w, _ = arr.shape
    # Shadow mask: diagonal band
    y, x = np.ogrid[:h, :w]
    mask = (x + y < (w + h) * 0.55).astype(np.float32)
    # Soften edge
    mask = np.clip(mask * 0.45 + 0.55, 0.45, 1.0)
    shadowed = arr * mask[:, :, np.newaxis]
    return Image.fromarray(np.clip(shadowed, 0, 255).astype(np.uint8))

def add_container_boundary(pil_img: Image.Image, container_type: str = "woven_basket") -> Image.Image:
    """Places the powder inside an authentic container or market sack with borders visible."""
    w, h = pil_img.size
    canvas = Image.new("RGB", (w, h))
    draw = ImageDraw.Draw(canvas)

    if container_type == "woven_basket":
        # Tan wicker pattern
        for y in range(0, h, 8):
            color = (195 + (y % 16), 150 + (y % 12), 105)
            draw.line([(0, y), (w, y)], fill=color, width=4)
    elif container_type == "plastic_sack":
        # White woven polypropylene with blue grain stripes (Madaberia)
        canvas.paste((240, 240, 245), (0, 0, w, h))
        for x in range(20, w, 40):
            draw.line([(x, 0), (x, h)], fill=(30, 90, 180), width=6)
    elif container_type == "red_clay_pot":
        # Terracotta earthenware dish
        canvas.paste((180, 85, 55), (0, 0, w, h))

    # Paste central powder circle
    mask = Image.new("L", (w, h), 0)
    mask_draw = ImageDraw.Draw(mask)
    margin = 25
    mask_draw.ellipse([margin, margin, w - margin, h - margin], fill=255)
    canvas.paste(pil_img, (0, 0), mask)
    return canvas

# ----------------- Test Suite 1: Ethiopian Adulterant Profiles -----------------
def run_ethiopian_adulterants_test(clf: TFLiteClassifier) -> dict:
    print("\n--- [TEST 1] African / Ethiopian Market Adulterant Matrix ---")
    profiles = {
        "Pure Ethiopian Wonder Hot (C1_PWH)": (RAW_DIR / "C1_PWH", "pure", 0),
        "Wood Sawdust — Ye-inchet fafuka 5% (WHWS_5)": (RAW_DIR / "WHWS_5", "adulterated", 1),
        "Wood Sawdust — Ye-inchet fafuka 10% (WHWS_10)": (RAW_DIR / "WHWS_10", "adulterated", 1),
        "Wood Sawdust — Ye-inchet fafuka 15% (WHWS_15)": (RAW_DIR / "WHWS_15", "adulterated", 1),
        "Brick Powder — Shekla duket 5% (WHBM_5)": (RAW_DIR / "WHBM_5", "adulterated", 1),
        "Brick Powder — Shekla duket 10% (WHBM_10)": (RAW_DIR / "WHBM_10", "adulterated", 1),
        "Brick Powder — Shekla duket 15% (WHBM_15)": (RAW_DIR / "WHBM_15", "adulterated", 1),
        "Wheat Bran Filler — Ye-sinde gulo 5% (WHWB_5)": (RAW_DIR / "WHWB_5", "adulterated", 1),
        "Wheat Bran Filler — Ye-sinde gulo 10% (WHWB_10)": (RAW_DIR / "WHWB_10", "adulterated", 1),
        "Wheat Bran Filler — Ye-sinde gulo 15% (WHWB_15)": (RAW_DIR / "WHWB_15", "adulterated", 1),
        "Milling Husk Chaff — Rice/Cereal 10% (WHRB_10)": (RAW_DIR / "WHRB_10", "adulterated", 1),
    }

    results = {}
    rng = np.random.RandomState(42)

    for name, (folder, exp_str, exp_label) in profiles.items():
        if not folder.exists():
            continue
        all_imgs = sorted(list(folder.glob("*.jpg")))
        sample_imgs = [all_imgs[i] for i in rng.permutation(len(all_imgs))[:75]]

        correct = 0
        confs = []
        gate_passes = 0

        for p in sample_imgs:
            pil_img = load_and_center_crop(p)
            arr = np.array(pil_img, dtype=np.float32)
            _, _, passed = compute_gates(arr)
            if passed:
                gate_passes += 1

            probs = clf.predict(arr)
            pred = int(np.argmax(probs))
            conf = float(probs[pred])
            confs.append(conf)

            if pred == exp_label:
                correct += 1

        acc = correct / len(sample_imgs)
        avg_c = float(np.mean(confs))
        gate_r = gate_passes / len(sample_imgs)

        results[name] = {
            "tested": len(sample_imgs),
            "accuracy": round(acc, 4),
            "avg_confidence": round(avg_c, 4),
            "gate_pass_rate": round(gate_r, 4),
            "class": exp_str,
        }
        print(f"  [{acc*100:5.1f}% Acc] {name:50s} | Conf: {avg_c*100:5.1f}% | Gate: {gate_r*100:5.1f}%")

    return results

# ----------------- Test Suite 2: Ethiopian Kitchen & Market Lookalikes -----------------
def run_ethiopian_lookalikes_test(clf: TFLiteClassifier) -> dict:
    print("\n--- [TEST 2] Ethiopian Kitchen & Market Cross-Food Lookalikes ---")
    items = {}

    # Real assets from the app
    if (APP_IMAGES_DIR / "injera-texture.jpg").exists():
        items["Authentic Injera Flatbread (ayen texture)"] = load_and_center_crop(APP_IMAGES_DIR / "injera-texture.jpg")
    if (APP_IMAGES_DIR / "teff-flour-bag.jpg").exists():
        items["Authentic Ethiopian Teff Flour Bag"] = load_and_center_crop(APP_IMAGES_DIR / "teff-flour-bag.jpg")
    if (APP_IMAGES_DIR / "teff-grain.jpg").exists():
        items["Authentic Brown Teff Grains"] = load_and_center_crop(APP_IMAGES_DIR / "teff-grain.jpg")

    # Synthetic realistic Ethiopian spice & market materials
    # 1. Ethiopian Turmeric (Ird) - golden yellow
    arr_ird = np.tile(np.array([225.0, 185.0, 25.0], dtype=np.float32), (IMG_SIZE, IMG_SIZE, 1))
    arr_ird += np.random.normal(0, 12, arr_ird.shape).astype(np.float32)
    items["Ethiopian Turmeric Powder (Ird)"] = Image.fromarray(np.clip(arr_ird, 0, 255).astype(np.uint8))

    # 2. Highland Red Clay Soil (Ye-key afer) - dark rust red, low saturation
    arr_afer = np.tile(np.array([160.0, 75.0, 50.0], dtype=np.float32), (IMG_SIZE, IMG_SIZE, 1))
    arr_afer += np.random.normal(0, 14, arr_afer.shape).astype(np.float32)
    items["Highland Red Clay Soil (Ye-key afer)"] = Image.fromarray(np.clip(arr_afer, 0, 255).astype(np.uint8))

    # 3. Roasted Barley Powder (Beso) - warm greyish tan
    arr_beso = np.tile(np.array([190.0, 160.0, 120.0], dtype=np.float32), (IMG_SIZE, IMG_SIZE, 1))
    arr_beso += np.random.normal(0, 10, arr_beso.shape).astype(np.float32)
    items["Roasted Barley Powder (Beso)"] = Image.fromarray(np.clip(arr_beso, 0, 255).astype(np.uint8))

    # 4. Traditional Woven Sefed/Mesob Basket (Empty)
    items["Traditional Mesob/Sefed Woven Basket"] = add_container_boundary(
        Image.fromarray(np.full((IMG_SIZE, IMG_SIZE, 3), [190, 150, 105], dtype=np.uint8)), "woven_basket"
    )

    # 5. Blue-striped Woven Polypropylene Sack (Empty Madaberia)
    items["Woven Market Grain Sack (Madaberia)"] = add_container_boundary(
        Image.fromarray(np.full((IMG_SIZE, IMG_SIZE, 3), [235, 235, 240], dtype=np.uint8)), "plastic_sack"
    )

    results = {}
    blocked_count = 0

    for name, pil_img in items.items():
        arr = np.array(pil_img, dtype=np.float32)
        rr, tv, passed_gates = compute_gates(arr)
        probs = clf.predict(arr)
        pred = int(np.argmax(probs))
        conf = float(probs[pred])
        is_conclusive = conf >= CONFIDENCE_THRESHOLD

        # A non-chili sample is safely handled if intercepted by color gate, texture variance gate, or low confidence
        is_safe = (not passed_gates) or (not is_conclusive)
        if is_safe:
            blocked_count += 1

        reasons = []
        if rr < RED_RATIO_MIN:
            reasons.append(f"Failed Red Ratio ({rr*100:.1f}% < 12%)")
        if tv < TEXTURE_VARIANCE_MIN:
            reasons.append(f"Failed Texture Variance ({tv:.1f} < 150)")
        if not is_conclusive:
            reasons.append(f"Inconclusive Margin ({conf*100:.1f}% < 70%)")

        status = "SAFELY BLOCKED" if is_safe else "UNPROTECTED PASS"
        results[name] = {
            "red_ratio": round(rr, 4),
            "texture_variance": round(tv, 2),
            "passed_gates": passed_gates,
            "raw_prediction": "pure" if pred == 0 else "adulterated",
            "confidence": round(conf, 4),
            "is_safely_intercepted": is_safe,
            "reasons": reasons,
        }
        print(f"  [{status:17s}] {name:42s} -> RedRatio: {rr*100:4.1f}% | Var: {tv:6.1f} | Intercepted by: {', '.join(reasons) if reasons else 'None'}")

    safety_rate = blocked_count / len(items)
    print(f"\n  >>> Non-Chili Ethiopian Market Protection Rate: {safety_rate*100:.1f}% ({blocked_count}/{len(items)})")
    return {"cases": results, "safety_rate": round(safety_rate, 4)}

# ----------------- Test Suite 3: Multi-Perspective Geometry Matrix -----------------
def run_perspective_geometry_test(clf: TFLiteClassifier, n_samples: int = 60) -> dict:
    print("\n--- [TEST 3] Multi-Perspective Geometry & Market Presentation Matrix ---")
    pure_candidates = sorted(list((RAW_DIR / "C1_PWH").glob("*.jpg")))[: n_samples // 2]
    adulterated_candidates = sorted(list((RAW_DIR / "WHWB_10").glob("*.jpg")) + list((RAW_DIR / "WHWS_10").glob("*.jpg")))[: n_samples // 2]
    test_set = [(p, 0) for p in pure_candidates] + [(p, 1) for p in adulterated_candidates]

    perspectives = {
        "1. Perpendicular Planar View (90° Top-Down)": lambda im: im,
        "2. Casual Handheld Tilt (15° Oblique Angle)": lambda im: apply_perspective_tilt(im, 15.0),
        "3. Slanted Market Counter View (30° Angle)": lambda im: apply_perspective_tilt(im, 30.0),
        "4. Steep Grain Sack View (45° Severe Slant)": lambda im: apply_perspective_tilt(im, 45.0),
        "5. Midday Phone Shadow (Addis Ababa Sun)": lambda im: add_cast_phone_shadow(im),
        "6. High-Altitude UV Glare (Addis 2,400m Sun)": lambda im: ImageEnhance.Contrast(ImageEnhance.Brightness(im).enhance(1.35)).enhance(1.25),
        "7. Sample Inside Traditional Woven Mesob Basket": lambda im: add_container_boundary(im, "woven_basket"),
        "8. Sample Inside Grain Sack (Madaberia Border)": lambda im: add_container_boundary(im, "plastic_sack"),
        "9. Sample Inside Terracotta Clay Dish (Dist)": lambda im: add_container_boundary(im, "red_clay_pot"),
    }

    results = {}
    for p_name, fn in perspectives.items():
        correct = 0
        confs = []
        gates_passed = 0

        for path, true_label in test_set:
            base_pil = load_and_center_crop(path)
            transformed = fn(base_pil)
            arr = np.array(transformed, dtype=np.float32)

            _, _, passed = compute_gates(arr)
            if passed:
                gates_passed += 1

            probs = clf.predict(arr)
            pred = int(np.argmax(probs))
            conf = float(probs[pred])
            confs.append(conf)

            if pred == true_label:
                correct += 1

        total = len(test_set)
        acc = correct / total
        avg_c = float(np.mean(confs))
        gate_r = gates_passed / total

        results[p_name] = {
            "accuracy": round(acc, 4),
            "avg_confidence": round(avg_c, 4),
            "gate_pass_rate": round(gate_r, 4),
        }
        rating = "EXCELLENT" if acc >= 0.90 else ("GOOD" if acc >= 0.75 else "ATTENUATED")
        print(f"  [{rating:10s}] {p_name:48s} -> Acc: {acc*100:5.1f}% | Conf: {avg_c*100:5.1f}% | Gate: {gate_r*100:5.1f}%")

    return results

# ----------------- Visualizations -----------------
def generate_ethiopian_dashboard(adulterants: dict, lookalikes: dict, perspectives: dict) -> None:
    print("\n--- Generating African & Ethiopian Market Dashboard ---")
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1: Adulterant Sensitivity
    names1 = [k.split(" (")[0] for k in adulterants.keys()]
    accs1 = [v["accuracy"] * 100 for v in adulterants.values()]
    colors1 = ["#2E7D32" if "Pure" in n else ("#FF8F00" if "5%" in n else ("#E65100" if "10%" in n else "#C62828")) for n in names1]
    y1 = np.arange(len(names1))
    axes[0].barh(y1, accs1, color=colors1, edgecolor="black", height=0.68)
    axes[0].set_yticks(y1)
    axes[0].set_yticklabels(names1, fontsize=9)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Detection Accuracy (%)", fontsize=11, fontweight="bold")
    axes[0].set_title("A. Ethiopian / African Adulterant Sensitivity", fontsize=12, fontweight="bold")
    axes[0].set_xlim(70, 105)
    axes[0].grid(axis="x", linestyle=":", alpha=0.6)
    for i, a in enumerate(accs1):
        axes[0].text(a + 0.5, i, f"{a:.1f}%", va="center", fontsize=8.5, fontweight="bold")

    # Panel 2: Multi-Perspective Geometry Robustness
    names2 = [k.split(". ")[1] for k in perspectives.keys()]
    accs2 = [v["accuracy"] * 100 for v in perspectives.values()]
    colors2 = ["#1565C0" if a >= 90 else ("#FF8F00" if a >= 75 else "#D32F2F") for a in accs2]
    y2 = np.arange(len(names2))
    axes[1].barh(y2, accs2, color=colors2, edgecolor="black", height=0.68)
    axes[1].set_yticks(y2)
    axes[1].set_yticklabels(names2, fontsize=9)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Accuracy Under Perspective Distortion (%)", fontsize=11, fontweight="bold")
    axes[1].set_title("B. Multi-Perspective & Market Container Geometry", fontsize=12, fontweight="bold")
    axes[1].set_xlim(60, 105)
    axes[1].grid(axis="x", linestyle=":", alpha=0.6)
    for i, a in enumerate(accs2):
        axes[1].text(a + 0.6, i, f"{a:.1f}%", va="center", fontsize=8.5, fontweight="bold")

    plt.suptitle("Tirat AI — African & Ethiopian Market & Multi-Perspective Verification", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    chart_path = OUTPUTS_DIR / "ethiopian_market_dashboard.png"
    plt.savefig(chart_path, dpi=135)
    plt.close()
    print(f"  Saved dashboard plot -> {chart_path}")

def main() -> None:
    t0 = time.time()
    print("=" * 75)
    print("TIRAT AI — AFRICAN & ETHIOPIAN MARKET MULTI-PERSPECTIVE TEST")
    print("=" * 75)

    clf = TFLiteClassifier(TFLITE_PATH)

    # 1. Ethiopian Adulterants
    adulterants_res = run_ethiopian_adulterants_test(clf)

    # 2. Ethiopian Kitchen & Market Lookalikes
    lookalikes_res = run_ethiopian_lookalikes_test(clf)

    # 3. Perspective & Geometric Views
    perspectives_res = run_perspective_geometry_test(clf, n_samples=60)

    # 4. Dashboard
    generate_ethiopian_dashboard(adulterants_res, lookalikes_res, perspectives_res)

    # Report
    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": str(TFLITE_PATH.name),
        "adulterants": adulterants_res,
        "lookalikes": lookalikes_res,
        "perspectives": perspectives_res,
        "elapsed_sec": round(time.time() - t0, 2),
    }
    report_file = OUTPUTS_DIR / "ethiopian_test_report.json"
    report_file.write_text(json.dumps(summary, indent=2))
    print(f"\nAll Ethiopian & Multi-Perspective tests completed in {time.time() - t0:.1f}s.")
    print(f"Report saved to {report_file}")

if __name__ == "__main__":
    main()
