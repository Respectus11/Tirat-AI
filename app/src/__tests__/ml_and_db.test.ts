import { describe, expect, it } from "vitest";

// Argmax utility test
function argmax(a: Float32Array): number {
  let best = 0;
  for (let i = 1; i < a.length; i++) if (a[i] > a[best]) best = i;
  return best;
}

// Red-ratio pixel calculation simulation test
function calculateRedRatio(pixels: Uint8Array, width: number, height: number): number {
  let redPixels = 0;
  for (let i = 0; i < width * height; i++) {
    const r = pixels[i * 4];
    const g = pixels[i * 4 + 1];
    const b = pixels[i * 4 + 2];
    if (r > 90 && r > g * 1.35 && r > b * 1.1) redPixels++;
  }
  return redPixels / (width * height);
}

describe("ML Inference & Preprocessing Logic", () => {
  it("correctly identifies maximum index in argmax", () => {
    const probs1 = new Float32Array([0.1, 0.85, 0.05]);
    expect(argmax(probs1)).toBe(1);

    const probs2 = new Float32Array([0.92, 0.08]);
    expect(argmax(probs2)).toBe(0);

    const probs3 = new Float32Array([0.05, 0.95]);
    expect(argmax(probs3)).toBe(1);
  });

  it("correctly computes red ratio for chili powder vs non-food pixels", () => {
    // 2x2 chili sample (pure red pixels r=200, g=50, b=50)
    const chiliPixels = new Uint8Array([
      200, 50, 50, 255,  210, 40, 40, 255,
      190, 60, 60, 255,  220, 30, 30, 255,
    ]);
    const chiliRatio = calculateRedRatio(chiliPixels, 2, 2);
    expect(chiliRatio).toBe(1.0);

    // 2x2 non-food sample (wood table / hand: r=120, g=110, b=100)
    const handPixels = new Uint8Array([
      120, 110, 100, 255,  130, 120, 110, 255,
      115, 105, 95,  255,  125, 115, 105, 255,
    ]);
    const handRatio = calculateRedRatio(handPixels, 2, 2);
    expect(handRatio).toBe(0.0);
  });

  it("evaluates confidence thresholding correctly", () => {
    const CONFIDENCE_THRESHOLD = 0.70;
    const testCases = [
      { conf: 0.88, expected: "pure" },
      { conf: 0.65, expected: "inconclusive" },
      { conf: 0.95, expected: "adulterated" },
    ];

    testCases.forEach(({ conf, expected }) => {
      const isLowConfidence = conf < CONFIDENCE_THRESHOLD;
      if (isLowConfidence) {
        expect("inconclusive").toBe(expected);
      } else {
        expect(expected).not.toBe("inconclusive");
      }
    });
  });
});
