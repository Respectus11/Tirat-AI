// Image preprocessing: camera photo -> Float32Array ready for MobileNetV3Small.
//
// Pipeline:
//   1. resize shortest side to 224 (aspect preserved)
//   2. center-crop to exactly 224x224
//   3. decode JPEG base64 -> RGBA pixels (pure JS; no native dependency)
//   4. interleave RGB floats in 0..255
//
// WHY raw 0..255 floats: the normalization (Rescaling) lives INSIDE the model
// graph, baked in during training/export. The app therefore ships zero
// preprocessing math that could drift from training — the single biggest source
// of silent train/serve skew in on-device ML.

import { Image } from "react-native";
import * as ImageManipulator from "expo-image-manipulator";
import jpeg from "jpeg-js";
import { MODEL_INPUT_SIZE as SIZE } from "../config";

async function getImageSize(uri: string): Promise<{ width: number; height: number }> {
  return new Promise((resolve, reject) => {
    Image.getSize(
      uri,
      (width: number, height: number) => resolve({ width, height }),
      (err: unknown) => reject(err),
    );
  });
}

/** Hermes exposes atob; tiny manual fallback keeps this testable anywhere. */
function base64ToBytes(b64: string): Uint8Array {
  if (typeof atob === "function") {
    return Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  }
  const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  const clean = b64.replace(/[^A-Za-z0-9+/=]/g, "");
  const bytes = new Uint8Array(Math.floor((clean.length * 3) / 4));
  let p = 0;
  for (let i = 0; i < clean.length; i += 4) {
    const n =
      (chars.indexOf(clean[i]) << 18) |
      (chars.indexOf(clean[i + 1]) << 12) |
      ((chars.indexOf(clean[i + 2]) & 63) << 6) |
      (chars.indexOf(clean[i + 3]) & 63);
    bytes[p++] = (n >> 16) & 255;
    if (clean[i + 2] !== "=") bytes[p++] = (n >> 8) & 255;
    if (clean[i + 3] !== "=") bytes[p++] = n & 255;
  }
  return bytes.subarray(0, p);
}

export interface PreprocessResult {
  input: Float32Array;
  /** Fraction of strongly red-dominant pixels (chili gate). */
  redRatio: number;
  /** Fraction of warm brown/tan pixels (teff gate). */
  brownRatio: number;
  /** Pixel-level variance across RGB channels (texture gate: walls/screens ≈ 0). */
  textureVariance: number;
}

export async function imageToModelInput(
  uri: string,
): Promise<PreprocessResult> {
  const dims = await getImageSize(uri);

  // Single native pass: resize shortest side to SIZE (aspect preserved), then
  // center-crop to exactly SIZE x SIZE. Crop offsets are computed analytically
  // so no second ImageManipulator round-trip is needed.
  const scale = SIZE / Math.min(dims.width, dims.height);
  const resizedW = Math.max(SIZE, Math.round(dims.width * scale));
  const resizedH = Math.max(SIZE, Math.round(dims.height * scale));
  const cropped = await ImageManipulator.manipulateAsync(
    uri,
    [
      { resize: { width: resizedW, height: resizedH } },
      {
        crop: {
          originX: Math.floor((resizedW - SIZE) / 2),
          originY: Math.floor((resizedH - SIZE) / 2),
          width: SIZE,
          height: SIZE,
        },
      },
    ],
    {
      compress: 0.85,
      format: ImageManipulator.SaveFormat.JPEG,
      base64: true,
    },
  );
  if (!cropped.base64) throw new Error("ImageManipulator returned no base64 data");

  // Decode to RGBA. 224x224 decodes in a few ms even on old phones.
  // (typed cast: @types/jpeg-js predates Buffer-free usage patterns)
  const raw = (jpeg as any).decode(base64ToBytes(cropped.base64), {
    useTArray: true,
    formatAsRGBA: true,
  }) as { width: number; height: number; data: Uint8Array };

  // RGBA interleaved -> planar RGB floats (NHWC, batch of 1), and in the same
  // loop measure color ratios and texture variance for not-food gating.
  const totalPx = SIZE * SIZE;
  const out = new Float32Array(1 * totalPx * 3);
  const px = raw.data;
  let o = 0;
  let redPixels = 0;
  let brownPixels = 0;
  let sumR = 0, sumG = 0, sumB = 0;
  let sumR2 = 0, sumG2 = 0, sumB2 = 0;
  for (let i = 0; i < totalPx; i++) {
    const r = px[i * 4];
    const g = px[i * 4 + 1];
    const b = px[i * 4 + 2];
    out[o++] = r;
    out[o++] = g;
    out[o++] = b;

    // Red-dominant pixel gate (chili powder is saturated red/orange;
    // skin, wood tables and walls are not). Tuned on DS-WH-1 test set.
    if (r > 90 && r > g * 1.35 && r > b * 1.1) redPixels++;

    // Brown/tan pixel gate (teff flour is warm golden-brown;
    // blue shirts, green plants, and white walls are not).
    if (r > 100 && g > 60 && r > b * 1.2 && g > b * 0.9 && r < 240 && Math.abs(r - g) < 80) brownPixels++;

    // Accumulate for variance (catch solid-color surfaces like walls).
    sumR += r; sumG += g; sumB += b;
    sumR2 += r * r; sumG2 += g * g; sumB2 += b * b;
  }

  // Variance across all channels: low variance = uniform surface = not food.
  const varR = sumR2 / totalPx - (sumR / totalPx) ** 2;
  const varG = sumG2 / totalPx - (sumG / totalPx) ** 2;
  const varB = sumB2 / totalPx - (sumB / totalPx) ** 2;
  const textureVariance = (varR + varG + varB) / 3;

  return {
    input: out,
    redRatio: redPixels / totalPx,
    brownRatio: brownPixels / totalPx,
    textureVariance,
  };
}
