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

import * as ImageManipulator from "expo-image-manipulator";
import jpeg from "jpeg-js";
import { MODEL_INPUT_SIZE as SIZE } from "../config";
import { readBase64 } from "../util/fs";

async function getImageSize(uri: string): Promise<{ width: number; height: number }> {
  return new Promise((resolve, reject) => {
    // RN's Image.getSize works with file:// URIs; only dimensions are needed.
    const { Image } = require("react-native");
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

export async function imageToModelInput(uri: string): Promise<Float32Array> {
  // Step 1-2: resize + center crop via native image pipeline (fast, low memory).
  const dims = await getImageSize(uri);
  const resizeAction =
    dims.width < dims.height ? { resize: { width: SIZE } } : { resize: { height: SIZE } };
  const resized = await ImageManipulator.manipulateAsync(
    uri,
    [resizeAction],
    { compress: 0.9, format: ImageManipulator.SaveFormat.JPEG },
  );

  const originX = Math.max(0, Math.floor((resized.width - SIZE) / 2));
  const originY = Math.max(0, Math.floor((resized.height - SIZE) / 2));
  const cropped = await ImageManipulator.manipulateAsync(
    resized.uri,
    [{ crop: { originX, originY, width: SIZE, height: SIZE } }],
    {
      compress: 0.85,
      format: ImageManipulator.SaveFormat.JPEG,
      base64: true,
    },
  );
  if (!cropped.base64) throw new Error("ImageManipulator returned no base64 data");

  // Step 3: decode to RGBA. 224x224 decodes in a few ms even on old phones.
  // (typed cast: @types/jpeg-js predates Buffer-free usage patterns)
  const raw = (jpeg as any).decode(base64ToBytes(cropped.base64), {
    useTArray: true,
    formatAsRGBA: true,
  }) as { width: number; height: number; data: Uint8Array };

  // Step 4: RGBA interleaved -> planar RGB floats (NHWC, batch of 1).
  const out = new Float32Array(1 * SIZE * SIZE * 3);
  const px = raw.data;
  let o = 0;
  for (let i = 0; i < SIZE * SIZE; i++) {
    out[o++] = px[i * 4]; // R
    out[o++] = px[i * 4 + 1]; // G
    out[o++] = px[i * 4 + 2]; // B
  }
  return out;
}
