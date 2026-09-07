// On-device inference via react-native-fast-tflite (Nitro/JSI — zero-copy).
//
// Model contract (see assets/models/labels.json + model_training/convert_tflite.py):
//   input : float32[1, 224, 224, 3], raw 0..255 RGB
//   out A : "verdict"  softmax[1, 3]   classes: pure | wood | gypsum
//   out B : "pct_bins" softmax[1, 7]   bins:    10..40% w/w in 5% steps
//
// The two outputs are told apart by byte length (12 vs 28) rather than order,
// so reordering heads during export can't silently swap them.

import { loadTensorflowModel, type TfliteModel } from "react-native-fast-tflite";
import { Asset } from "expo-asset";
import * as FileSystem from "expo-file-system/legacy";
import labels from "../../assets/models/labels.json";
import redchiliLabels from "../../assets/models/redchili_labels.json";
import { CONFIDENCE_THRESHOLD, RED_RATIO_MIN, TEFF_BROWN_RATIO_MIN, TEXTURE_VARIANCE_MIN } from "../config";
import { imageToModelInput } from "./preprocess";

export type FoodType = "teff" | "redchili";

export const TEFF_CLASS_KEYS = ["pure", "wood", "gypsum"] as const;
export const REDCHILI_CLASS_KEYS = ["pure", "adulterated"] as const;

export type VerdictKey = (typeof TEFF_CLASS_KEYS)[number] | (typeof REDCHILI_CLASS_KEYS)[number];
export const PCT_BINS: number[] = labels.pct_bins;

const modelCache: Partial<Record<FoodType, Promise<TfliteModel> | null>> = {};

// Model loading strategy (the release APK has no Metro server — the JS bundle and
// the models are embedded, and react-native-fast-tflite's native loader reads the
// source with plain `java.net.URL(path).readBytes()`, which only understands real
// file paths and http(s) URLs. Metro-bundled res/ entries and android_asset URIs
// are NOT readable by java.net.URL). Two loaders, first-fit:
//   1) expo-asset — the pattern from fast-tflite's own docs: Asset.fromModule +
//      downloadAsync() extracts the bundled resource to a real file in both dev
//      and release (same mechanism that makes expo-font work offline).
//   2) android_asset fallback — the .tflite files are ALSO packaged as native
//      Android assets (android/app/src/main/assets/models/, staged from
//      assets/models/ at configuration time in android/app/build.gradle); we copy
//      one to the app cache and hand fast-tflite a plain file:// URL.
const MODEL_FILES: Record<FoodType, string> = {
  teff: "tirat_model.tflite",
  redchili: "redchili_model.tflite",
};

/* eslint-disable @typescript-eslint/no-require-imports */
const MODEL_MODULES = {
  teff: require("../../assets/models/tirat_model.tflite"),
  redchili: require("../../assets/models/redchili_model.tflite"),
};
/* eslint-enable @typescript-eslint/no-require-imports */

async function materializeFromAndroidAsset(foodType: FoodType): Promise<string> {
  const fileName = MODEL_FILES[foodType];
  const dir = `${FileSystem.cacheDirectory}models`;
  const dest = `${dir}/${fileName}`;
  const bundledUri = `file:///android_asset/models/${fileName}`;

  // Always overwrite: keeps the cache in sync when a retrained model ships.
  await FileSystem.makeDirectoryAsync(dir, { intermediates: true });
  try {
    await FileSystem.copyAsync({ from: bundledUri, to: dest });
  } catch {
    // Fallback path in case copyAsync can't stream from the asset space.
    const b64 = await FileSystem.readAsStringAsync(bundledUri, {
      encoding: FileSystem.EncodingType.Base64,
    });
    await FileSystem.writeAsStringAsync(dest, b64, {
      encoding: FileSystem.EncodingType.Base64,
    });
  }
  return dest;
}

function getModel(foodType: FoodType = "redchili"): Promise<TfliteModel> {
  if (!modelCache[foodType]) {
    modelCache[foodType] = (async () => {
      try {
        const asset = Asset.fromModule(MODEL_MODULES[foodType]);
        await asset.downloadAsync();
        // localUri = real file in release, Metro http URL in dev — java.net.URL
        // reads both, which is exactly what fast-tflite's native loader needs.
        const uri = asset.localUri ?? asset.uri;
        const model = await loadTensorflowModel({ url: uri }, []);
        console.log(`[tflite] ${foodType} model loaded (expo-asset: ${uri})`);
        return model;
      } catch (e) {
        console.warn(`[tflite] expo-asset load failed for ${foodType}: ${String(e)}`);
      }
      const localPath = await materializeFromAndroidAsset(foodType);
      const model = await loadTensorflowModel({ url: `file://${localPath}` }, []);
      console.log(`[tflite] ${foodType} model loaded (android_asset)`);
      return model;
    })().catch((err) => {
      // Clear the cache entry so subsequent scan attempts can retry loading,
      // instead of permanently returning null after a transient failure.
      delete modelCache[foodType];
      throw err;
    });
  }
  return modelCache[foodType]!;
}

/**
 * Warm up both models right after launch so the first scan doesn't pay the
 * native model-load cost while the user is watching the analyzing screen.
 */
export async function preloadModels(): Promise<void> {
  const results = await Promise.allSettled([getModel("teff"), getModel("redchili")]);
  for (let i = 0; i < results.length; i++) {
    const r = results[i];
    if (r.status === "rejected") {
      // Never swallow preload failures silently — they show up in logcat.
      console.warn(`[tflite] preload of model #${i} failed: ${String(r.reason)}`);
    }
  }
}

export interface AnalysisOk {
  status: "ok";
  verdict: VerdictKey;
  confidence: number;
  estPct: number | null;
  foodType: FoodType;
}
export interface AnalysisFailed {
  status: "model_error" | "error" | "not_food";
  message: string;
}
export type AnalysisResult = AnalysisOk | AnalysisFailed;

function argmax(a: Float32Array): number {
  let best = 0;
  for (let i = 1; i < a.length; i++) if (a[i] > a[best]) best = i;
  return best;
}

export async function analyzePhoto(
  photoUri: string,
  foodType: FoodType = "redchili",
): Promise<AnalysisResult> {
  let model: TfliteModel;
  try {
    model = await getModel(foodType);
  } catch (err) {
    return { status: "model_error", message: String(err) };
  }

  try {
    const { input, redRatio, brownRatio, textureVariance } = await imageToModelInput(photoUri);
    const outputs: ArrayBuffer[] = model.runSync([input.buffer as ArrayBuffer]);

    // Universal texture gate: solid-color surfaces (walls, screens, paper,
    // dark pockets) have near-zero pixel variance — not food for any mode.
    if (textureVariance < TEXTURE_VARIANCE_MIN) {
      return {
        status: "not_food",
        message: `Image texture variance ${textureVariance.toFixed(0)} < minimum ${TEXTURE_VARIANCE_MIN} — appears to be a uniform surface, not a food sample`,
      };
    }

    if (foodType === "redchili") {
      // Red chili color gate: chili powder fills the frame with red pixels.
      // Hands, tables and walls don't — reject before the model can give a
      // false verdict.
      if (redRatio < RED_RATIO_MIN) {
        return {
          status: "not_food",
          message: `Red-dominant pixel ratio ${(redRatio * 100).toFixed(0)}% < threshold`,
        };
      }

      const classKeys = REDCHILI_CLASS_KEYS;
      const verdictBuf = outputs.find((b) => b.byteLength === classKeys.length * 4);
      if (!verdictBuf) {
        return {
          status: "error",
          message: `Unexpected red chili model output tensors (${outputs
            .map((b) => b.byteLength)
            .join(", ")} bytes)`,
        };
      }
      const verdictProbs = new Float32Array(verdictBuf);
      const idx = argmax(verdictProbs);
      const confidence = verdictProbs[idx];
      return {
        status: "ok",
        verdict: classKeys[idx],
        confidence,
        estPct: null,
        foodType,
      };
    }

    // Teff pipeline
    // Teff color gate: flour is warm brown/tan — reject non-teff objects.
    if (brownRatio < TEFF_BROWN_RATIO_MIN) {
      return {
        status: "not_food",
        message: `Brown/tan pixel ratio ${(brownRatio * 100).toFixed(0)}% < threshold — does not look like teff flour`,
      };
    }
    const verdictBuf = outputs.find((b) => b.byteLength === TEFF_CLASS_KEYS.length * 4);
    const pctBuf = outputs.find((b) => b.byteLength === PCT_BINS.length * 4);
    if (!verdictBuf || !pctBuf) {
      return {
        status: "error",
        message: `Unexpected model output tensors (${outputs
          .map((b) => b.byteLength)
          .join(", ")} bytes)`,
      };
    }

    const verdictProbs = new Float32Array(verdictBuf);
    const pctProbs = new Float32Array(pctBuf);

    const idx = argmax(verdictProbs);
    const confidence = verdictProbs[idx];

    if (idx === 0 || confidence < CONFIDENCE_THRESHOLD) {
      return { status: "ok", verdict: TEFF_CLASS_KEYS[idx], confidence, estPct: null, foodType };
    }

    return {
      status: "ok",
      verdict: TEFF_CLASS_KEYS[idx],
      confidence,
      estPct: PCT_BINS[argmax(pctProbs)],
      foodType,
    };
  } catch (err) {
    return { status: "error", message: String(err) };
  }
}

