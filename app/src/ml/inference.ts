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
import labels from "../../assets/models/labels.json";
import redchiliLabels from "../../assets/models/redchili_labels.json";
import { CONFIDENCE_THRESHOLD } from "../config";
import { imageToModelInput } from "./preprocess";

export type FoodType = "teff" | "redchili";

export const TEFF_CLASS_KEYS = ["pure", "wood", "gypsum"] as const;
export const REDCHILI_CLASS_KEYS = ["pure", "adulterated"] as const;

export const CLASS_KEYS = TEFF_CLASS_KEYS;
export type VerdictKey = (typeof TEFF_CLASS_KEYS)[number] | "adulterated";
export const PCT_BINS: number[] = labels.pct_bins;

const modelCache: Partial<Record<FoodType, Promise<TfliteModel> | null>> = {};

function getModel(foodType: FoodType = "teff"): Promise<TfliteModel> {
  if (!modelCache[foodType]) {
    const asset =
      foodType === "redchili"
        ? require("../../assets/models/redchili_model.tflite")
        : require("../../assets/models/tirat_model.tflite");

    modelCache[foodType] = loadTensorflowModel(asset, []).catch((err) => {
      modelCache[foodType] = null;
      throw err;
    });
  }
  return modelCache[foodType]!;
}

export interface AnalysisOk {
  status: "ok";
  verdict: VerdictKey;
  confidence: number;
  estPct: number | null;
  foodType: FoodType;
}
export interface AnalysisFailed {
  status: "model_error" | "error";
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
  foodType: FoodType = "teff",
): Promise<AnalysisResult> {
  let model: TfliteModel;
  try {
    model = await getModel(foodType);
  } catch (err) {
    return { status: "model_error", message: String(err) };
  }

  try {
    const input = await imageToModelInput(photoUri);
    const outputs: ArrayBuffer[] = model.runSync([input.buffer as ArrayBuffer]);

    if (foodType === "redchili") {
      const classKeys = REDCHILI_CLASS_KEYS;
      const verdictBuf = outputs.find((b) => b.byteLength === classKeys.length * 4) || outputs[0];
      if (!verdictBuf) {
        return { status: "error", message: "Unexpected red chili model output tensors" };
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
    const verdictBuf = outputs.find((b) => b.byteLength === TEFF_CLASS_KEYS.length * 4);
    const pctBuf = outputs.find((b) => b.byteLength === PCT_BINS.length * 4);
    if (!verdictBuf || !pctBuf) {
      return { status: "error", message: "Unexpected model output tensors" };
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

