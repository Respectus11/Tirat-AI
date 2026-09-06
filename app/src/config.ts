// Central app configuration.

/**
 * Below this verdict-confidence the app refuses to answer ("Inconclusive").
 *
 * WHY 0.70: a screening tool's worst failure mode is confidently wrong answers
 * destroying user trust (and potentially harming health). A false "retake this
 * photo" costs seconds; a confident wrong verdict costs the product.
 * Tune after the domain-gap stress test (evaluate.py) gives real numbers.
 */
export const CONFIDENCE_THRESHOLD = 0.7;

/** Model input size — must match training (MobileNetV3Small @ 224x224). */
export const MODEL_INPUT_SIZE = 224;

/**
 * Fraction of pixels that must look like red chili for a photo to be analyzed
 * in red-chili mode. Hands, tables and other non-samples have almost no
 * strongly-red pixels; the powder always has a lot. Calibrated against the
 * test set's red-ratio distribution (see model_training/redchili/outputs).
 */
export const RED_RATIO_MIN = 0.12;

/** Max history rows kept locally; photos of older rows are pruned with them. */
export const HISTORY_LIMIT = 500;
