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

/** Max history rows kept locally; photos of older rows are pruned with them. */
export const HISTORY_LIMIT = 500;
