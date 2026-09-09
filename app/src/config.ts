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

/**
 * Fraction of warm brown/tan pixels that must be present for a photo to be
 * analyzed in teff mode. Teff flour is warm golden-brown; blue clothes,
 * green plants and white walls will fall well below this threshold.
 */
export const TEFF_BROWN_RATIO_MIN = 0.15;

/**
 * Minimum pixel-level RGB variance across the image. Real food powder has
 * visible grain texture. Solid-color surfaces (walls, screens, paper,
 * dark pockets) have near-zero variance and are rejected as not-food.
 */
export const TEXTURE_VARIANCE_MIN = 150;

/**
 * Minimum average adjacent-pixel gradient sharpness. Blurry or out-of-focus
 * photos lose particulate boundary contrast and degrade adulteration sensitivity.
 * Photos below this threshold trigger an automatic retake prompt.
 */
export const SHARPNESS_MIN = 2.4;

/** Max history rows kept locally; photos of older rows are pruned with them. */
export const HISTORY_LIMIT = 500;
