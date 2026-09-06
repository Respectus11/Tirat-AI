# Tirat AI â€” Senior Code Review (red chili focus)

> **Scope note:** This review targets the **red chili powder adulteration pipeline** â€”
> the only flow with real data, a trained model, and a validated `.tflite`. The
> **teff pipeline is intentionally out of scope** because the teff dataset is not
> available (`app/assets/models/tirat_model.tflite` is a **593-byte placeholder**,
> `labels.json` has `"dummy": true`). See Finding #3 â€” it ships to users today.

**Date:** 2026-09-04 Â· **Environment:** Windows, Node 24, Expo SDK 57, RN 0.86, TS ~6.0, Python 3.12.10 / TF 2.20.

---

## 1. Validation results (run on this machine)

| Check | Command | Result |
|---|---|---|
| TypeScript strict typecheck | `npm run typecheck` (`tsc --noEmit`) | âœ… **PASS** (clean) |
| Android bundle + asset resolution | `npx expo export --platform android` | âœ… **PASS** (produces `entry-*.hbc`, 3.6 MB; red-chili `.tflite` resolves) |
| Shipped model I/O contract | TF Lite interpreter inspection | âœ… **PASS** â€” input `float32[1,224,224,3]`, output `float32[1,2]` (binary softmax), matches `src/ml/inference.ts` (2 classes Ã— 4 B = 8 B) |
| Linting | `npx eslint .` | âŒ **FAIL / not configured** â€” no `eslint.config.*`; eslint v9 requires flat config (`eslint-config-expo` is declared but no config file ties it in) |
| `expo-doctor` | `npx expo-doctor` | âš ï¸ **Inconclusive** â€” hung on network package-resolution (not a project defect; rerun with network egress) |

**Bottom line:** the red-chili path *type-checks, bundles, and the shipped model matches the app's expected I/O*. Good baseline. The gaps below are about **trust, integrity, and hygiene** around a health-adjacent screening product.

---

## 2. Prioritized findings

### ðŸ”´ P0 â€” Correctness & integrity before any distribution

**P0-1 Â· Shipped model cannot be traced to reported metrics (provenance gap).**
`model_training/redchili/outputs/training_summary.json` describes a **16-class** model
(`pure` + 15 `WH###` folders) at **best_val_accuracy 0.4971** â€” but the **shipped
`redchili_model.tflite` is binary** (`pure`/`adulterated`, verified output `[1,2]`).
A `checkpoints/backup_78pct/` folder hints at a ~78% binary run, yet **no committed
artifact records which checkpoint â†’ export produced the shipped file, nor its real
held-out accuracy / precision / recall.** Anyone who builds the APK ships a model whose
real-world accuracy is **undocumented**. The UI shows a confidence number and an
"Adulterated/Pure" verdict â€” users will trust them. *Fix:* record a `RUN_LOG`/`eval_summary.json`
with checkpoint hash â†’ converted `.tflite` â†’ test-set metrics (accuracy, P/R per class,
and per-adulterant breakdown) committed next to the model.

**P0-2 Â· Default food type is teff, but teff is a dummy/placeholder model.**
`result.tsx:101` defaults `foodType ?? "teff"`, and the whole copy/tagline is teff-centric
(`en.json:2` "Teff flour adulteration screening"). A user tapping Scan â†’ capturing teff
gets **meaningless output from a 593-byte stub model** (or a `model_error`). Since the
teff dataset is unavailable, the app is shipping a broken default flow. *Fix:* make
**red chili the primary/only path** for the productized APK (or explicitly disable/hide
the teff tab), and update tagline/copy. This is the single biggest *product* trust issue.

**P0-3 Â· Life-safety disclaimer is textual only.** The screening tool states it is "not a
certified lab test" (`disclaimer`), which is good, but there is no forced-acknowledgment,
no per-result confidence caveat beyond an "inconclusive" threshold, and no mention that a
false *Pure* result on a genuine sample is possible. For a food-safety product, consider a
one-time informed-consent gate plus a persistent "results are AI estimates" banner.

### ðŸŸ  P1 â€” Important

**P1-1 Â· No automated tests, no CI, no working linter.** Only a `typecheck` script exists;
linting is **non-functional** (no eslint flat config). `src/ml/inference.ts` (argmax,
tensor-length discrimination), `src/ml/preprocess.ts` (base64 decode, resize/crop), and
`src/db/db.ts` (prune/sql) are all pure logic ideal for unit tests. Add Vitest + a lint
config + a GitHub Actions workflow that runs typecheck/lint/test on PRs.

**P1-2 Â· Class mapping is a hardcoded dataset-naming heuristic.**
`data_prep_redchili.py:154` maps folder prefixes `C1_PWH`, `C2_AWH`, `WH00` â†’ `pure`,
everything else â†’ `adulterated`. This is a fragile assumption about the Mendeley DS-WH-1
layout: any renamed/new folder silently flips ~1,500 pure samples into `adulterated`.
`WH00` in particular is asserted to be "pure grade" â€” document the evidence or verify.
Replace with an explicit, validated mapping table + a sanity check (e.g., assert pure
count / folder set before training).

**P1-3 Â· Training-vs-shipped artifact chain is not reproducible from the repo.**
`evaluate_redchili.py` loads `redchili_model.keras`, but the shipped file is the converted
`.tflite`; there is no committed `eval_summary.json` for *that* model (only the stale 16-class
`training_summary.json`). Add a documented script chain:
`data â†’ train â†’ convert â†’ evaluate(tflite) â†’ copy to app/assets/models`, and commit the
resulting metrics JSON with each model bump. The float16 conversion also has **no
numeric-equivalence check** vs the float32 source â€” add one.

**P1-4 Â· Teff vs. red-chili scope is inconsistent in docs.** `docs/architecture.md` documents
the binary-only red-chili v1 scope decision well, but the leftover **16-class**
`training_summary.json` and the teff-first README/tagline contradict the shipped reality.
Sweep docs to match what actually ships (red chili primary) and delete or clearly mark stale
artifacts.

### ðŸŸ¡ P2 â€” Quality & hygiene

- **P2-1 Â· Sync DB calls on the UI thread.** `expo-sqlite` sync API is used throughout
  (`openDatabaseSync`, `runSync`, `getAllSync`). At 500-history-row scale this is fine, but
  `getHistory()`/`pruneHistory()` on the main thread can cause jank with `FlatList` + full
  rows + photo paths. Consider async queries or capping/streaming. (Mild at MVP scale.)
- **P2-2 Â· Dead code.** `readBase64` in `src/util/fs.ts` is imported by `preprocess.ts` but
  never called (the pipeline uses `cropped.base64` from `expo-image-manipulator` directly).
  Remove it or wire it up. Also `getImageSize` uses a mid-function `require("react-native")`
  â€” works, but move imports to the top for clarity.
- **P2-3 Â· Upload consent overpromises.** `en.json:38` tells users samples "are shared
  anonymously for research," but `flushQueue()` is a **stub** (no network calls). The
  consent prompt precedes a feature that doesn't exist yet. Either gate the consent UI until
  the backend is real, or label it "experimental / not yet active" to avoid misleading users
  into consenting to a no-op.
- **P2-4 Â· Log gating â€” verified good.** All three locations from the earlier audit
  (`_layout.tsx:74`, `result.tsx:148`, `uploader.ts:23`) are correctly wrapped in
  `if (__DEV__)`. âœ…
- **P2-5 Â· Git hygiene.** Working tree has many untracked scratch artifacts:
  `model_training/redchili/*.txt` (env/pip search logs), `model_training/plan/`, `tmp_sharp/`,
  `tmp_test_images/` (a copy of the whole red-chili dataset), and deleted
  `app/.claude/settings.json`. Add a root `.gitignore` for these (esp. big data dirs) and
  commit the intentional `app.json` EAS `projectId`/`owner` change (currently uncommitted).
- **P2-6 Â· No `expo-doctor` / CI health gate.** The project can't verify itself on a fresh
  checkout without a human running manual commands. Fold the checks above into CI.

---

## 3. What's genuinely good (keep)

- **Excellent design-system + i18n engineering.** `theme.ts` tokens (contrast on bright sun,
  pressure for AA on `inkMuted`), Amharic-first default with English fallback, dual Ethiopic/Latin
  font stacks loaded with a splash screen â€” thoughtful, unusual maturity for an MVP.
- **Honest "WHY" comments.** `config.ts` (threshold rationale), `preprocess.ts` (in-graph
  normalization to kill train/serve skew), `db.ts` (photo lifecycle from cacheâ†’persist), and
  `uploader.ts` (`STUB` clearly flagged) â€” this is the rare repo that explains *why*, not just *what*.
- **Offline-first persistence handled correctly.** SQLite WAL + eager pruning that removes
  photo files from disk with their rows; result screen guards against double-save on retry.
- **Model-cache + robust tensor discrimination.** `inference.ts` caches loaded models and tells
  heads apart by byte-length rather than fragile ordering â€” resilient to export reordering.
- **Responsiveness to prior review.** The earlier race-condition fix (`isCancelled`) and dev-log
  gating were actually implemented in `result.tsx`/`_layout.tsx`/`uploader.ts`. Good discipline.
- **Strict TypeScript actually passing**, plus a clean single-purpose `src/` layering.

---

## 4. Recommended roadmap

**Now (before any user distribution):**
1. Make red chili the primary path; gate/hide the teff (dummy) flow. Update copy.
2. Reproduce the shipped model's metrics (or rebuild binary from `backup_78pct`/best checkpoint),
   and commit provenance + eval JSON next to the `.tflite`.

**Next 2â€“4 weeks:**
3. Add Vitest unit tests for `inference`/`preprocess`/`db`; wire up an eslint flat config; add a CI
   workflow (typecheck + lint + test + expo-doctor).
4. Harden `data_prep_redchili.py` class mapping with an explicit, validated table + dataset sanity checks.
5. Add a float32â†”float16 equivalence check to `convert_tflite_redchili.py`.

**When scaling:**
6. Async/reviewed DB access; real opt-in upload backend (or demote the consent copy); per-result
   uncertainty caveat; field-pilot lab-reverification loop (see `docs/architecture.md`).

---

*Review produced via static analysis + local validation on this machine. Metrics of the shipped
model, the teff dataset, and a live device smoke test remain the user's outstanding items.*

---

## 5. Phase 3 — APK build: found & fixed a release blocker

The repo previously **could not produce an APK** — `npm ci` (the exact command EAS
runs in `INSTALL_DEPENDENCIES`) hard-failed with an **ERESOLVE** error, and an earlier
cloud build errored the same way on 2026-09-03. Root cause: `package.json` pinned
`react@19.2.3` while the lockfile resolved `react-dom@19.2.8` (via `expo-router` →
`vaul` → `@radix-ui`), which requires `react@^19.2.8`; and `eslint-config-expo` was
declared in `package.json` but not installed.

**Fix applied:**
- Bumped `react` `19.2.3 → 19.2.8` in `package.json`.
- Re-ran `npm install` to regenerate `package-lock.json` (also installing
  `eslint-config-expo@57.0.2`).
- Verified `npm ci --dry-run` → **exit 0** (was exit 1). Remaining messages are
  only `peerOptional` warnings for `react-native-worklets` (non-fatal).
- `npm run typecheck` still passes.

**Recommendation (P0):** enforce lockfile reproducibility in CI with `npm ci` so a
regression like this can never block a release again. Keep `react`/`react-dom` aligned;
prefer `npx expo install` for RN/React updates so SDK-resolved versions stay in sync.
