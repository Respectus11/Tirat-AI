# Tirat AI — v0.2.0 Release Notes (production-ready build)

**Released:** Sept 2026 · **versionName** 0.2.0 · **versionCode** 2
**APK:** `app/android/app/build/outputs/apk/release/app-release.apk` (173 MB, signed, JS bundle embedded — runs with **no Metro / no USB needed**)

---

## What was fixed

### Stability ("stuck at opening")
- Launch failsafe in `app/_layout.tsx`: 12 s timeout for fonts/DB, ErrorBoundary, splash **always** hides, errors surface instead of an infinite splash.
- Camera UI moved out of `<CameraView>` children (which Android logged as invalid) into absolute siblings; torch now defaults **off**.
- **Release APK embeds the JS bundle** — the #1 cause of "stuck at opening" (dev build waiting on Metro) is structurally gone.

### Model & verdicts
- **Red-chili model retrained on corrected labels: 99.2% test accuracy** (was 49.7% — `C2_AWH`, the adulterated class, had been mapped to *pure*; ~20% of the dataset was label-poisoned).
- Converted to TFLite (1.95 MB) at `app/assets/models/redchili_model.tflite`; I/O shapes verified against `redchili_labels.json`.
- **Automatic "not a chili sample" rejection** (no user photo collection needed): confidence threshold + softmax margin/entropy + red-pixel color prior (gate 0.12, minimum across in-distribution samples measured at 0.28). Shows an amber *"Not a chili powder sample — retake"* verdict, fully translated (en/am), not saved as pure/adulterated.
- Teff mode keeps a clearly-labeled dummy model until the real Harvard dataset is trained (see Follow-ups).

### Data integrity & i18n
- Red-chili "adulterated" no longer mislabeled as `"wood"` in history; `food_type` column added (safe migration); history badges + stats are food-aware.
- All hardcoded English moved into `src/i18n/en.json` + `am.json`.

### Branding
- New teff-panicle mark generated as SVG → PNG via `tmp_sharp/gen.js` (sharp).
- Fixed opaque-black adaptive-icon foreground, added proper white monochrome, fixed splash config mismatch (now `contain`, consistent light/dark).

---

## Signing — keep this safe

| Item | Value |
|---|---|
| Keystore | `app/keystores/tirat-release.keystore` (**outside** `android/` so `expo prebuild` can't wipe it) |
| Store password | `tirat2026release` |
| Key alias | `tirat` |
| Key password | `tirat2026release` |

Credentials live in `app/android/gradle.properties` as `TIRAT_UPLOAD_*` and are referenced by `signingConfigs.release` in `app/android/app/build.gradle` (falls back to debug key when absent). `.gitignore` excludes `keystores/`.

> ⚠️ `expo prebuild` regenerates `android/`, which **wipes `gradle.properties`** — re-add the 4 `TIRAT_UPLOAD_*` lines after any prebuild (kept in this file as the source of truth).

## How to rebuild

```powershell
cd "c:\Users\hp\Desktop\Tirat AI\app\android"
.\gradlew.bat assembleRelease
adb install -r app\build\outputs\apk\release\app-release.apk
```
(Requires JDK 17 at `C:\Program Files\Java\jdk-17`, Android SDK at `%LOCALAPPDATA%\Android\Sdk`.)

## Known notes / follow-ups

1. **APK is 173 MB** — universal ABI build. Add `abiFilters`/splits per-ABI (arm64-v8a alone ≈ halves it) or switch to AAB for Play Store.
2. **Real teff model** — dataset: Harvard Dataverse DOI `10.7910/DVN/XKCEX3` (~5 k images). Pipeline documented in `model_training/NOTE.md`: download → remap → `data_prep` → `train` → `evaluate` → `convert_tflite` → replace dummy.
3. **OOD gate thresholds** calibrated on the red-chili test set; if false rejects appear in the field, lower the red-ratio gate in `app/src/config.ts` (`MIN_RED_RATIO`).
4. Model training artifacts + evaluation logs: `model_training/redchili/` (`train_out.log`, `eval_out.log`, `convert_out.log`, `phase2_checks.log`).
