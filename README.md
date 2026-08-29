# ጥራት (Tirat)  Teff Flour Adulteration Screening

**Tirat** (Amharic for "quality") is a mobile app that uses **on-device machine
learning** to detect adulteration in teff flour, Ethiopia's staple grain. Unscrupulous
vendors commonly bulk up teff flour with cheap fillers — **wood flour (sawdust)** and
**gypsum (calcium sulfate) powder** — to cheat on weight and volume.

You take a photo of a flour sample with your phone; a TensorFlow Lite model running
entirely on the device classifies it as:

| Verdict | Meaning |
|---|---|
| `Pure` | No adulterant detected |
| `Adulterated — wood flour` | Sawdust contamination detected |
| `Adulterated — gypsum` | Calcium-sulfate contamination detected |

…plus an estimate of adulteration level in the **10–40% range (5% steps)**, matching
the bins used in published research. If model confidence is low (<70%) the app refuses
to guess and asks for a retake.

> ⚠️ Tirat is a **fast screening tool**, not a certified lab test. See
> `docs/risks.md` for the full risk and liability discussion.

---

## Project structure

```
Tirat AI/
├── model_training/     Python: dataset prep → training → TFLite export
│   ├── data/           ← YOU put the image dataset here (see NOTE.md)
│   ├── outputs/        trained models, plots, metrics land here
│   ├── requirements.txt
│   ├── data_prep.py    load/split/augment/cache TF datasets
│   ├── train.py        MobileNetV3Small transfer learning (verdict + % heads)
│   ├── evaluate.py     confusion matrix, per-class P/R, domain-gap stress test
│   ├── convert_tflite.py  Keras → float16 + int8 TFLite → app/assets/models/
│   ├── make_dummy_model.py  generates a placeholder .tflite so the app runs NOW
│   └── NOTE.md         ⚠️ READ THIS before training (dataset + Python version)
├── app/                React Native (Expo, TypeScript) mobile app — Android first
│   └── assets/models/  bundled .tflite model + labels.json
├── docs/
│   ├── architecture.md technical architecture & data strategy
│   └── risks.md        domain gap, lookalike adulterants, liability
└── README.md           this file
```

## Quick start

### 1. Mobile app (works immediately with the dummy model)

```bash
cd app
npm install
npx eas login          # free Expo account needed once
eas build -p android --profile preview   # cloud build → installable APK
```

Install the produced APK on an Android phone and run end-to-end: capture → inference →
history, fully offline. The repo ships a **placeholder model file** so everything builds;
predictions are meaningless until you either run `make_dummy_model.py` (valid but random
model) or train for real. Local iteration alternative: `npx expo run:android`
(requires Android Studio + JDK 17). Expo Go will NOT work — TFLite needs native code.

### 2. Model training pipeline

```bash
# Requires Python 3.10–3.12 (TensorFlow does not support 3.14 yet — see NOTE.md)
py -3.12 -m venv .venv && .venv\Scripts\activate     # Windows example
pip install -r model_training/requirements.txt

python model_training/download_dataset.py    # fetches the published dataset (doi:10.7910/DVN/XKCEX3)
python model_training/remap_dataset.py       # sorts into data/{pure,adulterated_wood,adulterated_gypsum}
python model_training/data_prep.py --preview # sanity-check splits + augmentation
python model_training/train.py               # trains, saves curves to outputs/
python model_training/evaluate.py            # metrics + simulated-phone-photo test
python model_training/convert_tflite.py      # writes app/assets/models/tirat_model.tflite
```

Rebuild/reinstall the APK after swapping in a newly converted model.

## Status / roadmap

- [x] Training pipeline scaffolding (runs the moment a dataset is added)
- [x] Expo app: capture guide → on-device TFLite inference → results → offline history
- [x] Opt-in "help improve Tirat" upload queue (backend stubbed)
- [x] Amharic-default UI with English fallback (`app/src/i18n/`)
- [ ] Real dataset sourced & trained model shipped
- [ ] Field pilot: crowdsourced photos, lab re-verification loop (see docs/architecture.md)
- [ ] Backend for opt-in uploads

See `docs/architecture.md` for how verified lab data and unverified field data are meant
to combine, and why the confidence threshold exists.
