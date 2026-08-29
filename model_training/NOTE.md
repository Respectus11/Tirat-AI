# READ ME FIRST — dataset and environment

## 1. The dataset (published, open access — CC BY-NC 4.0)

**"Images datasets of pure and adulterated Teff (Eragrostis tef (Zucc.) Trotter) flour
with wood flour (sawdust) and gypsum (calcium sulfate) powder for machine learning based
identification of Teff flour adulteration and prediction of the level adulteration"**

- Bochera, Gizaw & Onime (Addis Ababa University), Harvard Dataverse
- **DOI: [10.7910/DVN/XKCEX3](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/XKCEX3)**
  (~5,000 real photos; the archive also ships 25,000 pixel-shuffle augmented copies)
- Data article: *Data in Brief* (2026), doi:10.1016/j.dib.2026.112656 · PubMed 42272826
- Companion model paper: IEEE ICSAI 2024, doi:10.1109/ICSAI65059.2024.10893729
- Levels are exactly our bins: **10–40% w/w in 5% steps**; varieties white/mixed/red

Fetch and sort it in two commands:

```bash
python download_dataset.py    # zip via Dataverse API -> data/_raw_download/
python remap_dataset.py       # -> data/pure, data/adulterated_wood, data/adulterated_gypsum
```

`remap_dataset.py --dry-run` previews the plan first. It skips `processed/`
(augmented duplicates leak between splits) and writes `data/remap_report.json`.
If Harvard Dataverse's API is down (they have outages), the script prints a manual
browser fallback — downloading through the website still works.

After that, `data_prep.py` reads this layout:

```
model_training/data/
+-- pure/*.jpg                  any teff variety, no adulterant
+-- adulterated_wood/*.jpg      wood-flour (sawdust) adulterated samples
+-- adulterated_gypsum/*.jpg    gypsum-adulterated samples
```

### Adulteration-% labels come from FILENAMES

The three folders only provide the 3-class verdict label. The second output head
(adulteration percentage) is trained from numbers encoded in filenames.
`data_prep.py` extracts them with this constant at the top of the file:

```python
PCT_PATTERN = r"(?:^|[^0-9])(\d{1,2})\s*%?"
```

It matches names like `sample_25.jpg` or `wood_30pct_003.png`. ADJUST this regex to
match your dataset's actual naming convention before training — `remap_dataset.py`
prints the %-tag coverage per class so you know if it needs fixing. Files without a
parseable level in the 10-40 range get a weak fallback label (class-conditional median
bin) and are flagged in the summary printed by `data_prep.py` — if more than ~30% of
files fall back, stop and fix the pattern instead of training on garbage labels.

## 2. Python version warning

TensorFlow does NOT support Python 3.13/3.14 yet, and this machine has Python 3.14 as
default. Install Python 3.12 (python.org, side-by-side install is fine) and create a venv:

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Order of operations

1. `pip install -r requirements.txt`
2. Optionally: `python make_dummy_model.py` (valid-but-random model so the mobile app runs)
3. Add dataset to `data/` per above
4. `python data_prep.py --preview`   sanity check splits + augmentation preview PNG
5. `python train.py`                 checkpoints + curves land in `outputs/`
6. `python evaluate.py`              metrics + simulated phone-photo stress test
7. `python convert_tflite.py`        writes `../app/assets/models/tirat_model.tflite`

## 4. Honest-labels policy

This repo deliberately ships NO data and NO fabricated images. The dummy model contains
random weights; its predictions are meaningless. All accuracy claims must come from the
real dataset plus a field pilot (see docs/architecture.md).
