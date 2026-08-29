"""
remap_dataset.py — reorganize the downloaded Teff dataset into the layout
data_prep.py expects:

    model_training/data/
    +-- pure/*.jpg                  any teff variety, no adulterant
    +-- adulterated_wood/*.jpg      wood-flour (sawdust) samples
    +-- adulterated_gypsum/*.jpg    gypsum samples

Input is whatever download_dataset.py extracted into data/_raw_download/.
Per the Data in Brief paper the archive has:

    _raw_download/
      raw/                      <- 5,000 real photos  <-- WE USE THESE
        adulterants/            <- raw adulterant photos (NOT flour samples; skipped)
        teff_grains/            <- whole-grain photos    (NOT flour; skipped)
        <variety folders>       e.g. "white_teff_flour - gypsum_powder"
                                holding PURE *and* ADULTERATED shots
      processed/                <- 25,000 pixel-shuffle AUGMENTED copies

The processed/ tree is skipped BY DEFAULT: its near-duplicates leak between
train/val/test splits and inflate accuracy dishonestly. data_prep.py applies
its own split-safe augmentation instead. Use --include-augmented only for
sanity experiments.

Classification strategy (in order):
    1. data_statistics.csv (shipped in the archive) maps folders -> classes
    2. keyword rules on folder/file names (gypsum / wood / pure / level %)
Anything still ambiguous lands in data/_unsorted/ plus a written report —
never silently guessed.

Stdlib only. Usage:
    python remap_dataset.py
    python remap_dataset.py --include-augmented     # NOT recommended
    python remap_dataset.py --dry-run               # show plan, copy nothing
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
RAW_DIR = DATA_DIR / "_raw_download"

OUT_DIRS = {
    "pure": DATA_DIR / "pure",
    "wood": DATA_DIR / "adulterated_wood",
    "gypsum": DATA_DIR / "adulterated_gypsum",
}
UNSORTED_DIR = DATA_DIR / "_unsorted"
REPORT_PATH = DATA_DIR / "remap_report.json"

IMG_EXT = {".jpg", ".jpeg", ".png"}

# Same convention data_prep.py uses to read the adulteration level.
PCT_RE = re.compile(r"(?:^|[^0-9])(\d{1,2})\s*%?")
PCT_BINS = [10, 15, 20, 25, 30, 35, 40]


def parse_pct(name: str) -> int | None:
    m = PCT_RE.search(Path(name).stem)
    if not m:
        return None
    v = int(m.group(1))
    return v if v in PCT_BINS else None


def looks_pure(text: str) -> bool:
    t = text.lower()
    return ("pure" in t) or bool(re.search(r"(?:^|[^0-9])100\s*%?(?:[^0-9]|$)", t))


def classify_from_keywords(rel_path: Path) -> str | None:
    """Return 'pure' | 'wood' | 'gypsum' | None from path + file name."""
    parts = [p.lower().replace("-", "_").replace(" ", "_") for p in rel_path.parts]
    joined = "/".join(parts)
    stem = rel_path.stem.lower()

    # explicit pure markers win over adulterant words appearing in shared folders
    if looks_pure(stem):
        return "pure"

    if "gypsum" in joined:
        return "gypsum" if parse_pct(stem) is not None else (
            "pure" if looks_pure(joined) else None
        )
    if "wood" in joined:
        return "wood" if parse_pct(stem) is not None else (
            "pure" if looks_pure(joined) else None
        )

    if parse_pct(stem) is not None:
        # a level tag but no adulterant word anywhere -> cannot pick a side safely
        return None

    # no level tag, no keywords: treat as pure ONLY if folder says nothing either
    return "pure" if looks_pure(joined) else None


def load_csv_folder_map(csv_path: Path) -> dict[str, str]:
    """Parse data_statistics.csv into {folder_name_lower: 'pure'|'wood'|'gypsum'}.

    Column names are normalized; rows whose class column doesn't clearly name a
    class are ignored (keyword fallback handles them).
    """
    mapping: dict[str, str] = {}
    try:
        with open(csv_path, newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            cols = {c.lower(): c for c in (reader.fieldnames or [])}
            folder_col = next((orig for low, orig in cols.items() if "folder" in low), None)
            label_cols = [
                orig for low, orig in cols.items()
                if any(k in low for k in ("class", "label")) or low in {"category", "type"}
            ]
            if not folder_col or not label_cols:
                return mapping
            for row in reader:
                folder = (row.get(folder_col) or "").strip()
                if not folder:
                    continue
                label_blob = " ".join((row.get(c) or "") for c in label_cols).lower()
                if "gypsum" in label_blob:
                    cls = "gypsum"
                elif "wood" in label_blob:
                    cls = "wood"
                elif "pure" in label_blob:
                    cls = "pure"
                else:
                    continue
                mapping[folder.strip().lower()] = cls
    except Exception as err:  # noqa: BLE001 - CSV is optional metadata
        print(f"[csv] ignoring data_statistics.csv ({err})")
    return mapping


def classify_from_csv_map(rel_path: Path, folder_map: dict[str, str]) -> str | None:
    """Use the CSV folder->class map: nearest ancestor folder with a known mapping."""
    stem = rel_path.stem.lower()
    if looks_pure(stem):
        return "pure"
    pct = parse_pct(stem)
    for part in reversed(rel_path.parts[:-1]):
        hit = folder_map.get(part.strip().lower())
        if hit:
            if hit == "pure":
                return "pure"
            # adulterant-labelled folder: only trust files carrying a level tag.
            # Per the paper these folders ALSO contain pure shots, so a file with
            # neither a % tag nor a 'pure' marker stays ambiguous on purpose.
            return hit if pct is not None else None
    return None


def unique_dest(dest_dir: Path, src_name: str) -> Path:
    """Avoid clobbering same-named files from different variety folders."""
    candidate = dest_dir / src_name
    if not candidate.exists():
        return candidate
    slug = re.sub(r"[^a-z0-9]+", "_", Path(src_name).stem.lower())[:40].strip("_")
    return dest_dir / f"{slug}__{src_name}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Remap Teff dataset into pipeline layout")
    ap.add_argument("--include-augmented", action="store_true",
                    help="ALSO copy the processed/ pixel-shuffle copies (leakage risk!)")
    ap.add_argument("--dry-run", action="store_true", help="report only, copy nothing")
    args = ap.parse_args()

    if not RAW_DIR.exists():
        raise SystemExit(
            f"[!] {RAW_DIR} not found.\n"
            "    Run: python download_dataset.py   (or extract your manual zip there)"
        )

    # locate shipped metadata CSV anywhere under the extraction root
    csv_files = list(RAW_DIR.rglob("data_statistics.csv"))
    folder_map = load_csv_folder_map(csv_files[0]) if csv_files else {}
    print(f"[scan] folder->class mappings from metadata CSV: {len(folder_map)}")

    img_exts = IMG_EXT
    stats: Counter[str] = Counter()
    per_source: dict[str, Counter[str]] = defaultdict(Counter)
    unsorted_samples: list[str] = []
    n_copied = n_skipped = 0

    all_images = sorted(
        p for p in RAW_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in img_exts and not p.name.startswith(".")
    )
    print(f"[scan] {len(all_images)} image files found under {RAW_DIR.name}/")

    for src in all_images:
        rel = src.relative_to(RAW_DIR)

        # --- skip non-sample trees ------------------------------------------
        top_parts = {p.lower() for p in rel.parts}
        if not args.include_augmented and "processed" in top_parts:
            continue
        if "teff_grain" in " ".join(top_parts):
            continue
        if "adulterants" in top_parts:
            continue

        # --- classify --------------------------------------------------------
        cls = None
        if folder_map:
            cls = classify_from_csv_map(rel, folder_map)
        if cls is None:
            cls = classify_from_keywords(rel)

        if cls is None:
            UNSORTED_KEY = "_unsorted"
            stats[UNSORTED_KEY] += 1
            per_source[str(rel.parent)][UNSORTED_KEY] += 1
            if len(unsorted_samples) < 30:
                unsorted_samples.append(str(rel).replace("\\", "/"))
            continue

        stats[cls] += 1
        per_source[str(rel.parent)]["__total__"] += 1

        if args.dry_run:
            continue

        dest_dir = OUT_DIRS[cls]
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = unique_dest(dest_dir, src.name)
        if dest.exists() and dest.stat().st_size == src.stat().st_size:
            n_skipped += 1
            continue
        shutil.copy2(src, dest)
        n_copied += 1

    # stash genuinely ambiguous files where they can be inspected
    if not args.dry_run and stats["_unsorted"]:
        UNSORTED_DIR.mkdir(exist_ok=True)
        print(f"[!] {stats['_unsorted']} ambiguous files -> listed in the report "
              "(left in place; inspect and extend the rules)")

    # pct-tag coverage per adulterated class (drives the second training head)
    coverage: dict[str, dict] = {}
    for cls_key in ("wood", "gypsum"):
        folder = OUT_DIRS[cls_key]
        files = list(folder.glob("*")) if folder.exists() else []
        tagged = sum(1 for f in files if parse_pct(f.name) is not None)
        coverage[cls_key] = {
            "files": len(files),
            "with_pct_tag": tagged,
            "coverage": f"{100.0 * tagged / len(files):.1f}%" if files else "n/a",
        }

    report = {
        "dry_run": args.dry_run,
        "counts": dict(stats),
        "copied": n_copied,
        "already_present": n_skipped,
        "pct_tag_coverage": coverage,
        "unsorted_examples": unsorted_samples,
        "per_source_folder": {k: dict(v) for k, v in sorted(per_source.items())},
    }
    if not args.dry_run:
        REPORT_PATH.write_text(json.dumps(report, indent=2))

    print(json.dumps({k: v for k, v in report.items() if k != "per_source_folder"}, indent=2))
    print("\nCounts:", dict(stats))
    if args.dry_run:
        print("(dry run — nothing copied)")
    else:
        print(f"\nReport -> {REPORT_PATH}")
    print(
        "\nNext steps:\n"
        "  python data_prep.py --preview   # verify splits & pct-tag coverage\n"
        "  (if coverage is poor, adjust PCT_PATTERN in data_prep.py to the real filenames)"
    )


if __name__ == "__main__":
    main()
