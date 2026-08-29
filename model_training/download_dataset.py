"""
download_dataset.py — fetch the published Teff flour adulteration dataset.

Source (open access, CC BY-NC 4.0):
    "Images datasets of pure and adulterated Teff (Eragrostis tef (Zucc.) Trotter)
     flour with wood flour (sawdust) and gypsum (calcium sulfate) powder ..."
    Harvard Dataverse: doi:10.7910/DVN/XKCEX3   (~30,018 files, several GB)
    Paper: Data in Brief 2026, doi:10.1016/j.dib.2026.112656

What it does:
    1. Downloads the whole dataset as one zip via the Dataverse access API,
       streaming to disk with retry + exponential backoff.
    2. Verifies the zip and extracts into model_training/data/_raw_download/.

If the API is down (Harvard Dataverse has had outages), print the exact manual
browser fallback instead of failing cryptically.

Stdlib only — no TensorFlow/pandas needed. Run with the same Python you'll
train with (3.10-3.12).

Usage:
    python download_dataset.py
    python download_dataset.py --keep-zip        # keep archive after extract
    python download_dataset.py --url <direct>    # e.g. a zip you mirrored
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
import zipfile
from pathlib import Path

import urllib.request
import urllib.error

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
RAW_DIR = DATA_DIR / "_raw_download"

DATASET_DOI = "doi:10.7910/DVN/XKCEX3"
BASE_URL = "https://dataverse.harvard.edu"
ZIP_URL = f"{BASE_URL}/api/access/dataset?persistentId={DATASET_DOI}"
LANDING_PAGE = f"{BASE_URL}/dataset.xhtml?persistentId={DATASET_DOI}"

CHUNK = 1024 * 1024  # 1 MiB
RETRIES = 5
TIMEOUT = 120


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}GB"


def stream_to_file(req: urllib.request.Request, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    downloaded = 0
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp, open(tmp, "wb") as fh:
        total = resp.headers.get("Content-Length")
        total = int(total) if total else None
        while True:
            block = resp.read(CHUNK)
            if not block:
                break
            fh.write(block)
            downloaded += len(block)
            if total:
                pct = 100.0 * downloaded / total
                sys.stdout.write(f"\r[download] {human(downloaded)} / {human(total)} ({pct:.1f}%)")
            else:
                sys.stdout.write(f"\r[download] {human(downloaded)}")
            sys.stdout.flush()
    print()
    tmp.replace(dest)


def download_zip(zip_path: Path) -> None:
    last_err: Exception | None = None
    for attempt in range(1, RETRIES + 1):
        try:
            print(f"[download] attempt {attempt}/{RETRIES}: {ZIP_URL}")
            req = urllib.request.Request(ZIP_URL, headers={"User-Agent": "tirat-training/1.0"})
            stream_to_file(req, zip_path)
            return
        except (urllib.error.URLError, TimeoutError, ConnectionError) as err:
            last_err = err
            wait = min(60, 5 * 2 ** (attempt - 1))
            print(f"[!] attempt failed: {err} — retrying in {wait}s")
            # A .part file from a dead connection is garbage; drop it.
            zip_path.with_suffix(zip_path.suffix + ".part").unlink(missing_ok=True)
            time.sleep(wait)

    print(
        "\n[!] Harvard Dataverse API unreachable after "
        f"{RETRIES} attempts ({last_err}).\n\n"
        "    MANUAL FALLBACK (their browser downloads usually still work during API outages):\n"
        f"      1. Open {LANDING_PAGE}\n"
        "      2. Click 'Access Dataset' -> 'Download Zip'\n"
        f"      3. Save the zip as: {zip_path}\n"
        f"         (or anywhere, then re-run: python download_dataset.py --url <path-to-zip>)\n"
    )
    raise SystemExit(1)


def verify_and_extract(zip_path: Path) -> None:
    print(f"[extract] verifying {zip_path.name} ...")
    if not zipfile.is_zipfile(zip_path):
        raise SystemExit(
            f"[!] {zip_path} is not a valid zip — the download may have been cut short "
            "or the server returned an HTML error page. Delete it and retry."
        )
    if RAW_DIR.exists():
        print(f"[extract] clearing old {RAW_DIR}")
        shutil.rmtree(RAW_DIR)
    RAW_DIR.mkdir(parents=True)

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        print(f"[extract] {len(names)} entries; extracting ...")
        zf.extractall(RAW_DIR)
    print(f"[extract] done -> {RAW_DIR}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Download the Teff adulteration dataset")
    ap.add_argument("--url", help="path/URL of an already-downloaded zip", default=None)
    ap.add_argument("--keep-zip", action="store_true", help="don't delete the zip after extraction")
    args = ap.parse_args()

    DATA_DIR.mkdir(exist_ok=True)
    zip_path = DATA_DIR / "teff_adulteration_dataset.zip"

    if args.url:
        src = Path(args.url)
        if src.exists():
            zip_path = src
        else:
            # treat as a direct URL override
            req = urllib.request.Request(args.url, headers={"User-Agent": "tirat-training/1.0"})
            stream_to_file(req, zip_path)
    else:
        download_zip(zip_path)

    verify_and_extract(zip_path)

    if not args.keep_zip:
        zip_path.unlink(missing_ok=True)
        print("[cleanup] zip deleted (re-run with --keep-zip to keep it)")

    print(
        "\nNext step:\n"
        "  python remap_dataset.py     # sorts images into pure/ wood/ gypsum/ folders\n"
        "  python data_prep.py --preview"
    )


if __name__ == "__main__":
    main()
