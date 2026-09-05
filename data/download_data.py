"""Fetch the raw satellite archives from Zenodo into ``data/raw/``.

The Sentinel-1 / Sentinel-2 time series NetCDF is 280 MB, well above GitHub's
file size limit, so it is not redistributed here. It is published openly under
CC BY 4.0 as part of:

    Dunmire, D., Subramanian, A., Hossain, E., Gani, M. O., Banwell, A.,
    Younas, H., & Myers, B. (2025). Data and Code for: "Greenland Ice Sheet
    wide supraglacial lake evolution and dynamics: insights from the 2018 and
    2019 melt seasons" [Data set]. Zenodo.
    https://doi.org/10.5281/zenodo.14587026

Only ``notebooks/01_preprocessing`` needs these files. The processed CSVs used
by every experiment ship with the repository, so you can reproduce the paper's
results without running this script.

Usage
-----
    python data/download_data.py             # 2019 only (~65 MB download)
    python data/download_data.py --year 2018 # add the 2018 NetCDF
    python data/download_data.py --keep-archives
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import zipfile
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent / "raw"
ZENODO_RECORD = "14587026"
ZENODO_DOI = "10.5281/zenodo.14587026"
BASE_URL = f"https://zenodo.org/records/{ZENODO_RECORD}/files"

#: archive -> (md5, size in bytes) as reported by the Zenodo REST API.
ARCHIVES = {
    "INPUT.zip": ("2c7d4f92aa0d366d7798add94f9544df", 6_098_980),
    "OUTPUT.zip": ("7e2fc18fc71dd9ae00a917b26ee5152e", 59_205_856),
}

#: destination filename -> (archive, member path inside the archive)
MEMBERS = {
    "all_training.geojson": ("INPUT.zip", "INPUT/training_data/all_training.geojson"),
    "all_lakes_2019.nc": ("OUTPUT.zip", "OUTPUT/netcdfs/all_lakes_2019.nc"),
    "all_lakes_2018.nc": ("OUTPUT.zip", "OUTPUT/netcdfs/all_lakes_2018.nc"),
}


def md5sum(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(name: str, destination: Path) -> Path:
    """Download one Zenodo archive, reusing a valid local copy if present."""
    import requests

    expected_md5, expected_size = ARCHIVES[name]
    target = destination / name

    if target.exists() and md5sum(target) == expected_md5:
        print(f"  {name}: already downloaded and verified")
        return target

    url = f"{BASE_URL}/{name}?download=1"
    print(f"  {name}: downloading {expected_size / 1e6:.1f} MB from Zenodo...")

    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        written = 0
        with target.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                handle.write(chunk)
                written += len(chunk)
                pct = 100 * written / expected_size
                print(f"\r    {written / 1e6:7.1f} / {expected_size / 1e6:.1f} MB ({pct:5.1f}%)", end="")
    print()

    actual = md5sum(target)
    if actual != expected_md5:
        target.unlink(missing_ok=True)
        raise SystemExit(f"Checksum mismatch for {name}: expected {expected_md5}, got {actual}")
    print(f"  {name}: checksum verified")
    return target


def extract(archive_path: Path, member: str, destination: Path) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        try:
            info = archive.getinfo(member)
        except KeyError:
            raise SystemExit(f"{member} not found inside {archive_path.name}") from None
        with archive.open(info) as source, destination.open("wb") as target:
            shutil.copyfileobj(source, target)
    print(f"  extracted {destination.name} ({destination.stat().st_size / 1e6:.1f} MB)")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--year",
        type=int,
        nargs="*",
        default=[2019],
        choices=[2018, 2019],
        help="Melt season NetCDFs to fetch. The paper uses 2019 only.",
    )
    parser.add_argument("--keep-archives", action="store_true", help="Do not delete the downloaded zips.")
    parser.add_argument("--force", action="store_true", help="Re-extract even if the target exists.")
    args = parser.parse_args(argv)

    try:
        import requests  # noqa: F401
    except ImportError:
        raise SystemExit("This script needs 'requests'. Install it with: pip install requests") from None

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    wanted = ["all_training.geojson"] + [f"all_lakes_{year}.nc" for year in sorted(set(args.year))]
    wanted = [name for name in wanted if args.force or not (RAW_DIR / name).exists()]

    if not wanted:
        print(f"Everything is already present in {RAW_DIR}. Use --force to re-extract.")
        return 0

    print(f"Source: https://doi.org/{ZENODO_DOI}  (CC BY 4.0)")
    print(f"Target: {RAW_DIR}\n")

    needed_archives = sorted({MEMBERS[name][0] for name in wanted})
    downloaded = {name: download(name, RAW_DIR) for name in needed_archives}

    print()
    for name in wanted:
        archive, member = MEMBERS[name]
        extract(downloaded[archive], member, RAW_DIR / name)

    if not args.keep_archives:
        for path in downloaded.values():
            path.unlink(missing_ok=True)
        print("\nRemoved the downloaded archives (pass --keep-archives to retain them).")

    print("\nDone. Please cite Zenodo record " + ZENODO_DOI + " alongside the paper.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
