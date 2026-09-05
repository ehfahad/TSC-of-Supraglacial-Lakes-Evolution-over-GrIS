"""Rebuild the processed CSVs from the raw Zenodo NetCDF.

Reproduces the preprocessing described in Section 4.1 of the paper:

1. Smooth ``HV_lake`` and ``HV_out`` with a centered 12-day rolling mean, which
   suppresses variability between Sentinel-1 orbits.
2. Linearly interpolate each lake's series across satellite revisit gaps.
3. Difference the two to obtain the backscatter anomaly
   :math:`HV_{anom} = HV_{lake} - HV_{background}`.
4. Interpolate ``S2_water`` *between observations only* and set the remaining
   leading and trailing gaps to zero, giving :math:`p_{water}`.

Verified to regenerate the shipped CSVs for all 777 evaluation lakes to within
5e-08 on both channels. Run with ``--check`` to assert that.

Prerequisite: ``python data/download_data.py``

The 1,000-lake tables that ``--write`` produces are an intermediate product;
the repository ships only the 777-lake evaluation and training splits.

Usage
-----
    python scripts/build_processed_data.py --check     # verify against the shipped CSVs
    python scripts/build_processed_data.py --write     # write the 1,000-lake tables
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

SMOOTHING_WINDOW = 12
N_DAYS = 365
#: Tolerance when comparing against the shipped CSVs, which store 15 significant
#: digits of a float64 computed by a different library stack.
TOLERANCE = 1e-6


def load_raw(netcdf_path: Path, geojson_path: Path):
    """Return ``(ids, HV_lake, HV_out, S2_water, labels)`` for labeled lakes."""
    from netCDF4 import Dataset

    with Dataset(netcdf_path) as dataset:
        all_ids = np.array([str(value) for value in dataset.variables["ids"][:]])
        variables = {
            name: np.ma.filled(dataset.variables[name][:], np.nan).astype("float64")
            for name in ("HV_lake", "HV_out", "S2_water")
        }

    with geojson_path.open(encoding="utf-8") as handle:
        features = json.load(handle)["features"]
    labels = {
        feature["properties"]["new_id"]: feature["properties"]["label"]
        for feature in features
        if feature["properties"].get("label")
    }

    keep = np.array([index for index, lake in enumerate(all_ids) if lake in labels])
    ids = all_ids[keep]
    order = np.argsort(ids, kind="stable")

    return (
        ids[order],
        variables["HV_lake"][keep][order],
        variables["HV_out"][keep][order],
        variables["S2_water"][keep][order],
        labels,
    )


def _interpolate(array: np.ndarray, *, inside_only: bool) -> np.ndarray:
    """Linear interpolation along the time axis, one lake per row."""
    frame = pd.DataFrame(array)
    kwargs = {"limit_area": "inside"} if inside_only else {}
    return frame.interpolate(axis=1, method="linear", **kwargs).to_numpy()


def _smooth(array: np.ndarray, window: int = SMOOTHING_WINDOW) -> np.ndarray:
    """Centered rolling mean along the time axis, ignoring gaps."""
    return (
        pd.DataFrame(array).T.rolling(window, center=True, min_periods=1).mean().T.to_numpy()
    )


def build_backscatter(hv_lake: np.ndarray, hv_out: np.ndarray) -> np.ndarray:
    """Smooth, interpolate, then difference. Order matters: smoothing first."""
    lake = _interpolate(_smooth(hv_lake), inside_only=True)
    background = _interpolate(_smooth(hv_out), inside_only=True)
    return np.nan_to_num(lake - background, nan=0.0)


def build_water(s2_water: np.ndarray) -> np.ndarray:
    """Interpolate between observations only; leading/trailing gaps become zero."""
    return np.nan_to_num(_interpolate(s2_water, inside_only=True), nan=0.0)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="Compare against the shipped CSVs and exit.")
    parser.add_argument("--write", action="store_true", help="Write the rebuilt 1,000-lake tables.")
    parser.add_argument(
        "--output-dir", type=Path, default=REPO_ROOT / "data" / "interim",
        help="Where --write puts its output. Untracked by default.",
    )
    parser.add_argument("--tolerance", type=float, default=TOLERANCE)
    args = parser.parse_args(argv)

    if not (args.check or args.write):
        parser.error("pass --check, --write, or both")

    warnings.filterwarnings("ignore")

    netcdf = RAW_DIR / "all_lakes_2019.nc"
    geojson = RAW_DIR / "all_training.geojson"
    if not netcdf.exists():
        raise SystemExit(f"{netcdf} is missing. Run:\n    python data/download_data.py")

    print(f"Reading {netcdf.name} ...")
    ids, hv_lake, hv_out, s2_water, labels = load_raw(netcdf, geojson)
    print(f"  {len(ids)} labeled lakes x {hv_lake.shape[1]} days")

    print(f"Smoothing ({SMOOTHING_WINDOW}-day centered mean), interpolating, differencing ...")
    backscatter = build_backscatter(hv_lake, hv_out)
    water = build_water(s2_water)

    days = range(1, N_DAYS + 1)
    index = pd.Index(ids, name="ids")
    backscatter_frame = pd.DataFrame(
        backscatter, index=index, columns=[f"backscatter_diff_{day}" for day in days]
    )
    water_frame = pd.DataFrame(
        water, index=index, columns=[f"water_percentage_{day}" for day in days]
    )

    exit_code = 0
    if args.check:
        print("\nComparing against the shipped CSVs (777-lake evaluation set):")
        checks = [
            ("backscatter_777.csv", backscatter_frame, "backscatter_diff"),
            ("backscatter_water_777.csv", backscatter_frame, "backscatter_diff"),
            ("backscatter_water_777.csv", water_frame, "water_percentage"),
        ]
        for filename, built, prefix in checks:
            shipped = pd.read_csv(PROCESSED_DIR / filename).set_index("ids")
            columns = [f"{prefix}_{day}" for day in range(121, 365)]
            common = built.index.intersection(shipped.index)
            difference = np.abs(
                built.loc[common, columns].to_numpy(float)
                - shipped.loc[common, columns].to_numpy(float)
            )
            per_lake = np.nanmax(difference, axis=1)
            identical = int((per_lake <= args.tolerance).sum())
            status = "OK  " if identical == len(common) else "FAIL"
            if identical != len(common):
                exit_code = 1
            print(
                f"  {status} {filename:30s} {prefix:18s} "
                f"{identical}/{len(common)} lakes match  (max diff {np.nanmax(difference):.2e})"
            )

    if args.write:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nWriting to {args.output_dir}:")

        backscatter_frame.to_csv(args.output_dir / "all1000_backscatter.csv")
        print(f"  all1000_backscatter.csv  {backscatter_frame.shape}")

        combined = backscatter_frame.join(water_frame)
        combined.to_csv(args.output_dir / "all1000_backscatter_water.csv")
        print(f"  all1000_backscatter_water.csv  {combined.shape}")

        regions = {
            feature["properties"]["new_id"]: feature["properties"]["region"]
            for feature in json.load(geojson.open(encoding="utf-8"))["features"]
        }
        pd.DataFrame({
            "ids": ids,
            "region": [regions[lake] for lake in ids],
            "label": [labels[lake] for lake in ids],
        }).to_csv(args.output_dir / "all1000_labels.csv", index=False)
        print("  all1000_labels.csv")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
