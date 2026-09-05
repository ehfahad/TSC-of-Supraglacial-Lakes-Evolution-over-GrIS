"""Loading of the preprocessed supraglacial lake time series.

The processed CSVs hold one lake per row: an ``ids`` column, 365 daily
``backscatter_diff`` columns, optionally 365 daily ``water_percentage``
columns, and a ``label`` column.

Following the paper (Section 4.1) only the melt-season window is used: day
121 (1 May) through day 364 inclusive, giving 244 daily steps per channel. See
:data:`START_DAY` and :data:`END_DAY`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .model import CLASSES

__all__ = [
    "DATA_DIR",
    "START_DAY",
    "END_DAY",
    "LABEL_TO_CODE",
    "load_split",
    "load_backscatter",
    "load_backscatter_water",
]

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "processed"

#: First and last day-of-year retained, inclusive. Day 121 is 1 May 2019.
START_DAY = 121
END_DAY = 364

#: Class name to integer code, consistent with :data:`rpsgmm.model.CLASSES`.
LABEL_TO_CODE = {name: code for code, name in enumerate(CLASSES)}

_FILES = {
    ("backscatter", "representative"): "backscatter_representative.csv",
    ("backscatter", "full"): "backscatter_777.csv",
    ("backscatter_water", "representative"): "backscatter_water_representative.csv",
    ("backscatter_water", "full"): "backscatter_water_777.csv",
}

_CHANNELS = {
    "backscatter": ("backscatter_diff",),
    "backscatter_water": ("backscatter_diff", "water_percentage"),
}


def load_split(
    features: str,
    split: str,
    *,
    data_dir: Path | None = None,
    start_day: int = START_DAY,
    end_day: int = END_DAY,
) -> tuple[np.ndarray, np.ndarray, pd.Index]:
    """Load one split as ``(X, y, ids)``.

    Parameters
    ----------
    features
        ``"backscatter"`` for the Sentinel-1 only setting, or
        ``"backscatter_water"`` for Sentinel-1 combined with Sentinel-2.
    split
        ``"representative"`` for the one-sample-per-class training set, or
        ``"full"`` for the 777-lake evaluation set.

    Returns
    -------
    X
        Array of shape ``(n_samples, n_channels, n_time_steps)``.
    y
        Integer class codes indexing :data:`rpsgmm.model.CLASSES`.
    ids
        Lake identifiers, aligned with the rows of ``X``.
    """
    if features not in _CHANNELS:
        raise ValueError(f"features must be one of {sorted(_CHANNELS)}, got {features!r}")
    try:
        filename = _FILES[(features, split)]
    except KeyError:
        raise ValueError(f"split must be 'representative' or 'full', got {split!r}") from None

    path = (data_dir or DATA_DIR) / filename
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Processed CSVs ship with the repository; if you "
            "removed them, regenerate with notebooks/01_preprocessing."
        )

    frame = pd.read_csv(path)
    days = range(start_day, end_day + 1)

    channels = []
    for prefix in _CHANNELS[features]:
        columns = [f"{prefix}_{day}" for day in days]
        missing = [c for c in columns if c not in frame.columns]
        if missing:
            raise KeyError(f"{path.name} is missing {len(missing)} column(s), e.g. {missing[0]}")
        channels.append(frame[columns].to_numpy(dtype=float))

    X = np.stack(channels, axis=1)  # (n_samples, n_channels, n_time_steps)

    unknown = set(frame["label"]) - set(LABEL_TO_CODE)
    if unknown:
        raise ValueError(f"{path.name} contains unexpected label(s): {sorted(unknown)}")
    y = frame["label"].map(LABEL_TO_CODE).to_numpy()

    return X, y, pd.Index(frame["ids"], name="ids")


def load_backscatter(split: str, **kwargs):
    """Sentinel-1 only. Shorthand for ``load_split("backscatter", split)``."""
    return load_split("backscatter", split, **kwargs)


def load_backscatter_water(split: str, **kwargs):
    """Sentinel-1 and Sentinel-2. Shorthand for ``load_split("backscatter_water", split)``."""
    return load_split("backscatter_water", split, **kwargs)
