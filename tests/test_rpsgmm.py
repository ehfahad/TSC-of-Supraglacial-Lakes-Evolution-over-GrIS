"""Tests for the RPS-GMM implementation.

Run with::

    python -m pytest tests -q
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rpsgmm import CLASSES, RPSGMMClassifier, embed, load_split  # noqa: E402
from rpsgmm.data import END_DAY, START_DAY  # noqa: E402
from rpsgmm.rps import phase_space_length  # noqa: E402

N_TIME_STEPS = END_DAY - START_DAY + 1


def reference_embed(series: np.ndarray, tau: int, d: int) -> np.ndarray:
    """Straightforward loop implementation, used as the oracle."""
    if series.ndim == 1:
        series = series.reshape(1, 1, -1)
    elif series.ndim == 2:
        series = series.reshape(1, *series.shape)
    n_samples, n_features, n_time = series.shape
    length = n_time - (d - 1) * tau
    if length <= 0:
        return np.empty((0, d * n_features))

    stacked = []
    for i in range(n_samples):
        blocks = []
        for feature in range(n_features):
            channel = series[i, feature]
            vectors = np.zeros((length, d))
            for j in range(d):
                vectors[:, j] = channel[j * tau: j * tau + length]
            blocks.append(vectors)
        stacked.append(np.hstack(blocks))
    return np.vstack(stacked)


# ------------------------------------------------------------------ embedding

@pytest.mark.parametrize("tau,d", [(1, 2), (2, 3), (11, 9), (13, 15), (7, 20), (30, 3)])
@pytest.mark.parametrize("shape", [(244,), (2, 244), (5, 1, 244), (5, 2, 244)])
def test_embed_matches_reference(tau, d, shape):
    series = np.random.default_rng(0).normal(size=shape)
    np.testing.assert_allclose(embed(series, tau, d), reference_embed(series, tau, d))


def test_embed_shape():
    series = np.zeros((5, 2, 244))
    result = embed(series, 11, 9)
    length = phase_space_length(244, 11, 9)
    assert result.shape == (5 * length, 9 * 2)


def test_embed_returns_empty_when_window_too_long():
    assert embed(np.zeros(50), tau=30, d=10).shape == (0, 10)


def test_embed_rejects_bad_parameters():
    with pytest.raises(ValueError):
        embed(np.zeros(100), tau=0, d=3)


def test_embed_coordinates_are_delayed_copies():
    series = np.arange(20.0)
    result = embed(series, tau=2, d=3)
    # Column j must be the series advanced by j * tau.
    np.testing.assert_allclose(result[:, 0], series[0:16])
    np.testing.assert_allclose(result[:, 1], series[2:18])
    np.testing.assert_allclose(result[:, 2], series[4:20])


# ----------------------------------------------------------------------- data

@pytest.mark.parametrize("features,n_channels", [("backscatter", 1), ("backscatter_water", 2)])
def test_load_shapes(features, n_channels):
    X_train, y_train, train_ids = load_split(features, "representative")
    X_eval, y_eval, eval_ids = load_split(features, "full")

    assert X_train.shape == (3, n_channels, N_TIME_STEPS)
    assert X_eval.shape == (777, n_channels, N_TIME_STEPS)
    assert sorted(y_train) == [0, 1, 2]
    assert len(eval_ids) == len(y_eval) == 777


def test_class_balance_matches_paper():
    _, y, _ = load_split("backscatter", "full")
    counts = {CLASSES[code]: int((y == code).sum()) for code in range(len(CLASSES))}
    assert counts == {"refreeze": 189, "drain": 392, "buried": 196}


def _alignment(ids_from, ids_to):
    """Row permutation mapping ``ids_to`` onto the order of ``ids_from``."""
    position = {lake: index for index, lake in enumerate(ids_to)}
    return np.array([position[lake] for lake in ids_from])


def test_both_feature_sets_cover_the_same_lakes():
    """The two CSVs hold the same lakes, though not in the same row order.

    Row order differs between the files; each is internally consistent, so the
    comparison is made after aligning on lake ID.
    """
    _, y_b, ids_b = load_split("backscatter", "full")
    _, y_bw, ids_bw = load_split("backscatter_water", "full")

    assert set(ids_b) == set(ids_bw)
    assert len(set(ids_b)) == 777, "lake IDs must be unique"
    np.testing.assert_array_equal(y_b, y_bw[_alignment(ids_b, ids_bw)])


def test_backscatter_channel_is_shared_between_feature_sets():
    X_b, _, ids_b = load_split("backscatter", "full")
    X_bw, _, ids_bw = load_split("backscatter_water", "full")
    np.testing.assert_allclose(X_b[:, 0, :], X_bw[_alignment(ids_b, ids_bw), 0, :])


def test_load_split_rejects_unknown_arguments():
    with pytest.raises(ValueError):
        load_split("nonexistent", "full")
    with pytest.raises(ValueError):
        load_split("backscatter", "nonexistent")


# ---------------------------------------------------------------- classifier

def test_class_order_is_a_fixed_sequence():
    """CLASSES must be an ordered sequence, never a set.

    A set's iteration order is randomized per interpreter process, so using one
    here would silently permute predictions between runs.
    """
    assert isinstance(CLASSES, (tuple, list))
    assert CLASSES == ("refreeze", "drain", "buried")


def test_predictions_are_deterministic_across_processes():
    """The same inputs must give the same accuracy in a fresh interpreter."""
    snippet = (
        "import os, sys; sys.path.insert(0, os.environ['RPSGMM_SRC'])\n"
        "import warnings; warnings.filterwarnings('ignore')\n"
        "from rpsgmm import RPSGMMClassifier, load_split\n"
        "Xtr, ytr, _ = load_split('backscatter', 'representative')\n"
        "Xte, yte, _ = load_split('backscatter', 'full')\n"
        "clf = RPSGMMClassifier(tau=13, d=15, n_init=1).fit(Xtr, ytr)\n"
        "print(f'{clf.score(Xte, yte):.10f}')\n"
    )
    # Passed through the environment rather than interpolated into the source:
    # the repository path may contain quotes.
    env = {**os.environ, "RPSGMM_SRC": str(REPO_ROOT / "src")}

    outputs = set()
    for _ in range(3):
        completed = subprocess.run(
            [sys.executable, "-c", snippet], capture_output=True, text=True, check=True, env=env
        )
        outputs.add(completed.stdout.strip())
    assert len(outputs) == 1, f"accuracy varied across processes: {outputs}"


def test_fit_predict_roundtrip():
    X_train, y_train, _ = load_split("backscatter", "representative")
    clf = RPSGMMClassifier(tau=13, d=15, n_components=5, n_init=1).fit(X_train, y_train)

    predictions = clf.predict(X_train)
    assert predictions.shape == (3,)
    assert set(predictions) <= set(range(len(CLASSES)))

    scores = clf.log_likelihoods(X_train)
    assert scores.shape == (3, len(CLASSES))
    np.testing.assert_array_equal(np.argmax(scores, axis=1), predictions)


def test_batched_scoring_matches_per_sample_scoring():
    """log_likelihoods() batches all lakes; it must equal scoring one at a time."""
    X_train, y_train, _ = load_split("backscatter", "representative")
    X_eval, _, _ = load_split("backscatter", "full")
    clf = RPSGMMClassifier(tau=13, d=15, n_components=5, n_init=1).fit(X_train, y_train)

    subset = X_eval[:20]
    batched = clf.log_likelihoods(subset)
    one_at_a_time = np.vstack([clf.log_likelihoods(sample[None]) for sample in subset])
    np.testing.assert_allclose(batched, one_at_a_time, rtol=1e-10, atol=1e-10)


def test_fit_requires_tau_and_d():
    X_train, y_train, _ = load_split("backscatter", "representative")
    with pytest.raises(ValueError, match="tau and d"):
        RPSGMMClassifier().fit(X_train, y_train)


def test_predict_requires_fit():
    X_eval, _, _ = load_split("backscatter", "full")
    with pytest.raises(ValueError, match="not fitted"):
        RPSGMMClassifier(tau=13, d=15).predict(X_eval)


def test_confusion_matrix_is_square_and_totals_match():
    X_train, y_train, _ = load_split("backscatter", "representative")
    X_eval, y_eval, _ = load_split("backscatter", "full")
    clf = RPSGMMClassifier(tau=13, d=15, n_components=5, n_init=1).fit(X_train, y_train)
    matrix = clf.confusion_matrix(X_eval, y_eval)
    assert matrix.shape == (3, 3)
    assert matrix.sum() == len(y_eval)


def test_grid_search_skips_only_infeasible_embeddings():
    """A (tau, d) is evaluated exactly when its delay vectors fit the window.

    A delay vector spans (d-1)*tau days, so from an N-day window only
    L = N - (d-1)*tau vectors can be formed, and a GMM needs at least
    n_components of them. Combinations past that bound are impossible, not
    skipped for convenience.
    """
    X_train, y_train, _ = load_split("backscatter", "representative")
    X_eval, y_eval, _ = load_split("backscatter", "full")

    tau_range, d_range = range(2, 12), range(3, 20)
    n_components = 10
    clf = RPSGMMClassifier(n_components=n_components, n_init=1)
    grid = clf.grid_search(
        X_train, y_train, X_eval, y_eval, tau_range, d_range, verbose=False
    )

    expected = {
        (tau, d)
        for tau in tau_range
        for d in d_range
        if N_TIME_STEPS - (d - 1) * tau >= n_components
    }
    assert {(row["tau"], row["d"]) for row in grid.history} == expected


def test_parallel_grid_search_matches_serial():
    """--n-jobs must not change any result, only the wall time."""
    pytest.importorskip("joblib")
    X_train, y_train, _ = load_split("backscatter", "representative")
    X_eval, y_eval, _ = load_split("backscatter", "full")

    tau_range, d_range = range(2, 8), range(3, 9)
    grids, optima = {}, {}
    for n_jobs in (1, 2):
        clf = RPSGMMClassifier(n_components=10, n_init=1, random_state=42)
        grid = clf.grid_search(
            X_train, y_train, X_eval, y_eval, tau_range, d_range,
            verbose=False, n_jobs=n_jobs,
        )
        grids[n_jobs] = {(r["tau"], r["d"]): r["accuracy"] for r in grid.history}
        optima[n_jobs] = (grid.tau, grid.d, grid.accuracy)

    assert optima[1] == optima[2], f"chosen operating point differs: {optima}"
    assert grids[1].keys() == grids[2].keys()
    for key in grids[1]:
        # Exact equality, not approximate: every fit is separately seeded.
        assert grids[1][key] == grids[2][key], f"accuracy differs at {key}"


def test_grid_search_finds_a_valid_operating_point():
    X_train, y_train, _ = load_split("backscatter", "representative")
    X_eval, y_eval, _ = load_split("backscatter", "full")
    clf = RPSGMMClassifier(n_components=5, n_init=1)
    grid = clf.grid_search(
        X_train, y_train, X_eval, y_eval, range(2, 7, 2), range(3, 8, 2), verbose=False
    )
    assert 0.0 <= grid.accuracy <= 1.0
    assert (clf.tau, clf.d) == (grid.tau, grid.d)
    assert len(grid.history) > 0
    assert grid.accuracy == max(row["accuracy"] for row in grid.history)


# ----------------------------------------------------------------- manifest

def test_preprocessing_regenerates_the_shipped_csvs():
    """The raw NetCDF must rebuild data/processed/ exactly.

    Skipped unless the raw archive has been fetched with
    ``python data/download_data.py``.
    """
    import pandas as pd

    netcdf = REPO_ROOT / "data" / "raw" / "all_lakes_2019.nc"
    geojson = REPO_ROOT / "data" / "raw" / "all_training.geojson"
    if not netcdf.exists():
        pytest.skip("raw NetCDF not present; run python data/download_data.py")
    pytest.importorskip("netCDF4")

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from build_processed_data import build_backscatter, build_water, load_raw

    ids, hv_lake, hv_out, s2_water, _ = load_raw(netcdf, geojson)
    built = {
        "backscatter_diff": pd.DataFrame(
            build_backscatter(hv_lake, hv_out),
            index=pd.Index(ids, name="ids"),
            columns=[f"backscatter_diff_{day}" for day in range(1, 366)],
        ),
        "water_percentage": pd.DataFrame(
            build_water(s2_water),
            index=pd.Index(ids, name="ids"),
            columns=[f"water_percentage_{day}" for day in range(1, 366)],
        ),
    }

    for filename, prefix in [
        ("backscatter_777.csv", "backscatter_diff"),
        ("backscatter_water_777.csv", "backscatter_diff"),
        ("backscatter_water_777.csv", "water_percentage"),
    ]:
        shipped = pd.read_csv(REPO_ROOT / "data" / "processed" / filename).set_index("ids")
        columns = [f"{prefix}_{day}" for day in range(START_DAY, END_DAY + 1)]
        common = built[prefix].index.intersection(shipped.index)
        assert len(common) == 777

        difference = np.abs(
            built[prefix].loc[common, columns].to_numpy(float)
            - shipped.loc[common, columns].to_numpy(float)
        )
        assert difference.max() < 1e-6, (
            f"{filename}/{prefix} differs by up to {difference.max():.2e}"
        )


def test_manifest_agrees_with_the_evaluation_set():
    import pandas as pd

    manifest = pd.read_csv(REPO_ROOT / "data" / "processed" / "lake_manifest.csv")
    assert len(manifest) == 1000
    assert int(manifest["in_visual_qc"].sum()) == 793
    assert int(manifest["in_final_777"].sum()) == 777

    # The final set must be nested inside the visual-QC set.
    assert not (manifest["in_final_777"] & ~manifest["in_visual_qc"]).any()

    _, _, ids = load_split("backscatter", "full")
    assert set(manifest.loc[manifest["in_final_777"], "ids"]) == set(ids)
