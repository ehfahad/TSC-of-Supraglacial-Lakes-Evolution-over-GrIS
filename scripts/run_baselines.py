"""Reproduce the ML/DL baselines of Table 1.

Five established time series classifiers from ``sktime`` are evaluated with
5-fold cross-validation over all 777 labeled lakes, following Section 5.2 of the
paper: each fold trains on roughly 622 lakes and predicts the held-out fifth,
and accuracy is computed over the pooled out-of-fold predictions.

The combined setting concatenates the backscatter and water channels into a
single 488-step univariate series rather than treating them as two channels.
That reproduces the published configuration.

Requires the optional ``baselines`` extra (``sktime``, ``tensorflow`` and
``dtaidistance``), which the core RPS-GMM pipeline does not need. Install it in
a separate environment; see the README section "Reproducing the results".

Examples
--------
Estimate the runtime before committing to a full sweep::

    python scripts/run_baselines.py --estimate

Reproduce Table 1::

    python scripts/run_baselines.py --features backscatter
    python scripts/run_baselines.py --features backscatter_water

Run a single model, or a shortened sweep::

    python scripts/run_baselines.py --models knn
    python scripts/run_baselines.py --n-epochs 10 --n-splits 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rpsgmm import CLASSES, load_split  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results"

# Published protocol (paper, Section 5.2).
RANDOM_STATE = 42
BATCH_SIZE = 128
N_EPOCHS = 100
N_SPLITS = 5

#: Epochs used by --estimate before extrapolating to the full protocol.
PROBE_EPOCHS = 3

#: CLI name -> sktime class name, in the order Table 1 lists them.
MODELS = {
    "lstmfcn": "LSTMFCNClassifier",
    "fcn": "FCNClassifier",
    "resnet": "ResNetClassifier",
    "rnn": "SimpleRNNClassifier",
    "knn": "KNeighborsTimeSeriesClassifier",
}
DEEP_MODELS = ("lstmfcn", "fcn", "resnet", "rnn")



class DtwOneNearestNeighbor:
    """1-NN with an unconstrained DTW distance, via dtaidistance's C backend.

    Equivalent to sktime's ``KNeighborsTimeSeriesClassifier`` defaults used in
    the paper (``n_neighbors=1``, ``distance="dtw"``, ``algorithm="brute"``),
    but roughly three orders of magnitude faster.

    The two libraries report the same distance under a monotonic transform:
    sktime returns the squared DTW cost, dtaidistance its square root. Verified
    on this dataset to ``max |sktime - dtaidistance^2| = 3.6e-15`` with
    identical nearest neighbors, so 1-NN predictions are unchanged. Ranking is
    all a nearest-neighbor rule depends on.

    Series must be equal length, which holds here (every lake is daily over the
    same melt-season window).
    """

    def __init__(self) -> None:
        self._train = None
        self._labels = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DtwOneNearestNeighbor":
        self._train = np.ascontiguousarray(X, dtype=float)
        self._labels = np.asarray(y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        from dtaidistance import dtw

        query = np.ascontiguousarray(X, dtype=float)
        n_query, n_train = len(query), len(self._train)

        # distance_matrix_fast fills one rectangular block of a combined matrix,
        # computing only the query-to-train pairs and parallelizing in C.
        combined = np.ascontiguousarray(np.vstack([query, self._train]))
        matrix = dtw.distance_matrix_fast(
            combined,
            block=((0, n_query), (n_query, n_query + n_train)),
            compact=False,
        )
        cross = matrix[:n_query, n_query:]
        return self._labels[np.argmin(cross, axis=1)]


def display_path(path: Path) -> str:
    """Path relative to the repository root when possible, else as given."""
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def build_model(name: str, n_epochs: int = N_EPOCHS, n_jobs: int = -1):
    """Instantiate one baseline with the published hyperparameters.

    The ``knn`` entry returns sktime's own estimator, kept as a reference
    implementation. The sweep itself uses :class:`DtwOneNearestNeighbor`, which
    computes the same distance far more quickly; ``n_jobs`` applies only to this
    reference version.
    """
    if name == "knn":
        from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier

        return KNeighborsTimeSeriesClassifier(n_jobs=n_jobs)

    from sktime.classification import deep_learning

    return getattr(deep_learning, MODELS[name])(
        n_epochs=n_epochs,
        batch_size=BATCH_SIZE,
        random_state=RANDOM_STATE,
        verbose=False,
    )


def load_panel(features: str, *, normalize: bool = True):
    """Load the 777-lake evaluation set in sktime's nested panel format.

    Labels are returned as integer codes indexing :data:`rpsgmm.CLASSES`, not as
    class-name strings. sktime's ``fit_predict(..., cv=...)`` pools fold
    predictions with ``-np.ones(shape, dtype=y.dtype)``, which raises
    ``UFuncTypeError`` on a string dtype because NumPy has no ``negative`` for
    strings. Integer codes sidestep that; names are restored for reporting.
    """
    from sktime.datatypes._panel._convert import from_2d_array_to_nested

    X, y, _ = load_split(features, "full")
    # (n, channels, time) -> (n, channels * time): the published baselines treat
    # the concatenated channels as one long univariate series.
    flat = X.reshape(len(X), -1)
    if normalize:
        flat = MinMaxScaler().fit_transform(flat)
    return from_2d_array_to_nested(flat), np.ascontiguousarray(flat), np.asarray(y, dtype=int)


def estimate_runtime(features_list, model_names, n_epochs, n_splits, n_jobs=-1) -> int:
    """Time a short run of each model and extrapolate the full sweep."""
    print(f"Timing {PROBE_EPOCHS} epochs on one fold per model, then extrapolating")
    print(f"to {n_epochs} epochs x {n_splits} folds.\n")

    total = 0.0
    for features in features_list:
        X, X_flat, y = load_panel(features)
        # One fold's worth of training data.
        indices = np.arange(len(y))
        fold_idx, _ = train_test_split(
            indices, test_size=1 / n_splits, random_state=RANDOM_STATE, stratify=y
        )
        X_fold, y_fold = X.iloc[fold_idx], y[fold_idx]
        print(f"{features}: {X_flat.shape[1]} time steps, {len(fold_idx)} lakes per fold")

        for name in model_names:
            label = MODELS[name]
            try:
                if name == "knn":
                    # Times the same implementation run_sweep uses. There is no
                    # training loop to shorten, so a small batch of predictions
                    # is timed and scaled by the pair count, which dominates.
                    n_probe = 20
                    started = time.time()
                    model = DtwOneNearestNeighbor().fit(X_flat[fold_idx], y_fold)
                    model.predict(X_flat[:n_probe])
                    projected = (time.time() - started) / n_probe * len(y)
                else:
                    import tensorflow as tf

                    tf.keras.backend.clear_session()
                    started = time.time()
                    build_model(name, n_epochs=PROBE_EPOCHS).fit(X_fold, y_fold)
                    per_epoch = (time.time() - started) / PROBE_EPOCHS
                    projected = per_epoch * n_epochs * n_splits
            except Exception as error:
                print(f"  {label:32s} FAILED: {type(error).__name__}: {error}")
                continue

            total += projected
            print(f"  {label:32s} ~{projected / 60:6.1f} min")
        print()

    print(f"Estimated total: {total / 3600:.1f} hours "
          f"({len(features_list)} feature set(s), {len(model_names)} model(s))")
    return 0


def run_sweep(features, model_names, n_epochs, n_splits, normalize, output_dir, n_jobs=-1) -> int:
    """Cross-validate each baseline and write the results."""
    X, X_flat, y = load_panel(features, normalize=normalize)

    print(f"Features   : {features}")
    print(f"Panel      : {len(X)} lakes, {len(X.iloc[0, 0])} time steps")
    print(f"Protocol   : {n_splits}-fold CV, {n_epochs} epochs, pooled out-of-fold predictions")
    print(f"Models     : {', '.join(MODELS[n] for n in model_names)}\n")

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"baselines_{features}.csv"
    json_path = output_dir / f"baselines_{features}.json"

    rows = []
    for name in model_names:
        label = MODELS[name]
        print(f"--- {label} ---", flush=True)
        started = time.time()
        try:
            if name == "knn":
                # Hand-rolled CV: the fast 1-NN works on plain arrays rather
                # than sktime's panel format.
                predictions = np.empty(len(y), dtype=int)
                for fold, (train_idx, test_idx) in enumerate(cv.split(X_flat), start=1):
                    model = DtwOneNearestNeighbor().fit(X_flat[train_idx], y[train_idx])
                    predictions[test_idx] = model.predict(X_flat[test_idx])
                    print(f"  fold {fold}/{n_splits} done", flush=True)
            else:
                predictions = build_model(name, n_epochs=n_epochs, n_jobs=n_jobs).fit_predict(X=X, y=y, cv=cv)
        except Exception as error:
            # One failing baseline should not discard the rest of the sweep.
            print(f"  FAILED: {type(error).__name__}: {error}\n", flush=True)
            rows.append({"model": label, "accuracy": None, "error": str(error)})
            continue

        elapsed = time.time() - started
        predictions = np.asarray(predictions, dtype=int)
        if (predictions < 0).any():
            # -1 is sktime's fill value for lakes no fold predicted.
            raise RuntimeError(f"{int((predictions < 0).sum())} lakes received no prediction")

        accuracy = accuracy_score(y, predictions)
        labels = list(range(len(CLASSES)))
        report = classification_report(
            y, predictions, labels=labels, target_names=list(CLASSES),
            output_dict=True, zero_division=0,
        )
        print(f"  accuracy {accuracy * 100:.2f}%   ({elapsed / 60:.1f} min)")
        print(classification_report(
            y, predictions, labels=labels, target_names=list(CLASSES),
            digits=4, zero_division=0,
        ), flush=True)

        rows.append({
            "model": label,
            "accuracy": accuracy,
            "weighted_precision": report["weighted avg"]["precision"],
            "weighted_recall": report["weighted avg"]["recall"],
            "weighted_f1": report["weighted avg"]["f1-score"],
            "n_epochs": None if name == "knn" else n_epochs,
            "n_splits": n_splits,
            "elapsed_seconds": elapsed,
        })

        # Save after every model so a long sweep survives an interruption.
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print("\nSummary")
    summary = pd.DataFrame(rows)[["model", "accuracy"]]
    print(summary.to_string(index=False, na_rep="failed"))
    print(f"\nWrote {display_path(csv_path)}")
    print(f"Wrote {display_path(json_path)}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--features",
        choices=("backscatter", "backscatter_water"),
        default="backscatter",
        help="Sentinel-1 only, or Sentinel-1 combined with Sentinel-2.",
    )
    parser.add_argument(
        "--models", nargs="*", choices=tuple(MODELS), default=list(MODELS),
        help="Subset of baselines to run.",
    )
    parser.add_argument("--n-epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--n-splits", type=int, default=N_SPLITS)
    parser.add_argument(
        "--n-jobs", type=int, default=-1,
        help="Cores for the KNN distance computations; -1 uses every core. "
             "Does not affect predictions.",
    )
    parser.add_argument("--no-normalize", action="store_true", help="Skip min-max scaling.")
    parser.add_argument(
        "--estimate", action="store_true",
        help="Time a short run and project the full sweep, without running it.",
    )
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)

    warnings.filterwarnings("ignore")

    try:
        import sktime  # noqa: F401
    except ImportError:
        raise SystemExit(
            "sktime is required for the baselines. Install it with:\n"
            "    pip install -e '.[baselines]'\n"
            "See the README section 'Reproducing the baselines'."
        ) from None

    if args.estimate:
        # Estimating covers both feature sets, since that is the full job.
        return estimate_runtime(
            ("backscatter", "backscatter_water"), args.models,
            args.n_epochs, args.n_splits, args.n_jobs,
        )

    return run_sweep(
        args.features, args.models, args.n_epochs, args.n_splits,
        not args.no_normalize, args.output_dir, args.n_jobs,
    )


if __name__ == "__main__":
    raise SystemExit(main())
