"""Train and evaluate the RPS-GMM classifier.

Reproduces the two headline results of the paper: the Sentinel-1 only model and
the combined Sentinel-1 + Sentinel-2 model, each trained on a single
representative lake per class and evaluated over all 777 labeled lakes.

Examples
--------
Full grid search::

    python scripts/run_rps_gmm.py --features backscatter
    python scripts/run_rps_gmm.py --features backscatter_water

A fast sanity check over a small grid::

    python scripts/run_rps_gmm.py --features backscatter --quick

Evaluate a known operating point without searching::

    python scripts/run_rps_gmm.py --features backscatter --tau 13 --d 15
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rpsgmm import CLASSES, RPSGMMClassifier, load_split  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results"

#: Published search space: tau in [2, 30], d in [3, 30] (paper, Section 4.2).
FULL_TAU_RANGE = range(2, 31)
FULL_D_RANGE = range(3, 31)
QUICK_TAU_RANGE = range(2, 15, 3)
QUICK_D_RANGE = range(3, 16, 3)


def display_path(path: Path) -> str:
    """Path relative to the repository root when possible, else as given."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--features",
        choices=("backscatter", "backscatter_water"),
        default="backscatter",
        help="Sentinel-1 only, or Sentinel-1 combined with Sentinel-2.",
    )
    parser.add_argument("--tau", type=int, default=None, help="Fix the time delay, skipping the search.")
    parser.add_argument("--d", type=int, default=None, help="Fix the embedding dimension.")
    parser.add_argument("--n-components", type=int, default=10, help="Gaussian mixtures per class.")
    parser.add_argument(
        "--n-init",
        type=int,
        default=1,
        help="k-means initializations per GMM; the best-scoring one is kept.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Worker processes for the grid search; -1 uses every core. "
             "Results are identical for any value.",
    )
    parser.add_argument("--quick", action="store_true", help="Search a small grid instead of the published one.")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    warnings.filterwarnings("ignore")

    X_train, y_train, _ = load_split(args.features, "representative")
    X_eval, y_eval, eval_ids = load_split(args.features, "full")

    print(f"Features        : {args.features}")
    print(f"Training set    : {X_train.shape[0]} lakes (one per class), shape {X_train.shape}")
    print(f"Evaluation set  : {X_eval.shape[0]} lakes, shape {X_eval.shape}")
    print(f"GMM             : {args.n_components} components, n_init={args.n_init}, full covariance")
    print()

    clf = RPSGMMClassifier(
        n_components=args.n_components,
        n_init=args.n_init,
        random_state=args.random_state,
    )

    started = time.time()
    grid = None
    if args.tau is not None and args.d is not None:
        clf.tau, clf.d = args.tau, args.d
        print(f"Using fixed tau={args.tau}, d={args.d} (no search).")
    else:
        tau_range = QUICK_TAU_RANGE if args.quick else FULL_TAU_RANGE
        d_range = QUICK_D_RANGE if args.quick else FULL_D_RANGE
        n_combos = len(list(tau_range)) * len(list(d_range))
        print(f"Grid search over {n_combos} (tau, d) combinations...")
        grid = clf.grid_search(
            X_train, y_train, X_eval, y_eval, tau_range, d_range,
            verbose=not args.quiet, n_jobs=args.n_jobs,
        )
        print(f"\nBest: tau={grid.tau}, d={grid.d}, accuracy={grid.accuracy * 100:.2f}%")

    clf.fit(X_train, y_train)
    predictions = clf.predict(X_eval)
    accuracy = float(np.mean(predictions == y_eval))
    elapsed = time.time() - started

    print(f"\nFinal accuracy  : {accuracy * 100:.2f}%   (tau={clf.tau}, d={clf.d})")
    print(f"Elapsed         : {elapsed:.1f}s\n")
    print(classification_report(y_eval, predictions, target_names=CLASSES, digits=4, zero_division=0))
    print("Confusion matrix (rows = true, columns = predicted):")
    header = " " * 12 + "".join(f"{name:>10s}" for name in CLASSES)
    print(header)
    for name, row in zip(CLASSES, clf.confusion_matrix(X_eval, y_eval)):
        print(f"{name:>12s}" + "".join(f"{value:10d}" for value in row))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = classification_report(
        y_eval, predictions, target_names=CLASSES, output_dict=True, zero_division=0
    )
    summary = {
        "features": args.features,
        "tau": int(clf.tau),
        "d": int(clf.d),
        "n_components": args.n_components,
        "n_init": args.n_init,
        "random_state": args.random_state,
        "n_train": int(X_train.shape[0]),
        "n_eval": int(X_eval.shape[0]),
        "accuracy": accuracy,
        "weighted_precision": report["weighted avg"]["precision"],
        "weighted_recall": report["weighted avg"]["recall"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "per_class": {name: report[name] for name in CLASSES},
        "confusion_matrix": clf.confusion_matrix(X_eval, y_eval).tolist(),
        "grid_searched": grid is not None,
        "n_evaluated": len(grid.history) if grid is not None else None,
        "elapsed_seconds": elapsed,
    }
    suffix = "_quick" if args.quick else ""
    metrics_path = args.output_dir / f"metrics_{args.features}{suffix}.json"
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {display_path(metrics_path)}")

    if grid is not None:
        grid_path = args.output_dir / f"grid_search_{args.features}{suffix}.csv"
        grid.to_frame().to_csv(grid_path, index=False)
        print(f"Wrote {display_path(grid_path)}")

    predictions_path = args.output_dir / f"predictions_{args.features}{suffix}.csv"
    import pandas as pd

    pd.DataFrame(
        {
            "ids": eval_ids,
            "true": [CLASSES[code] for code in y_eval],
            "predicted": [CLASSES[code] for code in predictions],
        }
    ).to_csv(predictions_path, index=False)
    print(f"Wrote {display_path(predictions_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
