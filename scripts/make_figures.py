"""Render the repository figures.

Produces, for each feature set:

- ``representative_trajectories_<features>.png`` -- the three training lakes
- ``phase_spaces_<features>.png`` -- their reconstructed phase spaces
- ``confusion_matrix_<features>.png`` -- evaluation over all 777 lakes
- ``grid_search_<features>.png`` -- accuracy across the (tau, d) plane, written
  only when a grid search has been run and saved to ``results/``

The classifier is fitted here, so no prior run is required. Pass ``--tau`` and
``--d`` to plot a specific embedding; otherwise the values recorded by
``scripts/run_rps_gmm.py`` are used when present.

Output goes to ``figures/``, which is not tracked. The two images embedded in
the README live in ``assets/``.

Usage
-----
    python scripts/make_figures.py
    python scripts/make_figures.py --features backscatter --tau 20 --d 11
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rpsgmm import RPSGMMClassifier, load_split  # noqa: E402
from rpsgmm.viz import plot_class_examples, plot_confusion_matrix, plot_phase_spaces  # noqa: E402

FIGURES_DIR = REPO_ROOT / "figures"
RESULTS_DIR = REPO_ROOT / "results"

TITLES = {
    "backscatter": "Sentinel-1 backscatter only",
    "backscatter_water": "Sentinel-1 backscatter + Sentinel-2 water percentage",
}

#: Embeddings behind the checked-in figures, so the script runs with no
#: arguments and without a prior grid search.
DEFAULT_EMBEDDING = {
    "backscatter": (20, 11),
    "backscatter_water": (11, 9),
}


def display_path(path: Path) -> str:
    """Path relative to the repository root when possible, else as given."""
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def resolve_embedding(features: str, tau, d, results_dir: Path) -> tuple[int, int]:
    """Choose (tau, d): explicit arguments, then a saved run, then the default."""
    if tau is not None and d is not None:
        return tau, d

    metrics_path = results_dir / f"metrics_{features}.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        return metrics["tau"], metrics["d"]

    return DEFAULT_EMBEDDING[features]


def plot_grid(grid_csv: Path, title: str, save_path: Path) -> None:
    """Heat map of grid-search accuracy over the (tau, d) plane."""
    grid = pd.read_csv(grid_csv)
    table = grid.pivot(index="d", columns="tau", values="accuracy")

    fig, ax = plt.subplots(figsize=(9, 6))
    image = ax.imshow(
        table.to_numpy() * 100,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        extent=[table.columns.min() - 0.5, table.columns.max() + 0.5,
                table.index.min() - 0.5, table.index.max() + 0.5],
    )
    best = grid.loc[grid["accuracy"].idxmax()]
    ax.scatter([best["tau"]], [best["d"]], marker="*", s=260,
               edgecolor="white", facecolor="crimson", zorder=3,
               label=f"best: $\\tau$={int(best['tau'])}, $d$={int(best['d'])}")
    ax.set_xlabel(r"Time delay $\tau$")
    ax.set_ylabel("Embedding dimension $d$")
    ax.set_title(f"Grid search accuracy - {title}")
    ax.legend(loc="upper right", framealpha=0.9)
    fig.colorbar(image, ax=ax, label="Accuracy (%)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {display_path(save_path)}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--features", nargs="*", choices=tuple(TITLES), default=list(TITLES))
    parser.add_argument("--tau", type=int, default=None, help="Time delay; overrides a saved run.")
    parser.add_argument("--d", type=int, default=None, help="Embedding dimension.")
    parser.add_argument("--n-init", type=int, default=1, help="k-means initializations per GMM.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    args = parser.parse_args(argv)

    warnings.filterwarnings("ignore")
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    for features in args.features:
        tau, d = resolve_embedding(features, args.tau, args.d, args.results_dir)
        print(f"{features}  (tau={tau}, d={d}):")

        X_train, y_train, _ = load_split(features, "representative")
        X_eval, y_eval, _ = load_split(features, "full")

        path = args.figures_dir / f"representative_trajectories_{features}.png"
        plot_class_examples(X_train, y_train, save_path=path)
        plt.close("all")
        print(f"  wrote {display_path(path)}")

        path = args.figures_dir / f"phase_spaces_{features}.png"
        plot_phase_spaces(X_train, y_train, tau, d, save_path=path)
        plt.close("all")
        print(f"  wrote {display_path(path)}")

        classifier = RPSGMMClassifier(
            tau=tau, d=d, n_components=10, n_init=args.n_init, random_state=42
        ).fit(X_train, y_train)
        matrix = classifier.confusion_matrix(X_eval, y_eval)

        path = args.figures_dir / f"confusion_matrix_{features}.png"
        plot_confusion_matrix(matrix, save_path=path)
        plt.close("all")
        print(f"  wrote {display_path(path)}")

        grid_csv = args.results_dir / f"grid_search_{features}.csv"
        if grid_csv.exists():
            plot_grid(grid_csv, TITLES[features], args.figures_dir / f"grid_search_{features}.png")
        else:
            print("  skipped grid heat map - run scripts/run_rps_gmm.py first")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
