"""RPS-GMM classifier for supraglacial lake time series.

One Gaussian Mixture Model is fitted per class over the Reconstructed Phase
Space of that class's representative sample (paper, Section 4.2). A test lake is
assigned to the class whose GMM gives its trajectory the highest total
log-likelihood, following the Bayesian maximum-likelihood rule of Section 3.3:

.. math::

    \\hat{a} = \\arg\\max_i \\; p(X \\mid a_i)

The embedding parameters :math:`(\\tau, d)` are chosen by grid search rather than
by false-nearest-neighbour / automutual-information estimation.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from itertools import product

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.mixture import GaussianMixture

from .rps import embed, phase_space_length

__all__ = ["RPSGMMClassifier", "GridSearchResult", "CLASSES"]

#: Canonical class order for the three-class problem. A class's integer code is
#: its index here. This must stay an ordered sequence: the argmax over per-class
#: likelihoods is an index into it, so any reordering would relabel predictions.
#: A test enforces that it is never replaced by an unordered container.
CLASSES: tuple[str, ...] = ("refreeze", "drain", "buried")


@dataclass
class GridSearchResult:
    """Outcome of a :math:`(\\tau, d)` grid search."""

    tau: int
    d: int
    accuracy: float
    history: list[dict] = field(default_factory=list)

    def to_frame(self):
        """Return the full grid as a :class:`pandas.DataFrame`."""
        import pandas as pd

        return pd.DataFrame(self.history).sort_values(
            "accuracy", ascending=False, ignore_index=True
        )


class RPSGMMClassifier:
    """Gaussian Mixture Models over Reconstructed Phase Spaces.

    Parameters
    ----------
    tau, d
        Time delay and embedding dimension. Both may be left as ``None`` and
        determined with :meth:`grid_search`.
    n_components
        Number of Gaussian mixtures per class. The paper uses 10.
    n_init
        Number of k-means initializations per GMM; the best-scoring one is
        kept. Higher values cost proportionally more time and, on this dataset,
        do not improve accuracy.
    covariance_type, max_iter, random_state
        Passed through to :class:`sklearn.mixture.GaussianMixture`.
    classes
        Class names in canonical order; the integer code of a class is its
        index in this sequence.
    """

    def __init__(
        self,
        tau: int | None = None,
        d: int | None = None,
        *,
        n_components: int = 10,
        n_init: int = 1,
        covariance_type: str = "full",
        max_iter: int = 1000,
        random_state: int | None = 42,
        classes: tuple[str, ...] = CLASSES,
    ) -> None:
        self.tau = tau
        self.d = d
        self.n_components = n_components
        self.n_init = n_init
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.random_state = random_state
        self.classes = tuple(classes)
        self.gmms_: dict[str, GaussianMixture] = {}

    # ------------------------------------------------------------------ utils

    def _make_gmm(self, n_components: int) -> GaussianMixture:
        return GaussianMixture(
            n_components=n_components,
            n_init=self.n_init,
            covariance_type=self.covariance_type,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )

    def _class_embeddings(self, X: np.ndarray, y: np.ndarray, tau: int, d: int) -> dict:
        """Embed the training trajectories of every class."""
        return {
            name: embed(X[y == code], tau, d)
            for code, name in enumerate(self.classes)
        }

    def _fit_gmms(self, embeddings: dict, n_components: int) -> dict:
        return {
            name: self._make_gmm(n_components).fit(rps)
            for name, rps in embeddings.items()
        }

    # -------------------------------------------------------------------- api

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RPSGMMClassifier":
        """Fit one GMM per class.

        Parameters
        ----------
        X
            Training trajectories, shape ``(n_samples, n_features, n_time_steps)``.
        y
            Integer class codes indexing :attr:`classes`.
        """
        if self.tau is None or self.d is None:
            raise ValueError("tau and d must be set; call grid_search() first.")

        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        embeddings = self._class_embeddings(X, y, self.tau, self.d)
        missing = [name for name, rps in embeddings.items() if len(rps) == 0]
        if missing:
            raise ValueError(f"No training trajectories embed for class(es): {missing}")

        self.gmms_ = self._fit_gmms(embeddings, self.n_components)
        self.train_embeddings_ = embeddings
        return self

    def select_n_components(self, X, y, n_component_range) -> dict[str, int]:
        """Pick each class's mixture count by training log-likelihood.

        For each class the candidate with the highest average log-likelihood
        on its own training embedding is kept. Note that this selects on
        training data, so it tends to favor the largest candidate; use an
        information criterion such as BIC if you need genuine selection.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        embeddings = self._class_embeddings(X, y, self.tau, self.d)

        chosen: dict[str, int] = {}
        for name, rps in embeddings.items():
            best_score, best_k, best_gmm = -np.inf, None, None
            for k in n_component_range:
                gmm = self._make_gmm(k).fit(rps)
                score = gmm.score(rps)
                if score > best_score:
                    best_score, best_k, best_gmm = score, k, gmm
            chosen[name] = best_k
            self.gmms_[name] = best_gmm

        self.train_embeddings_ = embeddings
        self.best_n_components_ = chosen
        return chosen

    def log_likelihoods(self, X: np.ndarray) -> np.ndarray:
        """Per-class mean log-likelihood for each sample, shape ``(n, n_classes)``.

        Every sample is embedded once and all trajectories are scored in a
        single call per class, then averaged per sample. This is numerically
        identical to scoring each lake separately, since ``GaussianMixture.score``
        is the mean of ``score_samples``, but it avoids one Python-level call
        per lake per class, which dominates the grid search.
        """
        if not self.gmms_:
            raise ValueError("Model is not fitted; call fit() first.")

        X = np.asarray(X, dtype=float)
        if X.ndim == 2:
            X = X[None, ...]

        n_samples = len(X)
        length = phase_space_length(X.shape[-1], self.tau, self.d)
        if length == 0:
            return np.full((n_samples, len(self.classes)), -np.inf)

        # embed() stacks samples row-major with `length` rows each, so the flat
        # score vector reshapes straight back to (n_samples, length).
        rps = embed(X, self.tau, self.d)
        scores = np.empty((n_samples, len(self.classes)))
        for j, name in enumerate(self.classes):
            per_vector = self.gmms_[name].score_samples(rps)
            scores[:, j] = per_vector.reshape(n_samples, length).mean(axis=1)
        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign each sample the class with the highest likelihood."""
        return np.argmax(self.log_likelihoods(X), axis=1)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Classification accuracy."""
        return accuracy_score(np.asarray(y), self.predict(X))

    def confusion_matrix(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Confusion matrix over the canonical class order."""
        return confusion_matrix(
            np.asarray(y), self.predict(X), labels=np.arange(len(self.classes))
        )

    # ------------------------------------------------------------ grid search

    def _evaluate_combination(self, X_train, y_train, X_eval, y_eval, tau, d):
        """Fit and score one :math:`(\\tau, d)`. Returns ``None`` if it does not fit.

        Self-contained and free of side effects on ``self``, so it can run in a
        worker process. Every fit is seeded by ``random_state``, so the result
        does not depend on evaluation order or worker count.
        """
        if phase_space_length(X_train.shape[-1], tau, d) < self.n_components:
            return None

        embeddings = {
            name: embed(X_train[y_train == code], tau, d)
            for code, name in enumerate(self.classes)
        }
        if any(len(rps) < self.n_components for rps in embeddings.values()):
            return None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gmms = {
                name: self._make_gmm(self.n_components).fit(rps)
                for name, rps in embeddings.items()
            }

        length = phase_space_length(X_eval.shape[-1], tau, d)
        if length == 0:
            return None
        rps_eval = embed(X_eval, tau, d)
        scores = np.empty((len(X_eval), len(self.classes)))
        for j, name in enumerate(self.classes):
            scores[:, j] = gmms[name].score_samples(rps_eval).reshape(len(X_eval), length).mean(axis=1)

        return {"tau": tau, "d": d, "accuracy": float(np.mean(np.argmax(scores, axis=1) == y_eval))}

    def grid_search(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_eval: np.ndarray,
        y_eval: np.ndarray,
        tau_range,
        d_range,
        *,
        verbose: bool = True,
        n_jobs: int = 1,
    ) -> GridSearchResult:
        """Search :math:`(\\tau, d)` for the highest accuracy on ``X_eval``.

        This follows Algorithm 1 of the paper, which selects the embedding
        parameters on the evaluation set. Hold out a separate selection set if
        you need a generalization estimate.

        Parameters
        ----------
        n_jobs
            Worker processes for the search. Each combination is independent and
            individually seeded, so any ``n_jobs`` yields identical results;
            ``-1`` uses every core. Combinations whose embedding does not fit the
            observation window are skipped, which is why the evaluated count is
            below ``len(tau_range) * len(d_range)``.
        """
        X_train = np.asarray(X_train, dtype=float)
        X_eval = np.asarray(X_eval, dtype=float)
        y_train = np.asarray(y_train)
        y_eval = np.asarray(y_eval)

        combinations = list(product(tau_range, d_range))

        if n_jobs == 1:
            results = []
            for index, (tau, d) in enumerate(combinations, start=1):
                outcome = self._evaluate_combination(X_train, y_train, X_eval, y_eval, tau, d)
                results.append(outcome)
                if verbose and outcome is not None:
                    best_so_far = max(
                        (r["accuracy"] for r in results if r is not None), default=-np.inf
                    )
                    if outcome["accuracy"] >= best_so_far:
                        print(
                            f"  [{index:4d}/{len(combinations)}] "
                            f"tau={tau:2d} d={d:2d} -> {outcome['accuracy']:.4f} *"
                        )
        else:
            from joblib import Parallel, delayed

            if verbose:
                print(f"  running {len(combinations)} combinations across n_jobs={n_jobs} ...")
            results = Parallel(n_jobs=n_jobs, prefer="processes")(
                delayed(self._evaluate_combination)(X_train, y_train, X_eval, y_eval, tau, d)
                for tau, d in combinations
            )

        history = [outcome for outcome in results if outcome is not None]
        if not history:
            raise ValueError("No (tau, d) combination produced a usable embedding.")

        # Ties resolve to the first combination in grid order, matching the
        # serial scan regardless of the order workers finished in.
        best_entry = max(history, key=lambda r: (r["accuracy"], -combinations.index((r["tau"], r["d"]))))
        best = GridSearchResult(tau=best_entry["tau"], d=best_entry["d"], accuracy=best_entry["accuracy"])

        if verbose:
            skipped = len(combinations) - len(history)
            print(
                f"  evaluated {len(history)} of {len(combinations)} combinations "
                f"({skipped} skipped: embedding longer than the observation window)"
            )

        best.history = history
        self.tau, self.d = best.tau, best.d
        return best
