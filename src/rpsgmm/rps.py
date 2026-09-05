"""Reconstructed Phase Space (RPS) embedding.

Implements the time-delay embedding of Takens' theorem (paper, Section 3.1):

.. math::

    X_n = [x_n,\\; x_{n-\\tau},\\; \\ldots,\\; x_{n-(d-1)\\tau}]

for a time delay :math:`\\tau` and an embedding dimension :math:`d`.

For multivariate inputs each feature channel is embedded independently with the
same :math:`(\\tau, d)` and the resulting delay vectors are concatenated along
the coordinate axis, giving a ``d * n_features`` dimensional phase space.
"""

from __future__ import annotations

import numpy as np

__all__ = ["embed", "phase_space_length"]


def phase_space_length(n_time_steps: int, tau: int, d: int) -> int:
    """Number of delay vectors produced by an embedding, or 0 if none fit."""
    return max(n_time_steps - (d - 1) * tau, 0)


def embed(series: np.ndarray, tau: int, d: int) -> np.ndarray:
    """Embed one or more time series into a reconstructed phase space.

    Parameters
    ----------
    series
        Either a 1-D array of shape ``(n_time_steps,)``, a 2-D array of shape
        ``(n_features, n_time_steps)`` describing a single multivariate sample,
        or a 3-D array of shape ``(n_samples, n_features, n_time_steps)``.
    tau
        Time delay, must be >= 1.
    d
        Embedding dimension, must be >= 1.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_samples * L, d * n_features)`` where ``L`` is
        :func:`phase_space_length`. Delay vectors from every sample are stacked
        row-wise, which is what :class:`sklearn.mixture.GaussianMixture`
        consumes. Returns an empty array when the embedding does not fit.
    """
    if tau < 1 or d < 1:
        raise ValueError(f"tau and d must both be >= 1, got tau={tau}, d={d}")

    series = np.asarray(series, dtype=float)
    if series.ndim == 1:
        series = series.reshape(1, 1, -1)
    elif series.ndim == 2:
        series = series.reshape(1, *series.shape)
    elif series.ndim != 3:
        raise ValueError(f"series must be 1-, 2- or 3-dimensional, got {series.ndim}-D")

    n_samples, n_features, n_time_steps = series.shape
    length = phase_space_length(n_time_steps, tau, d)
    if length == 0:
        return np.empty((0, d * n_features))

    # Build the delay vectors with a strided gather rather than a Python loop
    # over d: rows index the trajectory, columns index the delay coordinate.
    offsets = np.arange(d) * tau
    positions = offsets[None, :] + np.arange(length)[:, None]  # (L, d)

    # (n_samples, n_features, L, d) -> (n_samples, L, n_features * d)
    embedded = series[:, :, positions]
    embedded = embedded.transpose(0, 2, 1, 3).reshape(n_samples, length, n_features * d)
    return embedded.reshape(n_samples * length, n_features * d)
