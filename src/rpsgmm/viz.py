"""Plotting helpers for the RPS-GMM experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .data import END_DAY, START_DAY
from .model import CLASSES
from .rps import embed

__all__ = [
    "plot_class_examples",
    "plot_phase_spaces",
    "plot_confusion_matrix",
    "month_ticks",
]

_CHANNEL_LABELS = ("Backscatter difference $HV_{anom}$ (dB)", "Water percentage $p_{water}$ (%)")
_CLASS_COLORS = {"refreeze": "#1f77b4", "drain": "#d62728", "buried": "#2ca02c"}
_MONTH_STARTS = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
_MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def month_ticks(ax, start_day: int = START_DAY, end_day: int = END_DAY) -> None:
    """Label the x-axis with month names for a day-of-year window."""
    positions, labels = [], []
    for day, name in zip(_MONTH_STARTS, _MONTH_NAMES):
        if start_day <= day <= end_day:
            positions.append(day - start_day)
            labels.append(name)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)


def plot_class_examples(X, y, *, classes=CLASSES, save_path: Path | None = None):
    """Plot the representative trajectory of each class, one panel per channel."""
    X = np.asarray(X, dtype=float)
    n_channels = X.shape[1]
    fig, axes = plt.subplots(n_channels, 1, figsize=(9, 3.2 * n_channels), sharex=True, squeeze=False)

    for channel in range(n_channels):
        ax = axes[channel][0]
        for code, name in enumerate(classes):
            for series in X[np.asarray(y) == code, channel, :]:
                ax.plot(series, linewidth=1.6, label=name.capitalize(), color=_CLASS_COLORS.get(name))
        ax.set_ylabel(_CHANNEL_LABELS[channel] if channel < len(_CHANNEL_LABELS) else f"Channel {channel}")
        ax.grid(alpha=0.3)
        if channel == 0:
            ax.legend(frameon=False, ncol=len(classes))
    month_ticks(axes[-1][0])
    axes[-1][0].set_xlabel("Time (2019 melt season)")
    fig.suptitle("Representative supraglacial lake trajectories", y=0.98)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_phase_spaces(X, y, tau: int, d: int, *, classes=CLASSES, save_path: Path | None = None):
    """Plot the first three RPS coordinates of each class's trajectory."""
    X = np.asarray(X, dtype=float)
    if d < 3:
        raise ValueError(f"Need d >= 3 to draw a 3-D phase space, got d={d}")

    fig = plt.figure(figsize=(4.2 * len(classes), 4.2))
    for index, name in enumerate(classes):
        rps = embed(X[np.asarray(y) == index], tau, d)
        ax = fig.add_subplot(1, len(classes), index + 1, projection="3d")
        ax.plot(rps[:, 0], rps[:, 1], rps[:, 2], linewidth=0.9, color=_CLASS_COLORS.get(name))
        ax.set_title(name.capitalize())
        ax.set_xlabel("$x_t$")
        ax.set_ylabel(r"$x_{t+\tau}$")
        ax.set_zlabel(r"$x_{t+2\tau}$")
    fig.suptitle(rf"Reconstructed phase spaces ($\tau$={tau}, $d$={d})", y=1.0)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_confusion_matrix(matrix, accuracy=None, *, classes=CLASSES, save_path: Path | None = None):
    """Render a confusion matrix with per-cell counts."""
    matrix = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    image = ax.imshow(matrix, cmap="Blues")

    threshold = matrix.max() / 2 if matrix.max() else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j, i, f"{matrix[i, j]:d}",
                ha="center", va="center",
                color="white" if matrix[i, j] > threshold else "black",
            )

    labels = [name.capitalize() for name in classes]
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion matrix" + (f"  (accuracy {accuracy * 100:.2f}%)" if accuracy else ""))
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
