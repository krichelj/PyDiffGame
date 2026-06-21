"""Small, dependency-light plotting helpers used by the solvers.

Keeping plotting in its own module means the numerical core has no hard
dependency on a particular figure layout, and makes the behaviour easy to test
(or to swap for a non-interactive backend in headless environments).
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from PyDiffGame._typing import FloatArray, PathInput


def plot_temporal(
    time: FloatArray,
    values: FloatArray,
    *,
    labels: Sequence[str] | None = None,
    title: str | None = None,
    show_legend: bool = True,
    y_label: str | None = None,
) -> Figure:
    """Plot one or more time series and return the :class:`~matplotlib.figure.Figure`.

    Parameters
    ----------
    time:
        1-D array of sample times.
    values:
        2-D array of shape ``(len(time), k)`` (or ``(len(time),)``).
    labels:
        Optional per-series legend labels.
    title, y_label:
        Optional figure title and y-axis label.
    show_legend:
        Whether to draw a legend (only if ``labels`` are supplied).
    """

    values = np.asarray(values)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    ax.plot(time[: values.shape[0]], values)
    ax.set_xlabel("Time $[s]$", fontsize=14)
    if y_label:
        ax.set_ylabel(y_label, fontsize=14)
    if title:
        ax.set_title(title)

    if values.size and np.nanmax(np.abs(values)) > 1e3:
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    if show_legend and labels is not None:
        ax.legend(
            labels,
            loc="upper right",
            ncol=2 if len(labels) <= 8 else 4,
            framealpha=0.4,
        )

    ax.grid(True)
    fig.tight_layout()
    return fig


def save_figure(fig: Figure, figure_path: PathInput, figure_filename: str) -> Path:
    """Save ``fig`` under ``figure_path``, avoiding clobbering existing files.

    A numeric suffix is appended until an unused filename is found, mirroring the
    original package behaviour but without its off-by-one filename bug.
    """

    directory = Path(figure_path)
    directory.mkdir(parents=True, exist_ok=True)

    candidate = directory / f"{figure_filename}.png"
    index = 0
    while candidate.is_file():
        index += 1
        candidate = directory / f"{figure_filename}_{index}.png"

    fig.savefig(candidate, bbox_inches="tight")
    return candidate


def show() -> None:
    """Show all open figures unless running headless (``MPLBACKEND=Agg``)."""

    if os.environ.get("MPLBACKEND", "").lower() != "agg":
        plt.show()


__all__ = ["plot_temporal", "save_figure", "show"]
