"""Figures for the FD02 report. Each title names the dataset, split and model."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import average_precision_score, precision_recall_curve  # noqa: E402


def save(fig, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_pr_curves(y_true, score_by_model: dict[str, np.ndarray], title: str, path: str | Path) -> Path:
    y = np.asarray(y_true)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for label, scores in score_by_model.items():
        precision, recall, _ = precision_recall_curve(y, scores)
        ap = average_precision_score(y, scores)
        ax.plot(recall, precision, label=f"{label} (PR-AUC {ap:.3f})")
    ax.axhline(y.mean(), ls="--", color="grey", label="No skill (fraud prevalence)")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    return save(fig, path)


def plot_calibration(y_true, probs, title: str, path: str | Path, bins: int = 10) -> Path:
    frac_pos, mean_pred = calibration_curve(y_true, probs, n_bins=bins, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfect calibration")
    ax.plot(mean_pred, frac_pos, marker="o", label="Model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraud rate")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    return save(fig, path)


def plot_gain_importance(table, title: str, path: str | Path, top: int = 15) -> Path:
    top_table = table.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(top_table["feature"], top_table["gain"], color="tab:blue")
    ax.set_xlabel("Total gain")
    ax.set_title(title)
    ax.grid(alpha=0.3, axis="x")
    return save(fig, path)
