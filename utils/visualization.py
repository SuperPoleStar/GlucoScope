"""Headless visualization helpers for pretraining and downstream evaluation."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def plot_history(history, path, train_key="train_loss", val_key="validation_loss"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(epochs, [row[train_key] for row in history], label="train")
    axis.plot(
        epochs,
        [row[val_key] for row in history],
        label=val_key.removesuffix("_loss"),
    )
    axis.set(xlabel="Epoch", ylabel="Loss", title="GlucoScope loss")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_downstream(task_type, y_true, raw, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(5.2, 4.6))
    if task_type == "regression":
        axis.scatter(y_true, raw, alpha=0.65, edgecolor="none")
        lower = min(np.min(y_true), np.min(raw))
        upper = max(np.max(y_true), np.max(raw))
        axis.plot([lower, upper], [lower, upper], "--", color="black")
        axis.set(xlabel="Observed", ylabel="Predicted", title="GlucoScope regression")
    else:
        predicted = np.asarray(raw).argmax(axis=1)
        matrix = confusion_matrix(y_true, predicted)
        ConfusionMatrixDisplay(matrix).plot(ax=axis, colorbar=False)
        axis.set_title("GlucoScope confusion matrix")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_attention(ids, weights, path, max_samples=100):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = min(int(max_samples), len(weights))
    figure, axis = plt.subplots(figsize=(11, max(3, count * 0.08)))
    image = axis.imshow(np.asarray(weights)[:count], aspect="auto", cmap="viridis")
    axis.set(xlabel="Phase index (15-min bins)", ylabel="Sample")
    axis.set_title("GlucoScope phase-attention weights")
    figure.colorbar(image, ax=axis, fraction=0.02)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
