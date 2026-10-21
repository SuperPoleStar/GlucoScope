"""Metrics for GlucoScope binary, multiclass, and regression tasks."""

from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def safe_auc(y_true, probability, multi_class=False):
    try:
        if multi_class:
            return float(
                roc_auc_score(y_true, probability, multi_class="ovr", average="macro")
            )
        return float(roc_auc_score(y_true, probability))
    except ValueError:
        return float("nan")


def compute_metrics(task_type, y_true, raw_prediction):
    y_true = np.asarray(y_true)
    raw = np.asarray(raw_prediction)
    if task_type == "regression":
        mse = mean_squared_error(y_true, raw)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pearson = pearsonr(y_true, raw)[0] if len(y_true) > 1 else np.nan
            spearman = spearmanr(y_true, raw)[0] if len(y_true) > 1 else np.nan
        return {
            "mae": float(mean_absolute_error(y_true, raw)),
            "mse": float(mse),
            "rmse": float(np.sqrt(mse)),
            "r2": float(r2_score(y_true, raw)),
            "pearson": float(pearson),
            "spearman": float(spearman),
        }
    if task_type in {"binary", "classification"}:
        if raw.ndim != 2 or raw.shape[1] != 2:
            raise ValueError("Binary classification expects logits with shape (N,2)")
        shifted = raw - raw.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        probability = probabilities[:, 1]
        predicted = probabilities.argmax(axis=1)
        matrix = confusion_matrix(y_true, predicted, labels=[0, 1])
        tn, fp, _, _ = matrix.ravel()
        specificity = tn / max(1, tn + fp)
        return {
            "accuracy": float(accuracy_score(y_true, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
            "f1": float(f1_score(y_true, predicted, zero_division=0)),
            "f1_macro": float(f1_score(y_true, predicted, average="macro", zero_division=0)),
            "precision": float(precision_score(y_true, predicted, zero_division=0)),
            "precision_macro": float(precision_score(y_true, predicted, average="macro", zero_division=0)),
            "recall": float(recall_score(y_true, predicted, zero_division=0)),
            "recall_macro": float(recall_score(y_true, predicted, average="macro", zero_division=0)),
            "specificity": float(specificity),
            "roc_auc": safe_auc(y_true, probability),
            "average_precision": float(average_precision_score(y_true, probability)),
        }
    shifted = raw - raw.max(axis=1, keepdims=True)
    probability = np.exp(shifted)
    probability /= probability.sum(axis=1, keepdims=True)
    predicted = probability.argmax(axis=1)
    matrix = confusion_matrix(y_true, predicted)
    specificities = []
    for index in range(len(matrix)):
        true_negative = matrix.sum() - matrix[index, :].sum() - matrix[:, index].sum() + matrix[index, index]
        false_positive = matrix[:, index].sum() - matrix[index, index]
        specificities.append(true_negative / max(1, true_negative + false_positive))
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "specificity_macro": float(np.mean(specificities)),
        "f1_macro": float(
            f1_score(y_true, predicted, average="macro", zero_division=0)
        ),
        "precision_macro": float(
            precision_score(y_true, predicted, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, predicted, average="macro", zero_division=0)
        ),
        "roc_auc_ovr_macro": safe_auc(y_true, probability, multi_class=True),
        "average_precision_macro": float(
            average_precision_score(
                np.eye(raw.shape[1])[y_true.astype(int)],
                probability,
                average="macro",
            )
        ),
    }


def selection_value(task_type, metrics, loss):
    if task_type == "regression":
        return -float(metrics["mae"])
    if task_type == "multiclass":
        return float(metrics["f1_macro"])
    if task_type in {"binary", "classification"}:
        return float(metrics["f1_macro"])
    return -float(loss)
