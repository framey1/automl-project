from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import FIGURES_DIR, REPORTS_DIR


def calculate_metrics(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }


def save_metrics(metrics: dict, output_path: Path = REPORTS_DIR / "metrics.json") -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")


def save_confusion_matrix(model, X_test, y_test, output_path: Path = FIGURES_DIR / "confusion_matrix.png") -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax)
    ax.set_title("Confusion matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_roc_curve(model, X_test, y_test, output_path: Path = FIGURES_DIR / "roc_curve.png") -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
    ax.set_title("ROC curve")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_class_distribution(df: pd.DataFrame, target_col: str, output_path: Path = FIGURES_DIR / "class_distribution.png") -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    df[target_col].value_counts().sort_index().plot(kind="bar", ax=ax)
    ax.set_title("Target class distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_feature_importance(model, feature_names, output_path: Path = FIGURES_DIR / "feature_importance.png") -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    estimator = model.named_steps.get("model", model)
    values = None
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = abs(estimator.coef_[0])
    if values is None:
        return
    importance = pd.Series(values, index=feature_names).sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(7, 5))
    importance.sort_values().plot(kind="barh", ax=ax)
    ax.set_title("Top-10 feature importance")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
