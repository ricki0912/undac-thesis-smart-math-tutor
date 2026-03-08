from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class MetricsEvaluator:
    """
    Clase para calcular metricas estandar de clasificacion.
    """

    @staticmethod
    def compute(y_true: Any, y_pred: Any) -> dict[str, float]:
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        }

    @staticmethod
    def to_dataframe(results: list[dict[str, float | str]]) -> pd.DataFrame:
        return pd.DataFrame(results).sort_values("f1_score", ascending=False)

