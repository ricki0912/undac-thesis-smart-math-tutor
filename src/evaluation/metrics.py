from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


class MetricsEvaluator:
    """
    Clase para calcular metricas estandar de clasificacion.
    """

    @staticmethod
    def compute(
        y_true: Any,
        y_pred: Any,
        y_score: Any | None = None,
    ) -> dict[str, float]:
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
            "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            # Alias para compatibilidad con reportes anteriores.
            "f1_score": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "auc_ovr_weighted": 0.0,
            "auc_ovr_macro": 0.0,
            # Alias para compatibilidad con reportes anteriores.
            "auc": 0.0,
        }
        auc_ovr_weighted, auc_ovr_macro = MetricsEvaluator._safe_multiclass_auc(y_true, y_score)
        metrics["auc_ovr_weighted"] = auc_ovr_weighted
        metrics["auc_ovr_macro"] = auc_ovr_macro
        metrics["auc"] = auc_ovr_weighted
        return metrics

    @staticmethod
    def to_dataframe(results: list[dict[str, float | str]]) -> pd.DataFrame:
        return pd.DataFrame(results).sort_values("f1_weighted", ascending=False)

    @staticmethod
    def _safe_multiclass_auc(y_true: Any, y_score: Any | None) -> tuple[float, float]:
        if y_score is None:
            return 0.0, 0.0
        try:
            auc_weighted = float(
                roc_auc_score(y_true, y_score, multi_class="ovr", average="weighted")
            )
            auc_macro = float(
                roc_auc_score(y_true, y_score, multi_class="ovr", average="macro")
            )
            return auc_weighted, auc_macro
        except Exception:
            return 0.0, 0.0

