from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics import MetricsEvaluator
from src.utils.config import ProjectConfig


class ModelTrainer:
    """
    Clase encargada de entrenar y comparar multiples modelos.
    """

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self.evaluator = MetricsEvaluator()

    def train(
        self,
        x: pd.DataFrame,
        y: pd.Series,
        split_series: pd.Series | None = None,
        time_series: pd.Series | None = None,
    ) -> dict[str, Any]:
        x_train, x_test, y_train, y_test, validation_type = self._split_data(
            x, y, split_series, time_series
        )
        models = self._build_models()

        results: list[dict[str, float | str]] = []
        evaluation_payloads: list[dict[str, Any]] = []
        best_name = ""
        best_model: Pipeline | None = None
        best_score = -np.inf

        for model_name, model_pipeline in models.items():
            model = clone(model_pipeline)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            y_score = self._extract_scores(model, x_test)
            metrics = self.evaluator.compute(y_test, y_pred, y_score)
            metrics["model_name"] = model_name
            results.append(metrics)

            evaluation_payloads.append(
                {
                    "model_name": model_name,
                    "y_true": y_test.copy(),
                    "y_pred": pd.Series(y_pred, index=y_test.index),
                    "y_score": y_score,
                    "labels": sorted(pd.unique(y)),
                }
            )

            if metrics["f1_weighted"] > best_score:
                best_score = metrics["f1_weighted"]
                best_name = model_name
                best_model = model

        if best_model is None:
            raise RuntimeError("No se pudo entrenar ningun modelo.")

        leaderboard = self.evaluator.to_dataframe(results)
        self._save_artifacts(best_model, leaderboard)
        importances = self._extract_importance(best_model, list(x.columns))

        return {
            "best_model_name": best_name,
            "best_model": best_model,
            "leaderboard": leaderboard,
            "feature_importance": importances,
            "validation_type": validation_type,
            "evaluation_payloads": evaluation_payloads,
        }

    def _split_data(
        self,
        x: pd.DataFrame,
        y: pd.Series,
        split_series: pd.Series | None = None,
        time_series: pd.Series | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, str]:
        if split_series is not None:
            split_values = split_series.fillna("unknown").astype(str).str.lower()
            train_mask = split_values == "train"
            test_mask = split_values == "test"
            if train_mask.any() and test_mask.any():
                x_train = x.loc[train_mask]
                y_train = y.loc[train_mask]
                x_test = x.loc[test_mask]
                y_test = y.loc[test_mask]
                # Si test no tiene variacion o es muy pequeno, volvemos al split aleatorio.
                if len(y_test) > 1 and len(np.unique(y_test)) > 1:
                    return x_train, x_test, y_train, y_test, "fixed_train_test_split"

        if time_series is not None and len(x) > 10:
            order = pd.to_numeric(time_series, errors="coerce")
            fallback = pd.Series(np.arange(len(x), dtype=float), index=x.index)
            order = order.where(order.notna(), fallback)
            sorted_idx = order.sort_values().index
            split_point = max(1, int(len(sorted_idx) * (1.0 - self.config.test_size)))
            train_idx = sorted_idx[:split_point]
            test_idx = sorted_idx[split_point:]
            if len(test_idx) > 1 and len(np.unique(y.loc[test_idx])) > 1:
                return (
                    x.loc[train_idx],
                    x.loc[test_idx],
                    y.loc[train_idx],
                    y.loc[test_idx],
                    "temporal_holdout_split",
                )

        try:
            x_train, x_test, y_train, y_test = train_test_split(
                x,
                y,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=y,
            )
            return x_train, x_test, y_train, y_test, "random_stratified_split"
        except ValueError:
            x_train, x_test, y_train, y_test = train_test_split(
                x,
                y,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=None,
            )
            return x_train, x_test, y_train, y_test, "random_split"

    def _build_models(self) -> dict[str, Pipeline]:
        return {
            "random_forest": Pipeline(
                steps=[
                    ("model", RandomForestClassifier(n_estimators=200, random_state=self.config.random_state))
                ]
            ),
            "logistic_regression": Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=1000,
                            random_state=self.config.random_state,
                        ),
                    ),
                ]
            ),
            "gradient_boosting": Pipeline(
                steps=[
                    ("model", GradientBoostingClassifier(random_state=self.config.random_state))
                ]
            ),
        }

    def _save_artifacts(self, model: Pipeline, leaderboard: pd.DataFrame) -> None:
        self.config.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.metrics_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, self.config.model_path)
        leaderboard.to_csv(self.config.leaderboard_path, index=False)

    def _extract_importance(self, model: Pipeline, feature_names: list[str]) -> list[float]:
        estimator = model.named_steps["model"]
        if hasattr(estimator, "feature_importances_"):
            values = estimator.feature_importances_
            return [float(v) for v in values]
        if hasattr(estimator, "coef_"):
            coeffs = np.abs(estimator.coef_)
            if coeffs.ndim == 2:
                coeffs = coeffs.mean(axis=0)
            return [float(v) for v in coeffs]
        return [0.0 for _ in feature_names]

    @staticmethod
    def _extract_scores(model: Pipeline, x_test: pd.DataFrame) -> np.ndarray | None:
        if hasattr(model, "predict_proba"):
            try:
                scores = model.predict_proba(x_test)
                return np.asarray(scores)
            except Exception:
                return None
        if hasattr(model, "decision_function"):
            try:
                scores = model.decision_function(x_test)
                arr = np.asarray(scores)
                if arr.ndim == 1:
                    arr = np.vstack([1.0 - arr, arr]).T
                return arr
            except Exception:
                return None
        return None
