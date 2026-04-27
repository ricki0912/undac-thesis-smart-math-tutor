from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.multiclass import type_of_target

from src.evaluation.metrics import MetricsEvaluator
from src.utils.config import ProjectConfig
from src.utils.logger import Logger


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
        groups: pd.Series | None = None,
    ) -> dict[str, Any]:
        x_train, x_test, y_train, y_test, validation_type = self._split_data(
            x, y, split_series, time_series, groups
        )
        y_train, y_test, label_thresholds = self._ensure_discrete_target(y_train, y_test)
        Logger.print(f"=== INICIO ENTRENAMIENTO DE MODELOS ===")
        Logger.print(f"Tipo de validaciÃ³n: {validation_type}")
        Logger.print(f"TamaÃ±o train: {len(x_train)} filas, {x_train.shape[1]} features")
        Logger.print(f"TamaÃ±o test: {len(x_test)} filas, {x_test.shape[1]} features")
        Logger.print(f"Target discretizado: {label_thresholds is not None}")
        if label_thresholds:
            Logger.print(f"Umbrales de discretizaciÃ³n: {label_thresholds}")

        models = self._build_models()
        Logger.print(f"Modelos a evaluar: {list(models.keys())}")

        results: list[dict[str, float | str]] = []
        evaluation_payloads: list[dict[str, Any]] = []
        best_name = ""
        best_model: Pipeline | None = None
        best_score = -np.inf

        training_times = {}

        for model_name, model_pipeline in models.items():
            Logger.print(f"=== ENTRENANDO MODELO: {model_name} ===")

            # Log hiperparÃ¡metros
            model_params = model_pipeline.get_params()
            Logger.print(f"HiperparÃ¡metros de {model_name}:")
            for param_name, param_value in model_params.items():
                if 'model__' in param_name:  # Solo parÃ¡metros del estimador final
                    Logger.print(f"  {param_name}: {param_value}")

            # Entrenamiento con timing
            import time
            start_time = time.time()
            model = clone(model_pipeline)
            model.fit(x_train, y_train)
            training_time = time.time() - start_time
            training_times[model_name] = training_time
            Logger.print(f"Tiempo de entrenamiento {model_name}: {training_time:.2f} segundos")

            # Predicciones
            y_pred = model.predict(x_test)
            y_score = self._extract_scores(model, x_test)

            # CÃ¡lculo de mÃ©tricas
            metrics = self.evaluator.compute(y_test, y_pred, y_score)
            metrics["model_name"] = model_name
            metrics["training_time"] = training_time
            results.append(metrics)

            # Log detallado de mÃ©tricas
            Logger.print(f"=== MÃ‰TRICAS DE {model_name} ===")
            Logger.print(f"  F1-weighted: {metrics['f1_weighted']:.4f}")
            Logger.print(f"  Accuracy: {metrics['accuracy']:.4f}")
            Logger.print(f"  Precision: {metrics['precision']:.4f}")
            Logger.print(f"  Recall: {metrics['recall']:.4f}")
            Logger.print(f"  F1-macro: {metrics['f1_macro']:.4f}")
            Logger.print(f"  AUC: {metrics['auc']:.4f}")
            Logger.print(f"  Tiempo entrenamiento: {training_time:.2f}s")

            # Matriz de confusiÃ³n resumida
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test, y_pred)
            Logger.print(f"  Matriz de confusiÃ³n:\n{cm}")

            evaluation_payloads.append(
                {
                    "model_name": model_name,
                    "y_true": y_test.copy(),
                    "y_pred": pd.Series(y_pred, index=y_test.index),
                    "y_score": y_score,
                    "labels": sorted({int(v) for v in pd.unique(y_train)}),
                    "training_time": training_time,
                    "hyperparameters": model_params,
                }
            )

            if metrics["f1_weighted"] > best_score:
                best_score = metrics["f1_weighted"]
                best_name = model_name
                best_model = model
                Logger.print(f"Â¡NUEVO MEJOR MODELO: {model_name} (F1={best_score:.4f})!")

        Logger.print(f"=== RESUMEN ENTRENAMIENTO ===")
        Logger.print(f"Mejor modelo: {best_name} (F1-weighted={best_score:.4f})")
        Logger.print(f"Tiempos de entrenamiento:")
        for model, time_taken in training_times.items():
            Logger.print(f"  {model}: {time_taken:.2f}s")

        if best_model is None:
            raise RuntimeError("No se pudo entrenar ningun modelo.")

        leaderboard = self.evaluator.to_dataframe(results)
        self._save_artifacts(best_model, leaderboard)
        Logger.print(f"Mejor modelo seleccionado: {best_name} (f1_weighted={best_score:.4f}).")

        importances = self._extract_importance(best_model, list(x.columns))
        Logger.print(f"Importancia de features extraÃ­da para {best_name}.")

        return {
            "best_model_name": best_name,
            "best_model": best_model,
            "leaderboard": leaderboard,
            "feature_importance": importances,
            "validation_type": validation_type,
            "evaluation_payloads": evaluation_payloads,
            "label_thresholds": label_thresholds,
            "training_times": training_times,
        }

    @staticmethod
    def _ensure_discrete_target(
        y_train: pd.Series, y_test: pd.Series
    ) -> tuple[pd.Series, pd.Series, dict[str, float] | None]:
        """
        Los modelos definidos son clasificadores; si el target es continuo, lo
        discretizamos a 3 clases usando cuantiles SOLO del train.
        """
        Logger.print("=== EVALUACIÃ“N DEL TARGET ===")
        y_train_num = pd.to_numeric(y_train, errors="coerce").fillna(0.0)
        y_test_num = pd.to_numeric(y_test, errors="coerce").fillna(0.0)

        target_type = type_of_target(y_train_num)
        Logger.print(f"Tipo de target detectado: {target_type}")

        if target_type != "continuous":
            Logger.print("Target ya es discreto/categÃ³rico, no se requiere discretizaciÃ³n")
            return y_train_num, y_test_num, None

        Logger.print("=== DISCRETIZACIÃ“N DEL TARGET CONTINUO ===")
        Logger.print("Estrategia: discretizaciÃ³n a 3 clases usando cuantiles del conjunto de train")
        Logger.print(f"EstadÃ­sticas target train: media={y_train_num.mean():.4f}, std={y_train_num.std():.4f}, min={y_train_num.min():.4f}, max={y_train_num.max():.4f}")

        q1 = float(y_train_num.quantile(0.33))
        q2 = float(y_train_num.quantile(0.66))
        Logger.print(f"Cuartil 33% (Q1): {q1:.4f}")
        Logger.print(f"Cuartil 66% (Q2): {q2:.4f}")

        if q1 == q2:
            Logger.print("Q1=Q2, ajustando cuantiles para evitar colapso de clases")
            q1 = float(y_train_num.quantile(0.25))
            q2 = float(y_train_num.quantile(0.75))
            Logger.print(f"Nuevos cuantiles - Q1 (25%): {q1:.4f}, Q2 (75%): {q2:.4f}")

        bins = [-np.inf, q1, q2, np.inf]
        labels = [0, 1, 2]
        Logger.print(f"Bins de discretizaciÃ³n: {bins}")
        Logger.print(f"Labels de clase: {labels} (0=bajo, 1=medio, 2=alto esfuerzo)")

        y_train_binned = pd.cut(
            y_train_num, bins=bins, labels=labels, include_lowest=True
        ).astype(int)
        y_test_binned = pd.cut(
            y_test_num, bins=bins, labels=labels, include_lowest=True
        ).astype(int)

        Logger.print("=== DISTRIBUCIÃ“N POST-DISCRETIZACIÃ“N ===")
        Logger.print(f"Train - Clase 0: {(y_train_binned == 0).sum()}, Clase 1: {(y_train_binned == 1).sum()}, Clase 2: {(y_train_binned == 2).sum()}")
        Logger.print(f"Test - Clase 0: {(y_test_binned == 0).sum()}, Clase 1: {(y_test_binned == 1).sum()}, Clase 2: {(y_test_binned == 2).sum()}")

        return y_train_binned, y_test_binned, {"q1": q1, "q2": q2}

    def _split_data(
        self,
        x: pd.DataFrame,
        y: pd.Series,
        split_series: pd.Series | None = None,
        time_series: pd.Series | None = None,
        groups: pd.Series | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, str]:
        Logger.print("=== ESTRATEGIA DE SPLIT DE DATOS ===")
        Logger.print(f"Dataset total: {len(x)} filas")
        Logger.print(f"Test size configurado: {self.config.test_size}")

        if split_series is not None:
            Logger.print("Intentando split fijo (train/test) basado en split_series...")
            split_values = split_series.fillna("unknown").astype(str).str.lower()
            train_mask = split_values == "train"
            test_mask = split_values == "test"
            if train_mask.any() and test_mask.any():
                x_train = x.loc[train_mask]
                y_train = y.loc[train_mask]
                x_test = x.loc[test_mask]
                y_test = y.loc[test_mask]
                Logger.print(f"Split fijo encontrado: Train={len(x_train)}, Test={len(x_test)}")
                # Si test no tiene variacion o es muy pequeno, volvemos al split aleatorio.
                if len(y_test) > 1 and len(np.unique(y_test)) > 1:
                    Logger.print("Split fijo vÃ¡lido - usando fixed_train_test_split")
                    return x_train, x_test, y_train, y_test, "fixed_train_test_split"
                else:
                    Logger.print("Split fijo invÃ¡lido (test sin variaciÃ³n) - fallback a aleatorio")

        if groups is not None and len(x) > 10:
            Logger.print("Intentando split por grupos (estudiantes)...")
            group_values = groups.fillna("unknown").astype(str)
            num_groups = group_values.nunique()
            Logger.print(f"Grupos detectados: {num_groups} grupos Ãºnicos")
            if group_values.nunique() > 1:
                splitter = GroupShuffleSplit(
                    n_splits=1,
                    test_size=self.config.test_size,
                    random_state=self.config.random_state,
                )
                train_idx, test_idx = next(splitter.split(x, y, groups=group_values))
                x_train = x.iloc[train_idx]
                y_train = y.iloc[train_idx]
                x_test = x.iloc[test_idx]
                y_test = y.iloc[test_idx]
                Logger.print(f"Split por grupos: Train={len(x_train)} ({len(y_train.unique())} clases), Test={len(x_test)} ({len(y_test.unique())} clases)")
                if len(y_test) > 1 and len(np.unique(y_test)) > 1:
                    Logger.print("Split por grupos vÃ¡lido - usando group_student_holdout_split")
                    return x_train, x_test, y_train, y_test, "group_student_holdout_split"
                else:
                    Logger.print("Split por grupos invÃ¡lido - fallback a otra estrategia")

        if time_series is not None and len(x) > 10:
            Logger.print("Intentando split temporal...")
            order = pd.to_numeric(time_series, errors="coerce")
            fallback = pd.Series(np.arange(len(x), dtype=float), index=x.index)
            order = order.where(order.notna(), fallback)
            sorted_idx = order.sort_values().index
            split_point = max(1, int(len(sorted_idx) * (1.0 - self.config.test_size)))
            train_idx = sorted_idx[:split_point]
            test_idx = sorted_idx[split_point:]
            Logger.print(f"Split temporal: punto de corte en posiciÃ³n {split_point}/{len(sorted_idx)}")
            if len(test_idx) > 1 and len(np.unique(y.loc[test_idx])) > 1:
                x_train = x.loc[train_idx]
                y_train = y.loc[train_idx]
                x_test = x.loc[test_idx]
                y_test = y.loc[test_idx]
                Logger.print(f"Split temporal vÃ¡lido: Train={len(x_train)}, Test={len(x_test)} - usando temporal_holdout_split")
                return (
                    x_train,
                    x_test,
                    y_train,
                    y_test,
                    "temporal_holdout_split",
                )
            else:
                Logger.print("Split temporal invÃ¡lido - fallback a aleatorio")

        Logger.print("Usando split aleatorio estratificado...")
        try:
            x_train, x_test, y_train, y_test = train_test_split(
                x,
                y,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=y,
            )
            Logger.print(f"Split estratificado exitoso: Train={len(x_train)}, Test={len(x_test)}")
            return x_train, x_test, y_train, y_test, "random_stratified_split"
        except ValueError as e:
            Logger.print(f"Split estratificado fallÃ³ ({e}) - usando split aleatorio simple")
            x_train, x_test, y_train, y_test = train_test_split(
                x,
                y,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=None,
            )
            Logger.print(f"Split aleatorio: Train={len(x_train)}, Test={len(x_test)}")
            return x_train, x_test, y_train, y_test, "random_split"

    def _build_models(self) -> dict[str, Pipeline]:
        Logger.print("=== CONFIGURACION DE MODELOS ===")
        Logger.print(f"Perfil de modelo activo: {self.config.model_profile}")
        models: dict[str, Pipeline] = {
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
            "hist_gradient_boosting": Pipeline(
                steps=[
                    (
                        "model",
                        HistGradientBoostingClassifier(
                            max_depth=8,
                            max_iter=150,
                            learning_rate=0.08,
                            min_samples_leaf=20,
                            random_state=self.config.random_state,
                        ),
                    )
                ]
            ),
        }

        if self.config.random_forest_enabled:
            models["random_forest"] = Pipeline(
                steps=[
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=self.config.random_forest_n_estimators,
                            max_depth=self.config.random_forest_max_depth,
                            min_samples_leaf=self.config.random_forest_min_samples_leaf,
                            n_jobs=self.config.random_forest_n_jobs,
                            random_state=self.config.random_state,
                        ),
                    )
                ]
            )

        Logger.print("Modelos configurados:")
        for name, pipeline in models.items():
            Logger.print(f"  - {name}: {pipeline}")
            if name == "random_forest":
                Logger.print(
                    "    "
                    f"n_estimators={self.config.random_forest_n_estimators}, "
                    f"max_depth={self.config.random_forest_max_depth}, "
                    f"min_samples_leaf={self.config.random_forest_min_samples_leaf}, "
                    f"n_jobs={self.config.random_forest_n_jobs}, "
                    f"random_state={self.config.random_state}"
                )
            elif name == "logistic_regression":
                Logger.print(f"    max_iter=1000, random_state={self.config.random_state}, con StandardScaler")
            elif name == "hist_gradient_boosting":
                Logger.print("    max_depth=8, max_iter=150, learning_rate=0.08, min_samples_leaf=20")

        return models

    def _save_artifacts(self, model: Pipeline, leaderboard: pd.DataFrame) -> None:
        self.config.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.metrics_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, self.config.model_path, compress=self.config.model_artifact_compress)
        leaderboard.to_csv(self.config.leaderboard_path, index=False)
        try:
            artifact_size_mb = self.config.model_path.stat().st_size / (1024 * 1024)
            Logger.print(f"Tamano del artefacto guardado: {artifact_size_mb:.2f} MB")
        except OSError:
            Logger.print("No se pudo medir el tamano del artefacto guardado.", level="WARNING")

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



