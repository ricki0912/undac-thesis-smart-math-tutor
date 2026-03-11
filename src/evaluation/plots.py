from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize


class PlotGenerator:
    """
    Clase para crear visualizaciones del comportamiento y del entrenamiento.
    """

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid")

    def plot_error_distribution(self, df: pd.DataFrame) -> Path:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(df["incorrects"], bins=20, kde=True, ax=ax, color="#e67e22")
        ax.set_title("Distribucion de errores")
        ax.set_xlabel("Errores por paso")
        ax.set_ylabel("Frecuencia")
        return self._save(fig, "error_distribution.png")

    def plot_avg_time_by_step(self, df: pd.DataFrame) -> Path:
        top_steps = (
            df.groupby("step_name", as_index=False)["step_duration_sec"]
            .mean()
            .sort_values("step_duration_sec", ascending=False)
            .head(15)
        )
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=top_steps, x="step_duration_sec", y="step_name", ax=ax, color="#3498db")
        ax.set_title("Tiempo promedio por ejercicio")
        ax.set_xlabel("Tiempo promedio (seg)")
        ax.set_ylabel("Paso")
        return self._save(fig, "avg_time_by_step.png")

    def plot_correlation_matrix(self, df: pd.DataFrame) -> Path:
        numeric_df = df.select_dtypes(include=["number"])
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="viridis", ax=ax)
        ax.set_title("Matriz de correlacion")
        return self._save(fig, "correlation_matrix.png")

    def plot_feature_importance(self, feature_names: list[str], importances: list[float]) -> Path:
        rank_df = pd.DataFrame({"feature": feature_names, "importance": importances})
        rank_df = rank_df.sort_values("importance", ascending=False)
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=rank_df, x="importance", y="feature", ax=ax, color="#2ecc71")
        ax.set_title("Importancia de features")
        ax.set_xlabel("Importancia")
        ax.set_ylabel("Feature")
        return self._save(fig, "feature_importance.png")

    def plot_training_evolution(self, leaderboard: pd.DataFrame) -> Path:
        fig, ax = plt.subplots(figsize=(8, 4))
        plot_df = leaderboard.reset_index(drop=True).copy()
        plot_df["iteration"] = plot_df.index + 1
        sns.lineplot(data=plot_df, x="iteration", y="f1_weighted", marker="o", ax=ax, color="#9b59b6")
        ax.set_title("Evolucion del entrenamiento por modelo")
        ax.set_xlabel("Iteracion de modelo")
        ax.set_ylabel("F1-weighted")
        return self._save(fig, "training_evolution.png")

    def plot_model_metric_comparison(self, leaderboard: pd.DataFrame) -> Path:
        metrics = ["accuracy", "f1_weighted", "auc_ovr_weighted"]
        plot_df = leaderboard[["model_name", *metrics]].melt(
            id_vars=["model_name"],
            value_vars=metrics,
            var_name="metric",
            value_name="value",
        )
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=plot_df, x="model_name", y="value", hue="metric", ax=ax)
        ax.set_title("Comparacion de metricas por modelo")
        ax.set_xlabel("Modelo")
        ax.set_ylabel("Valor")
        ax.set_ylim(0.0, 1.05)
        ax.tick_params(axis="x", rotation=20)
        return self._save(fig, "model_metric_comparison.png")

    def plot_per_model_diagnostics(self, evaluation_payloads: list[dict[str, object]]) -> list[Path]:
        generated: list[Path] = []
        for payload in evaluation_payloads:
            model_name = str(payload["model_name"])
            y_true = pd.Series(payload["y_true"]).astype(int)
            y_pred = pd.Series(payload["y_pred"]).astype(int)
            labels = sorted({int(v) for v in payload.get("labels", [0, 1, 2])})
            y_score = payload.get("y_score")

            generated.append(self._plot_confusion_matrix(model_name, y_true, y_pred, labels))
            if y_score is not None:
                try:
                    generated.append(
                        self._plot_roc_ovr(model_name, y_true, np.asarray(y_score), labels)
                    )
                except Exception:
                    # Si no se puede graficar ROC para algun modelo, continuamos.
                    continue
        return generated

    def _plot_confusion_matrix(
        self,
        model_name: str,
        y_true: pd.Series,
        y_pred: pd.Series,
        labels: list[int],
    ) -> Path:
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
        )
        ax.set_title(f"Matriz de confusion - {model_name}")
        ax.set_xlabel("Prediccion")
        ax.set_ylabel("Real")
        safe_name = model_name.replace(" ", "_").lower()
        return self._save(fig, f"model_{safe_name}_confusion_matrix.png")

    def _plot_roc_ovr(
        self,
        model_name: str,
        y_true: pd.Series,
        y_score: np.ndarray,
        labels: list[int],
    ) -> Path:
        y_bin = label_binarize(y_true, classes=labels)
        if y_bin.ndim == 1:
            y_bin = np.vstack([1 - y_bin, y_bin]).T
        if y_score.ndim == 1:
            y_score = np.vstack([1.0 - y_score, y_score]).T

        fig, ax = plt.subplots(figsize=(6, 5))
        for idx, cls in enumerate(labels):
            if idx >= y_score.shape[1]:
                break
            fpr, tpr, _ = roc_curve(y_bin[:, idx], y_score[:, idx])
            cls_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"Clase {cls} (AUC={cls_auc:.2f})")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1.0, label="Azar")
        ax.set_title(f"ROC OVR - {model_name}")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.legend(loc="lower right")
        safe_name = model_name.replace(" ", "_").lower()
        return self._save(fig, f"model_{safe_name}_roc_ovr.png")

    def _save(self, fig: plt.Figure, filename: str) -> Path:
        output_path = self.output_dir / filename
        fig.tight_layout()
        fig.savefig(output_path, dpi=140)
        plt.close(fig)
        return output_path

