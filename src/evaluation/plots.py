from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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
        sns.lineplot(data=plot_df, x="iteration", y="f1_score", marker="o", ax=ax, color="#9b59b6")
        ax.set_title("Evolucion del entrenamiento por modelo")
        ax.set_xlabel("Iteracion de modelo")
        ax.set_ylabel("F1-score")
        return self._save(fig, "training_evolution.png")

    def _save(self, fig: plt.Figure, filename: str) -> Path:
        output_path = self.output_dir / filename
        fig.tight_layout()
        fig.savefig(output_path, dpi=140)
        plt.close(fig)
        return output_path

