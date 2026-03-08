from __future__ import annotations

import numpy as np
import pandas as pd


class FeatureEngineer:
    """
    Clase para construir variables explicativas y objetivo de dificultad.
    """

    feature_columns = [
        "incorrects",
        "hints",
        "step_duration_sec",
        "correct_first_attempt",
        "error_rate",
        "time_efficiency",
        "difficulty_score",
    ]

    def transform(self, df: pd.DataFrame, include_target: bool = True) -> pd.DataFrame:
        """
        Crea nuevas features para capturar comportamiento del estudiante.
        """
        data = df.copy()
        attempts = data["incorrects"] + data["hints"] + 1.0
        data["error_rate"] = data["incorrects"] / attempts
        data["time_efficiency"] = data["correct_first_attempt"] / (data["step_duration_sec"] + 1.0)

        # Combinacion simple y explicable para una puntuacion inicial de dificultad.
        inc_norm = self._minmax(data["incorrects"])
        hints_norm = self._minmax(data["hints"])
        time_norm = self._minmax(data["step_duration_sec"])
        fail_penalty = 1.0 - data["correct_first_attempt"]
        data["difficulty_score"] = (
            (0.35 * inc_norm)
            + (0.25 * hints_norm)
            + (0.25 * time_norm)
            + (0.15 * fail_penalty)
        )

        if include_target:
            # Convertimos difficulty_score a etiquetas discretas solo en entrenamiento.
            q1 = data["difficulty_score"].quantile(0.33)
            q2 = data["difficulty_score"].quantile(0.66)
            if q1 == q2:
                data["difficulty_level"] = 1
            else:
                bins = [-np.inf, q1, q2, np.inf]
                labels = [0, 1, 2]  # 0=baja, 1=media, 2=alta
                data["difficulty_level"] = pd.cut(
                    data["difficulty_score"], bins=bins, labels=labels, include_lowest=True
                ).astype(int)
        return data

    def split_features_target(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """
        Separa matriz de features y vector objetivo.
        """
        x = df[self.feature_columns].copy()
        y = df["difficulty_level"].copy()
        return x, y

    @staticmethod
    def _minmax(series: pd.Series) -> pd.Series:
        minimum = float(series.min())
        maximum = float(series.max())
        if maximum == minimum:
            return pd.Series([0.0] * len(series), index=series.index)
        return (series - minimum) / (maximum - minimum)
