from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.utils.logger import Logger


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
        "student_attempt_count_prev",
        "student_avg_incorrects_prev",
        "student_avg_time_prev",
        "student_accuracy_prev",
        "step_success_rate_prev",
        "step_len",
        "step_num_ops",
        "step_has_parentheses",
        "step_num_digits",
        "step_num_variables",
        "step_abs_constant_sum",
    ]

    def transform(self, df: pd.DataFrame, include_target: bool = True) -> pd.DataFrame:
        """
        Crea nuevas features para capturar comportamiento del estudiante.
        """
        Logger.print(f"Construyendo features (include_target={include_target})...")
        data = df.copy()
        if "student_id" not in data.columns:
            data["student_id"] = "unknown_student"
        if "step_name" not in data.columns:
            data["step_name"] = "unknown_step"
        data["student_id"] = data["student_id"].fillna("unknown_student").astype(str)
        data["step_name"] = data["step_name"].fillna("unknown_step").astype(str)

        # Orden temporal para evitar fuga de informacion en features historicas.
        data["event_order"] = self._build_event_order(data)
        data = data.sort_values("event_order").reset_index(drop=True)

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

        # Features historicas por estudiante (solo informacion previa).
        data["student_attempt_count_prev"] = data.groupby("student_id").cumcount().astype(float)
        data["student_avg_incorrects_prev"] = self._group_prev_mean(data, "student_id", "incorrects")
        data["student_avg_time_prev"] = self._group_prev_mean(data, "student_id", "step_duration_sec")
        data["student_accuracy_prev"] = self._group_prev_mean(data, "student_id", "correct_first_attempt")

        # Historial de dificultad por tipo de paso.
        data["step_success_rate_prev"] = self._group_prev_mean(data, "step_name", "correct_first_attempt")

        # Features estructurales de la ecuacion/paso.
        data["step_len"] = data["step_name"].str.len().astype(float)
        data["step_num_ops"] = data["step_name"].str.count(r"[+\-*/=]").astype(float)
        data["step_has_parentheses"] = data["step_name"].str.contains(r"[()]", regex=True).astype(float)
        data["step_num_digits"] = data["step_name"].str.count(r"\d").astype(float)
        data["step_num_variables"] = data["step_name"].str.count(r"[a-zA-Z]").astype(float)
        data["step_abs_constant_sum"] = data["step_name"].apply(self._sum_abs_constants).astype(float)

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
        Logger.print(f"Features listas. Filas: {len(data)}.")
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

    @staticmethod
    def _group_prev_mean(df: pd.DataFrame, group_col: str, value_col: str) -> pd.Series:
        grouped = df.groupby(group_col)[value_col]
        cumsum = grouped.cumsum() - df[value_col]
        prev_count = grouped.cumcount()
        prev_mean = cumsum / prev_count.replace(0, np.nan)
        default_value = float(df[value_col].mean()) if len(df) else 0.0
        return prev_mean.fillna(default_value).astype(float)

    @staticmethod
    def _sum_abs_constants(text: str) -> int:
        numbers = re.findall(r"-?\d+", str(text))
        if not numbers:
            return 0
        # Evita outliers por registros corruptos con numeros extremadamente largos.
        safe_numbers = [n for n in numbers if len(n.lstrip("-")) <= 4]
        if not safe_numbers:
            return 0
        return int(sum(abs(int(n)) for n in safe_numbers))

    @staticmethod
    def _build_event_order(df: pd.DataFrame) -> pd.Series:
        if "timestamp" in df.columns:
            parsed = pd.to_datetime(df["timestamp"], errors="coerce")
            if parsed.notna().any():
                fallback = pd.Series(np.arange(len(df), dtype=float), index=df.index)
                parsed_ns = pd.Series(parsed.astype("int64"), index=df.index)
                return parsed_ns.where(parsed.notna(), fallback)
        return pd.Series(np.arange(len(df), dtype=float), index=df.index)
