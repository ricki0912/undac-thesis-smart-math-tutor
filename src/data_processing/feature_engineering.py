from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.utils.logger import Logger


class FeatureEngineer:
    """
    Clase para construir variables explicativas y objetivo de dificultad.

    Restriccion anti-leakage:
    - NO usamos variables del intento actual (incorrects/hints/tiempo/correct_first_attempt)
      como features.
    - Solo se permiten agregados historicos (shift/solo pasado) y features del ejercicio
      derivadas de step_name.
    """

    feature_columns = [
        "student_attempt_count_prev",
        "student_avg_incorrects_prev",
        "student_avg_time_prev",
        "student_accuracy_prev",
        "student_trend_accuracy",
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

        include_target=True:
        - agrega effort_score (target continuo) usando variables post-resolucion.
          (Estas variables NO entran como input del modelo).
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

        if include_target:
            data["effort_score"] = self.build_effort_score(data)

        # Features historicas por estudiante (solo informacion previa).
        data["student_attempt_count_prev"] = data.groupby("student_id").cumcount().astype(float)
        data["student_avg_incorrects_prev"] = self._group_prev_mean(data, "student_id", "incorrects")
        data["student_avg_time_prev"] = self._group_prev_mean(data, "student_id", "step_duration_sec")
        data["student_accuracy_prev"] = self._group_prev_mean(data, "student_id", "correct_first_attempt")
        data["student_trend_accuracy"] = self._group_prev_rolling_mean(
            data, group_col="student_id", value_col="correct_first_attempt", window=5
        )
        Logger.print(f"Feature student_avg_incorrects_prev: media={data['student_avg_incorrects_prev'].mean():.4f}, std={data['student_avg_incorrects_prev'].std():.4f}, nulos={data['student_avg_incorrects_prev'].isnull().sum()}")
        Logger.print(f"Feature student_avg_time_prev: media={data['student_avg_time_prev'].mean():.4f}, std={data['student_avg_time_prev'].std():.4f}, nulos={data['student_avg_time_prev'].isnull().sum()}")
        Logger.print(f"Feature student_accuracy_prev: media={data['student_accuracy_prev'].mean():.4f}, std={data['student_accuracy_prev'].std():.4f}, nulos={data['student_accuracy_prev'].isnull().sum()}")
        Logger.print(f"Feature student_trend_accuracy: media={data['student_trend_accuracy'].mean():.4f}, std={data['student_trend_accuracy'].std():.4f}, nulos={data['student_trend_accuracy'].isnull().sum()}")
        # Features estructurales de la ecuacion/paso.
        data["step_len"] = data["step_name"].str.len().astype(float)
        data["step_num_ops"] = data["step_name"].str.count(r"[+\-*/=]").astype(float)
        data["step_has_parentheses"] = data["step_name"].str.contains(r"[()]", regex=True).astype(float)
        data["step_num_digits"] = data["step_name"].str.count(r"\d").astype(float)
        data["step_num_variables"] = data["step_name"].str.count(r"[a-zA-Z]").astype(float)
        data["step_abs_constant_sum"] = data["step_name"].apply(self._sum_abs_constants).astype(float)
        Logger.print(f"Features listas. Filas: {len(data)}.")
        
        # === CAPPING DE OUTLIERS ===
        Logger.print("Aplicando capping de outliers...")
        outlier_caps = {
            "step_abs_constant_sum": 0.99,
            "step_len": 0.99,
            "student_attempt_count_prev": 0.99,
            "step_num_ops": 0.95,
            "student_avg_time_prev": 0.99,
            "step_num_digits": 0.99,
            "step_num_variables": 0.95,
        }
        
        for col, quantile in outlier_caps.items():
            if col in data.columns:
                cap_value = data[col].quantile(quantile)
                original_max = data[col].max()
                data[col] = data[col].clip(upper=cap_value)
                Logger.print(f"{col}: capped percentil {int(quantile*100)} = {cap_value:.0f} (original max: {original_max:.0f})")
        
        Logger.print("Outliers capped completado.")
        Logger.print(f"Resumen features post-capping: {len(self.feature_columns)} creadas. Estadísticas globales: {data[self.feature_columns].describe().to_string()}")
        return data

    def split_features_target(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """
        Separa matriz de features y vector objetivo (target continuo).
        """
        x = df[self.feature_columns].copy()
        if "effort_score" not in df.columns:
            raise KeyError("Falta columna effort_score. Ejecuta transform(include_target=True).")
        y = df["effort_score"].copy()
        return x, y

    @staticmethod
    def build_effort_score(df: pd.DataFrame) -> pd.Series:
        """
        Target continuo: esfuerzo observado post-resolucion.

        Nota: incorrects/hints/tiempo/correct_first_attempt NO se usan como features,
        solo para construir el target.
        """
        inc_norm = FeatureEngineer._minmax(pd.to_numeric(df.get("incorrects", 0), errors="coerce").fillna(0))
        hints_norm = FeatureEngineer._minmax(pd.to_numeric(df.get("hints", 0), errors="coerce").fillna(0))
        time_norm = FeatureEngineer._minmax(
            pd.to_numeric(df.get("step_duration_sec", 0), errors="coerce").fillna(0)
        )
        cfa = pd.to_numeric(df.get("correct_first_attempt", 0), errors="coerce").fillna(0).clip(0, 1)
        fail_penalty = 1.0 - cfa

        #score = (0.40 * inc_norm) + (0.30 * hints_norm) + (0.30 * time_norm) + (0.15 * fail_penalty)
        score = (0.15 * inc_norm) + (0.15 * hints_norm) + (0.55 * time_norm) + (0.15 * fail_penalty)
        Logger.print("Construyendo effort_score con pesos: inc=0.15, hints=0.15, time=0.35, fail_penalty=0.15.")
        Logger.print("Effort score construido. Estadísticas:")
        Logger.print(score.describe())
        return score.astype(float)

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
    def _group_prev_rolling_mean(
        df: pd.DataFrame, group_col: str, value_col: str, window: int = 5
    ) -> pd.Series:
        """
        Rolling mean SOLO con pasado: shift(1) + rolling(window).
        """
        values = pd.to_numeric(df[value_col], errors="coerce").fillna(0)
        shifted = values.groupby(df[group_col]).shift(1)
        rolled = (
            shifted.groupby(df[group_col])
            .rolling(int(window), min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        default_value = float(values.mean()) if len(df) else 0.0
        return rolled.fillna(default_value).astype(float)

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
    def exercise_features_from_step_name(step_name: str) -> dict[str, float]:
        """
        Features del ejercicio derivadas solo de step_name (independientes del intento).
        """
        text = str(step_name or "")
        return {
            "step_len": float(len(text)),
            "step_num_ops": float(len(re.findall(r"[+\-*/=]", text))),
            "step_has_parentheses": float(1 if re.search(r"[()]", text) else 0),
            "step_num_digits": float(len(re.findall(r"\d", text))),
            "step_num_variables": float(len(re.findall(r"[a-zA-Z]", text))),
            "step_abs_constant_sum": float(FeatureEngineer._sum_abs_constants(text)),
        }

    @staticmethod
    def student_history_snapshot(history_df: pd.DataFrame, student_id: str) -> dict[str, float]:
        """
        Snapshot historico (solo pasado) para un estudiante, a partir de logs/dataset.

        Nota: este snapshot se usa para inferencia "antes del intento".
        """
        if history_df is None or history_df.empty:
            return {
                "student_attempt_count_prev": 0.0,
                "student_avg_incorrects_prev": 0.0,
                "student_avg_time_prev": 0.0,
                "student_accuracy_prev": 0.0,
                "student_trend_accuracy": 0.0,
            }

        df = history_df.copy()
        df["student_id"] = df.get("student_id", "").fillna("").astype(str)
        sid = str(student_id)
        student_df = df[df["student_id"] == sid].copy()
        if student_df.empty:
            # defaults globales (cuando es estudiante nuevo)
            incorrects = pd.to_numeric(df.get("incorrects", 0), errors="coerce").fillna(0)
            durations = pd.to_numeric(df.get("step_duration_sec", 0), errors="coerce").fillna(0)
            cfa = pd.to_numeric(df.get("correct_first_attempt", 0), errors="coerce").fillna(0).clip(0, 1)
            return {
                "student_attempt_count_prev": 0.0,
                "student_avg_incorrects_prev": float(incorrects.mean()) if len(df) else 0.0,
                "student_avg_time_prev": float(durations.mean()) if len(df) else 0.0,
                "student_accuracy_prev": float(cfa.mean()) if len(df) else 0.0,
                "student_trend_accuracy": float(cfa.tail(5).mean()) if len(df) else 0.0,
            }

        # ordenar por timestamp si existe; si no, por orden natural
        if "timestamp" in student_df.columns:
            t = pd.to_datetime(student_df["timestamp"], errors="coerce")
            if t.notna().any():
                student_df = student_df.assign(_t=t).sort_values("_t")

        incorrects = pd.to_numeric(student_df.get("incorrects", 0), errors="coerce").fillna(0)
        durations = pd.to_numeric(student_df.get("step_duration_sec", 0), errors="coerce").fillna(0)
        cfa = pd.to_numeric(student_df.get("correct_first_attempt", 0), errors="coerce").fillna(0).clip(0, 1)

        return {
            "student_attempt_count_prev": float(len(student_df)),
            "student_avg_incorrects_prev": float(incorrects.mean()) if len(student_df) else 0.0,
            "student_avg_time_prev": float(durations.mean()) if len(student_df) else 0.0,
            "student_accuracy_prev": float(cfa.mean()) if len(student_df) else 0.0,
            "student_trend_accuracy": float(cfa.tail(5).mean()) if len(student_df) else 0.0,
        }

    @staticmethod
    def _build_event_order(df: pd.DataFrame) -> pd.Series:
        if "timestamp" in df.columns:
            parsed = pd.to_datetime(df["timestamp"], errors="coerce")
            if parsed.notna().any():
                fallback = pd.Series(np.arange(len(df), dtype=float), index=df.index)
                parsed_ns = pd.Series(parsed.astype("int64"), index=df.index)
                return parsed_ns.where(parsed.notna(), fallback)
        return pd.Series(np.arange(len(df), dtype=float), index=df.index)
