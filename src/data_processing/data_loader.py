from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import ProjectConfig
from src.utils.logger import Logger


class DataLoader:
    """
    Clase encargada de cargar y limpiar datos de interacciones estudiantiles.
    """

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config

    def load_dataset(self) -> pd.DataFrame:
        """
        Carga dataset priorizando archivos KDD train/test en data/external.
        """
        Logger.print("=== INICIO CARGA DE DATASET ===")
        Logger.print(f"Buscando datos en: {self.config.external_data_dir}")
        Logger.print(f"Archivo local alternativo: {self.config.data_path}")

        # 1) Fuente principal: KDD externo (mismo nivel del proyecto)
        external_df = self._build_from_kdd_if_available()
        if external_df is not None:
            Logger.print("=== DATOS KDD ENCONTRADOS ===")
            Logger.print(f"Filas totales KDD: {len(external_df)}")
            Logger.print(f"Columnas KDD: {list(external_df.columns)}")
            Logger.print(f"Tipos de datos iniciales:\n{external_df.dtypes.to_string()}")

            # Estadísticas iniciales detalladas
            Logger.print("=== ESTADÍSTICAS INICIALES KDD ===")
            numeric_cols = external_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                Logger.print(f"Estadísticas numéricas iniciales:\n{external_df[numeric_cols].describe().to_string()}")

            Logger.print(f"Valores nulos por columna:\n{external_df.isnull().sum().to_string()}")
            Logger.print(f"Duplicados iniciales: {external_df.duplicated().sum()}")

            cleaned = self.clean_data(external_df)
            Logger.print(f"=== POST-LIMPIEZA KDD ===")
            Logger.print(f"Filas tras limpieza: {len(cleaned)}")
            Logger.print(f"Columnas tras limpieza: {len(cleaned.columns)}")
            Logger.print(f"Filas eliminadas: {len(external_df) - len(cleaned)}")

            cleaned = self._append_gameplay_logs(cleaned)
            Logger.print(f"Filas tras append gameplay logs: {len(cleaned)}")

            final_df = self.apply_dataset_rules(cleaned)
            Logger.print(f"=== DATASET FINAL KDD ===")
            Logger.print(f"Filas finales: {len(final_df)}")
            Logger.print(f"Columnas finales: {len(final_df.columns)}")
            Logger.print(f"Ratio de retención: {len(final_df)/len(external_df):.3f}")

            return final_df

        # 2) Respaldo: CSV local en data/dataset.csv
        if self.config.data_path.exists():
            Logger.print("=== USANDO DATASET LOCAL ===")
            Logger.print(f"Cargando: {self.config.data_path}")
            df = pd.read_csv(self.config.data_path)
            Logger.print(f"Filas dataset local: {len(df)}")
            Logger.print(f"Columnas dataset local: {list(df.columns)}")

            cleaned = self.clean_data(df)
            cleaned = self._append_gameplay_logs(cleaned)
            final_df = self.apply_dataset_rules(cleaned)

            Logger.print(f"=== DATASET LOCAL FINAL ===")
            Logger.print(f"Filas finales: {len(final_df)}")
            Logger.print(f"Ratio de retención: {len(final_df)/len(df):.3f}")

            return final_df

        raise FileNotFoundError(
            f"No se encontraron archivos en {self.config.external_data_dir} "
            f"ni el archivo local {self.config.data_path}."
        )

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpia nulos y fuerza tipos numericos en columnas clave.
        """
        df = df.copy()
        required_columns = [
            "student_id",
            "step_name",
            "incorrects",
            "hints",
            "correct_first_attempt",
            "step_duration_sec",
        ]
        for col in required_columns:
            if col not in df.columns:
                df[col] = 0 if col != "step_name" else "unknown_step"
        df["student_id"] = df["student_id"].fillna("unknown_student").astype(str)
        df["step_name"] = df["step_name"].fillna("unknown_step").astype(str)
        for col in ["incorrects", "hints", "correct_first_attempt", "step_duration_sec"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        df["correct_first_attempt"] = df["correct_first_attempt"].clip(0, 1).astype(int)
        df["incorrects"] = df["incorrects"].clip(lower=0)
        df["hints"] = df["hints"].clip(lower=0)
        df["step_duration_sec"] = df["step_duration_sec"].clip(lower=0)
        # Limpieza de outliers: cap al percentil 99.9 (winsorizacion).
        df = self._cap_column_percentile(df, "incorrects", percentile=0.999)
        Logger.print("Capping incorrects al percentil 99.9.")
        df = self._cap_column_percentile(df, "hints", percentile=0.999)
        Logger.print("Capping hints al percentil 99.9.")
        df = self._cap_column_percentile(df, "step_duration_sec", percentile=0.999)
        Logger.print("Capping step_duration_sec al percentil 99.9.")
        #df = self._cap_column_percentile(df, "incorrects", percentile=0.95)
        #df = self._cap_column_percentile(df, "hints", percentile=0.95)
        #df = self._cap_column_percentile(df, "step_duration_sec", percentile=0.95)
        if "timestamp" in df.columns:
            df["timestamp"] = df["timestamp"].astype(str)
        
        return df

    def apply_dataset_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica reglas opcionales de depuracion:
        - Filtrado de step_name por blacklist (si existe archivo).
        - Cap/drop de outliers en columnas numericas (si config lo indica).
        """
        data = df.copy()
        blacklist = self._load_step_name_blacklist()
        if blacklist:
            Logger.print(f"Blacklist de step_name cargada: {len(blacklist)} nombres a filtrar.")
        if blacklist:
            data = self.filter_step_names(data, blacklist)

        data = self._apply_outlier_rules(data)
        Logger.print(f"Dataset tras aplicar reglas: {len(data)} filas.")
        return data.reset_index(drop=True)

    @staticmethod
    def filter_step_names(df: pd.DataFrame, step_names_to_drop: set[str]) -> pd.DataFrame:
        """
        Elimina filas donde step_name pertenece a la lista dada.
        """
        if df.empty or not step_names_to_drop:
            return df
        if "step_name" not in df.columns:
            return df
        step_series = df["step_name"].fillna("unknown_step").astype(str)
        mask = ~step_series.isin(step_names_to_drop)
        return df.loc[mask].copy()

    def _build_from_kdd_if_available(self) -> pd.DataFrame | None:
        """
        Convierte archivos KDD a esquema simple del proyecto.
        """
        external_dir = self.config.external_data_dir
        if not external_dir.exists():
            return None

        train_files = sorted(external_dir.glob("*train*.txt"))
        test_files = sorted(external_dir.glob("*test*.txt"))
        candidates = train_files + test_files

        # Respaldo: usar master si no se detectan train/test.
        if not candidates:
            candidates = sorted(external_dir.glob("*master*.txt"))
        if not candidates:
            candidates = sorted(external_dir.glob("*.txt"))
        if not candidates:
            return None

        frames: list[pd.DataFrame] = []
        for path in candidates:
            raw = pd.read_csv(path, sep="\t", low_memory=False)
            split_name = self._detect_split_name(path.name)
            Logger.print(f"Archivo KDD detectado: {path.name} (split: {split_name}). Filas: {len(raw)}.")
            transformed = pd.DataFrame(
                {
                    "student_id": raw.get("Anon Student Id", raw.get("Student ID", "unknown_student")),
                    "step_name": raw.get("Step Name", "unknown_step"),
                    "incorrects": raw.get("Incorrects", 0),
                    "hints": raw.get("Hints", 0),
                    "correct_first_attempt": raw.get("Correct First Attempt", 0),
                    "step_duration_sec": raw.get("Step Duration (sec)", 0),
                    "source_split": split_name,
                    "timestamp": raw.get("First Transaction Time", raw.get("Transaction Time", "")),
                }
            )
            frames.append(transformed)
        if not frames:
            return None
        return pd.concat(frames, ignore_index=True)

    def _append_gameplay_logs(self, base_df: pd.DataFrame) -> pd.DataFrame:
        """
        Incorpora datos de uso real del minijuego para reentrenamiento.
        """
        log_path = self.config.gameplay_log_path
        if not log_path.exists():
            return base_df

        try:
            log_df = pd.read_csv(log_path, on_bad_lines="skip", engine="python")
        except Exception:
            return base_df
        needed = [
            "student_id",
            "step_name",
            "incorrects",
            "hints",
            "correct_first_attempt",
            "step_duration_sec",
            "source_split",
            "timestamp",
        ]
        for col in needed:
            if col not in log_df.columns:
                if col == "source_split":
                    log_df[col] = "user_gameplay"
                elif col == "timestamp":
                    log_df[col] = ""
                else:
                    log_df[col] = 0
        log_df = log_df[needed].copy()
        combined = pd.concat([base_df, log_df], ignore_index=True)
        return self.clean_data(combined)

    def _load_step_name_blacklist(self) -> set[str]:
        path = self.config.step_name_blacklist_path
        if not path.exists():
            return set()

        if path.suffix.lower() == ".csv":
            try:
                table = pd.read_csv(path)
            except Exception:
                return set()
            if table.empty:
                return set()
            if "step_name" in table.columns:
                values = table["step_name"]
            else:
                values = table.iloc[:, 0]
            return {str(v).strip() for v in values.dropna().tolist() if str(v).strip()}

        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except Exception:
            return set()
        return {line.strip() for line in lines if line.strip() and not line.strip().startswith("#")}

    def _apply_outlier_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        data = self._cap_column_quantile(data, "incorrects", self.config.incorrects_cap_quantile)
        data = self._cap_column_quantile(data, "hints", self.config.hints_cap_quantile)
        data = self._cap_column_quantile(
            data, "step_duration_sec", self.config.step_duration_cap_quantile
        )

        data = self._drop_column_quantile(data, "incorrects", self.config.incorrects_drop_quantile)
        data = self._drop_column_quantile(data, "hints", self.config.hints_drop_quantile)
        data = self._drop_column_quantile(
            data, "step_duration_sec", self.config.step_duration_drop_quantile
        )
        return data

    @staticmethod
    def _cap_column_percentile(
        df: pd.DataFrame,
        column: str,
        percentile: float = 0.999,
        *,
        positive_only: bool = True,
    ) -> pd.DataFrame:
        if df.empty or column not in df.columns:
            return df
        q = float(percentile)
        if not (0.0 < q < 1.0):
            return df
        series = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
        base = series[series > 0] if positive_only else series.dropna()
        if base.empty:
            return df
        threshold = float(base.quantile(q))
        out = df.copy()
        out[column] = series.clip(lower=0.0, upper=threshold)
        return out

    @staticmethod
    def _cap_column_quantile(df: pd.DataFrame, column: str, q: float | None) -> pd.DataFrame:
        if q is None or df.empty or column not in df.columns:
            return df
        q = float(q)
        if not (0.0 < q < 1.0):
            return df
        series = pd.to_numeric(df[column], errors="coerce")
        if series.dropna().empty:
            return df
        threshold = float(series.quantile(q))
        capped = series.clip(upper=threshold)
        out = df.copy()
        out[column] = capped
        return out

    @staticmethod
    def _drop_column_quantile(df: pd.DataFrame, column: str, q: float | None) -> pd.DataFrame:
        if q is None or df.empty or column not in df.columns:
            return df
        q = float(q)
        if not (0.0 < q < 1.0):
            return df
        series = pd.to_numeric(df[column], errors="coerce")
        if series.dropna().empty:
            return df
        threshold = float(series.quantile(q))
        return df.loc[series <= threshold].copy()

    @staticmethod
    def _detect_split_name(filename: str) -> str:
        lower_name = filename.lower()
        if "train" in lower_name:
            return "train"
        if "test" in lower_name:
            return "test"
        if "master" in lower_name:
            return "master"
        return "unknown"
