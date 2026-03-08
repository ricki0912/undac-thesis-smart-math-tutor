from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.config import ProjectConfig


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
        # 1) Fuente principal: KDD externo (mismo nivel del proyecto)
        external_df = self._build_from_kdd_if_available()
        if external_df is not None:
            cleaned = self.clean_data(external_df)
            cleaned = self._append_gameplay_logs(cleaned)
            self.config.data_path.parent.mkdir(parents=True, exist_ok=True)
            cleaned.to_csv(self.config.data_path, index=False)
            return cleaned

        # 2) Respaldo: CSV local en data/dataset.csv
        if self.config.data_path.exists():
            df = pd.read_csv(self.config.data_path)
            cleaned = self.clean_data(df)
            return self._append_gameplay_logs(cleaned)

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
        return df

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
            transformed = pd.DataFrame(
                {
                    "student_id": raw.get("Anon Student Id", raw.get("Student ID", "unknown_student")),
                    "step_name": raw.get("Step Name", "unknown_step"),
                    "incorrects": raw.get("Incorrects", 0),
                    "hints": raw.get("Hints", 0),
                    "correct_first_attempt": raw.get("Correct First Attempt", 0),
                    "step_duration_sec": raw.get("Step Duration (sec)", 0),
                    "source_split": split_name,
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
        ]
        for col in needed:
            if col not in log_df.columns:
                log_df[col] = "user_gameplay" if col == "source_split" else 0
        log_df = log_df[needed].copy()
        combined = pd.concat([base_df, log_df], ignore_index=True)
        return self.clean_data(combined)

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
