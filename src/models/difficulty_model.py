from __future__ import annotations

from typing import Any

import joblib
import pandas as pd

from src.data_processing.feature_engineering import FeatureEngineer
from src.utils.config import ProjectConfig
from src.utils.logger import Logger


class DifficultyModel:
    """
    Clase para cargar modelo entrenado y generar predicciones de dificultad.
    """

    label_map = {0: "baja", 1: "media", 2: "alta"}

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self.model = None
        self.engineer = FeatureEngineer()

    def load(self) -> None:
        if not self.config.model_path.exists():
            raise FileNotFoundError(
                f"No se encontro {self.config.model_path}. Ejecuta primero main_train.py"
            )
        Logger.print(f"Cargando modelo desde: {self.config.model_path}")
        self.model = joblib.load(self.config.model_path)

    def predict(self, record: dict[str, Any]) -> dict[str, Any]:
        if self.model is None:
            self.load()
        Logger.print("Ejecutando prediccion de dificultad...")

        base_df = pd.DataFrame([record])
        transformed = self.engineer.transform(base_df, include_target=False)
        x = transformed[self.engineer.feature_columns]
        model_inputs = x.iloc[0].to_dict()

        pred_level = int(self.model.predict(x)[0])
        proba = self._safe_probability(x)
        return {
            "difficulty_level": pred_level,
            "difficulty_label": self.label_map.get(pred_level, "desconocida"),
            "probability": proba,
            "model_inputs": {k: float(v) for k, v in model_inputs.items()},
        }

    def _safe_probability(self, x: pd.DataFrame) -> float:
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(x)[0]
            return float(max(probs))
        return 0.0
