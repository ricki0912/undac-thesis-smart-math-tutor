from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODELS_DIR = Path("models")

app = FastAPI(
    title="API Tutor Inteligente de Matematicas",
    description="Prediccion de Correct First Attempt con modelos entrenados en KDD Cup 2010.",
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    duration: float = Field(..., description="Duracion del paso en segundos.")
    incorrects: float = Field(..., description="Numero de intentos incorrectos.")
    hints: float = Field(..., description="Numero de pistas usadas.")
    corrects: float = Field(..., description="Numero de aciertos previos.")
    skill: str = Field(..., description="Habilidad principal del paso.")


def _load_best_model() -> tuple[object, object]:
    meta_path = MODELS_DIR / "meta.pkl"
    if not meta_path.exists():
        raise FileNotFoundError(
            "No existe models/meta.pkl. Ejecuta primero: python main.py train"
        )

    meta = joblib.load(meta_path)
    best_name = meta.get("best")
    label_encoder = meta.get("le")
    if not best_name or label_encoder is None:
        raise ValueError("Archivo meta.pkl invalido.")

    model_path = MODELS_DIR / f"model_{best_name}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No se encontro el modelo seleccionado: {model_path}"
        )
    model = joblib.load(model_path)
    return model, label_encoder


@app.get("/")
def root() -> dict[str, str]:
    return {"mensaje": "API activa. Usa /docs para probar el endpoint /predict."}


@app.post("/predict")
def predict(payload: PredictionRequest) -> dict[str, float | int]:
    try:
        model, label_encoder = _load_best_model()
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Error cargando modelos: {exc}") from exc

    x_input = pd.DataFrame(
        [
            {
                "Step Duration (sec)": payload.duration,
                "Incorrects": payload.incorrects,
                "Hints": payload.hints,
                "Corrects": payload.corrects,
                "Skill": payload.skill,
                "Opportunity(SubSkills)": 0.0,
            }
        ]
    )

    try:
        pred_encoded = int(model.predict(x_input)[0])
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(x_input)[0][1])
        else:
            probability = float(pred_encoded)
        prediction = int(label_encoder.inverse_transform([pred_encoded])[0])
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=400, detail=f"Error en prediccion: {exc}") from exc

    return {"prediction": prediction, "probability": probability}
