from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from main_train import main as run_training_pipeline
from src.app.adaptive_engine import AdaptiveEngine
from src.app.auth_store import AuthStore
from src.app.student_simulator import Question, StudentSimulator
from src.data_processing.data_loader import DataLoader
from src.models.difficulty_model import DifficultyModel
from src.utils.config import ProjectConfig

config = ProjectConfig()
auth_store = AuthStore(config)
simulator = StudentSimulator(seed=42)
model = DifficultyModel(config)
loader = DataLoader(config)

WEB_DIR = Path("web")
REPORTS_DIR = Path("reports")
SESSIONS: dict[str, dict[str, Any]] = {}


def init_player_state() -> dict[str, Any]:
    return {
        "current_level": 3,
        "score": 0,
        "streak": 0,
        "round": 1,
        "history": [],
        "current_question": None,
        "question_started_at": None,
        "question_attempts": 0,
        "question_hints_used": 0,
        "show_hint": False,
        "latest_prediction": None,
        "latest_points": 0,
        "latest_recommendation": None,
    }


def append_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row_df = pd.DataFrame([row])
    if path.exists():
        try:
            existing_cols = pd.read_csv(path, nrows=0).columns.tolist()
            if existing_cols:
                for col in existing_cols:
                    if col not in row_df.columns:
                        row_df[col] = ""
                row_df = row_df[existing_cols]
        except Exception:
            pass
        row_df.to_csv(path, mode="a", header=False, index=False)
    else:
        row_df.to_csv(path, index=False)


@lru_cache(maxsize=1)
def cached_dataset() -> pd.DataFrame:
    return loader.load_dataset()


def student_avg_time(student_id: str) -> float:
    df = cached_dataset()
    student_df = df[df["student_id"].astype(str) == str(student_id)]
    if student_df.empty:
        return 60.0
    return float(student_df["step_duration_sec"].mean())


def ensure_session(username: str) -> dict[str, Any]:
    key = username.strip().lower()
    if key not in SESSIONS:
        saved = auth_store.load_progress(key)
        SESSIONS[key] = saved if saved else init_player_state()
    return SESSIONS[key]


def build_prediction_message(prediction: dict[str, Any]) -> str:
    label = str(prediction["difficulty_label"]).lower()
    prob = float(prediction["probability"])
    if label == "baja":
        level_msg = "facil"
    elif label == "media":
        level_msg = "intermedio"
    else:
        level_msg = "retador"
    return f"La IA estima que este ejercicio es {level_msg} (confianza: {prob:.0%})."


app = FastAPI(title="Smart Math Tutor Local API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if WEB_DIR.exists():
    app.mount("/web", StaticFiles(directory=str(WEB_DIR)), name="web")
if REPORTS_DIR.exists():
    app.mount("/reports", StaticFiles(directory=str(REPORTS_DIR)), name="reports")


class RegisterPayload(BaseModel):
    username: str
    password: str
    role: str = "player"


class LoginPayload(BaseModel):
    username: str
    password: str


class UserPayload(BaseModel):
    username: str


class SubmitPayload(BaseModel):
    username: str
    student_id: str
    answer: str


class AdminProbePayload(BaseModel):
    duration: float
    incorrects: int
    hints: int
    correct_first_attempt: int
    step_name: str


@app.get("/", response_model=None)
def root():
    index_path = WEB_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "Frontend no encontrado. Crea /web/index.html"}


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "mode": "local"}


@app.post("/api/auth/register")
def register(payload: RegisterPayload) -> dict[str, Any]:
    ok, msg = auth_store.register_user(payload.username, payload.password, payload.role)
    return {"ok": ok, "message": msg}


@app.post("/api/auth/login")
def login(payload: LoginPayload) -> dict[str, Any]:
    ok, msg, role = auth_store.login(payload.username, payload.password)
    if not ok:
        return {"ok": False, "message": msg}
    session = ensure_session(payload.username)
    return {"ok": True, "message": msg, "role": role, "state": session}


@app.post("/api/player/save-progress")
def save_progress(payload: UserPayload) -> dict[str, Any]:
    session = ensure_session(payload.username)
    auth_store.save_progress(payload.username, session)
    return {"ok": True, "message": "Progreso guardado."}


@app.post("/api/player/load-progress")
def load_progress(payload: UserPayload) -> dict[str, Any]:
    session = ensure_session(payload.username)
    return {"ok": True, "state": session}


@app.post("/api/player/reset-progress")
def reset_progress(payload: UserPayload) -> dict[str, Any]:
    key = payload.username.strip().lower()
    SESSIONS[key] = init_player_state()
    auth_store.save_progress(payload.username, SESSIONS[key])
    return {"ok": True, "message": "Progreso reiniciado.", "state": SESSIONS[key]}


@app.post("/api/auth/logout")
def logout(payload: UserPayload) -> dict[str, Any]:
    key = payload.username.strip().lower()
    if key in SESSIONS:
        auth_store.save_progress(payload.username, SESSIONS[key])
        del SESSIONS[key]
    return {"ok": True, "message": "Sesion cerrada."}


@app.post("/api/game/next-question")
def next_question(payload: UserPayload) -> dict[str, Any]:
    session = ensure_session(payload.username)
    q = simulator.get_question_for_level(int(session["current_level"]))
    session["current_question"] = {
        "text": q.text,
        "answer": q.answer,
        "hint": q.hint,
        "level": q.level,
    }
    session["question_started_at"] = datetime.utcnow().isoformat()
    session["question_attempts"] = 0
    session["question_hints_used"] = 0
    session["show_hint"] = False
    auth_store.save_progress(payload.username, session)
    return {"ok": True, "question": session["current_question"], "state": session}


@app.post("/api/game/hint")
def hint(payload: UserPayload) -> dict[str, Any]:
    session = ensure_session(payload.username)
    if session["current_question"] is None:
        return {"ok": False, "message": "No hay pregunta activa."}
    session["question_hints_used"] += 1
    session["show_hint"] = True
    auth_store.save_progress(payload.username, session)
    return {"ok": True, "hint": session["current_question"]["hint"], "state": session}


@app.post("/api/game/submit")
def submit_answer(payload: SubmitPayload) -> dict[str, Any]:
    session = ensure_session(payload.username)
    q_raw = session.get("current_question")
    if q_raw is None:
        raise HTTPException(status_code=400, detail="No hay pregunta activa.")

    question = Question(
        text=q_raw["text"],
        answer=int(q_raw["answer"]),
        hint=q_raw["hint"],
        level=int(q_raw["level"]),
    )

    started_at = datetime.fromisoformat(session["question_started_at"])
    elapsed = max(1.0, (datetime.utcnow() - started_at).total_seconds())
    session["question_attempts"] += 1
    attempt_number = int(session["question_attempts"])
    hints_used = int(session["question_hints_used"])

    is_correct = simulator.is_correct_answer(payload.answer, question.answer)
    if is_correct:
        incorrects = attempt_number - 1
        correct_first_attempt = 1 if attempt_number == 1 else 0
    else:
        incorrects = attempt_number
        correct_first_attempt = 0

    record = {
        "student_id": payload.student_id,
        "step_name": question.text,
        "incorrects": int(incorrects),
        "hints": int(hints_used),
        "correct_first_attempt": int(correct_first_attempt),
        "step_duration_sec": float(elapsed),
        "level": int(session["current_level"]),
    }

    prediction = model.predict(record)
    avg_time = student_avg_time(payload.student_id)
    engine = AdaptiveEngine(initial_level=int(session["current_level"]))
    recommendation = engine.recommend_next_level(
        errors=int(record["incorrects"]),
        time_spent=float(record["step_duration_sec"]),
        average_time=float(avg_time),
        correct_first_attempt=int(record["correct_first_attempt"]),
        predicted_difficulty=str(prediction["difficulty_label"]),
        predicted_probability=float(prediction["probability"]),
    )

    points = 0
    if is_correct:
        session["current_level"] = int(recommendation["next_level"])
        points = max(0, 20 - int(record["incorrects"]) * 3 - int(record["hints"]) * 2)
        session["score"] = int(session["score"]) + points
        session["streak"] = int(session["streak"]) + 1
        session["round"] = int(session["round"]) + 1
        session["current_question"] = None
        session["question_started_at"] = None
        session["question_attempts"] = 0
        session["question_hints_used"] = 0
        session["show_hint"] = False
    else:
        session["streak"] = 0

    detail = {
        "ronda": int(session["round"]) - (1 if is_correct else 0),
        "pregunta": question.text,
        "respuesta_usuario": payload.answer,
        "respuesta_correcta": question.answer,
        "es_correcta": int(is_correct),
        "nivel_antes": record["level"],
        "nivel_despues": int(session["current_level"]),
        "prediccion_modelo": prediction["difficulty_label"],
        "confianza_modelo": prediction["probability"],
        "accion_adaptativa": recommendation["action"],
        "puntaje_ronda": points,
        "model_inputs": prediction["model_inputs"],
        "model_output": {
            "difficulty_level": prediction["difficulty_level"],
            "difficulty_label": prediction["difficulty_label"],
            "probability": prediction["probability"],
        },
    }
    session["history"].append(detail)
    session["latest_prediction"] = prediction
    session["latest_points"] = points
    session["latest_recommendation"] = recommendation

    event_row = {
        "timestamp": datetime.utcnow().isoformat(),
        "username": payload.username.lower(),
        "student_id": payload.student_id,
        "question_text": question.text,
        "user_answer": payload.answer,
        "expected_answer": question.answer,
        "is_correct": int(is_correct),
        "step_name": record["step_name"],
        "incorrects": record["incorrects"],
        "hints": record["hints"],
        "correct_first_attempt": record["correct_first_attempt"],
        "step_duration_sec": record["step_duration_sec"],
        "source_split": "user_gameplay",
        "predicted_difficulty": prediction["difficulty_label"],
        "predicted_probability": prediction["probability"],
        "recommended_action": recommendation["action"],
        "recommended_reason": recommendation["reason"],
        "game_level_before": record["level"],
        "game_level_after": session["current_level"],
        "attempt_number": attempt_number,
        "question_status": "resolved" if is_correct else "retry",
    }
    for key, value in prediction["model_inputs"].items():
        event_row[f"model_input_{key}"] = value
    append_row(config.gameplay_log_path, event_row)
    auth_store.save_progress(payload.username, session)

    return {
        "ok": True,
        "message": build_prediction_message(prediction),
        "is_correct": is_correct,
        "prediction": prediction,
        "recommendation": recommendation,
        "state": session,
    }


@app.get("/api/admin/summary")
def admin_summary() -> dict[str, Any]:
    leaderboard = []
    if config.leaderboard_path.exists():
        leaderboard = pd.read_csv(config.leaderboard_path).to_dict(orient="records")
    logs = []
    if config.gameplay_log_path.exists():
        logs = pd.read_csv(config.gameplay_log_path, on_bad_lines="skip", engine="python").tail(200).to_dict(
            orient="records"
        )
    return {"ok": True, "leaderboard": leaderboard, "logs": logs}


@app.get("/api/admin/figures")
def admin_figures() -> dict[str, Any]:
    config.figures_dir.mkdir(parents=True, exist_ok=True)
    items = []
    for file_path in sorted(config.figures_dir.glob("*.png")):
        items.append(
            {
                "name": file_path.name,
                "url": f"/reports/figures/{file_path.name}",
            }
        )
    return {"ok": True, "figures": items}


@app.post("/api/admin/test-model")
def admin_test_model(payload: AdminProbePayload) -> dict[str, Any]:
    record = {
        "student_id": "admin_probe",
        "step_name": payload.step_name,
        "incorrects": payload.incorrects,
        "hints": payload.hints,
        "correct_first_attempt": payload.correct_first_attempt,
        "step_duration_sec": payload.duration,
        "level": 3,
    }
    prediction = model.predict(record)
    return {
        "ok": True,
        "message": build_prediction_message(prediction),
        "prediction": prediction,
    }


@app.post("/api/admin/retrain")
def admin_retrain() -> dict[str, Any]:
    cached_dataset.cache_clear()
    run_training_pipeline()
    return {"ok": True, "message": "Modelo reentrenado localmente."}
