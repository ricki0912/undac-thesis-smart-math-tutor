from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from main_train import main as run_training_pipeline
from src.app.adaptive_engine import AdaptiveEngine
from src.app.auth_store import AuthStore
from src.app.student_simulator import Question, StudentSimulator
from src.data_processing.data_loader import DataLoader
from src.models.difficulty_model import DifficultyModel
from src.utils.config import ProjectConfig


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


def read_csv_safe(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, on_bad_lines="skip", engine="python")


def init_state() -> None:
    defaults = {
        "is_authenticated": False,
        "current_user": "",
        "current_role": "player",
        "current_level": 3,
        "score": 0,
        "streak": 0,
        "lives": 3,
        "round": 1,
        "history": [],
        "current_question": None,
        "question_started_at": None,
        "question_attempts": 0,
        "question_hints_used": 0,
        "show_hint": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def logout() -> None:
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    init_state()


def reset_player_state() -> None:
    st.session_state.current_level = 3
    st.session_state.score = 0
    st.session_state.streak = 0
    st.session_state.lives = 3
    st.session_state.round = 1
    st.session_state.history = []
    st.session_state.current_question = None
    st.session_state.question_started_at = None
    st.session_state.question_attempts = 0
    st.session_state.question_hints_used = 0
    st.session_state.show_hint = False


def serialize_progress() -> dict[str, Any]:
    return {
        "current_level": st.session_state.current_level,
        "score": st.session_state.score,
        "streak": st.session_state.streak,
        "lives": st.session_state.lives,
        "round": st.session_state.round,
        "history": st.session_state.history,
        "current_question": st.session_state.current_question,
        "question_started_at": st.session_state.question_started_at,
        "question_attempts": st.session_state.question_attempts,
        "question_hints_used": st.session_state.question_hints_used,
        "show_hint": st.session_state.show_hint,
    }


def load_progress_to_state(progress: dict[str, Any]) -> None:
    for key, value in progress.items():
        st.session_state[key] = value


def load_student_stats(df: pd.DataFrame, student_id: str) -> dict[str, float]:
    student_df = df[df["student_id"] == student_id]
    if student_df.empty:
        return {"avg_errors": 0.0, "avg_time": 60.0, "avg_hints": 0.0}
    return {
        "avg_errors": float(student_df["incorrects"].mean()),
        "avg_time": float(student_df["step_duration_sec"].mean()),
        "avg_hints": float(student_df["hints"].mean()),
    }


def compute_points(is_correct: bool, incorrects: int, hints: int, duration: float, action: str) -> int:
    points = 10
    if is_correct:
        points += 10
    points -= incorrects * 3
    points -= hints * 2
    if duration < 50:
        points += 5
    if action == "subir":
        points += 3
    return max(0, points)


def build_prediction_message(prediction: dict[str, Any]) -> str:
    label = str(prediction["difficulty_label"]).lower()
    prob = float(prediction["probability"])
    if label == "baja":
        level_msg = "facil"
    elif label == "media":
        level_msg = "intermedio"
    else:
        level_msg = "retador"
    return (
        f"La IA estima que este ejercicio es **{level_msg}** "
        f"(confianza: {prob:.0%})."
    )


def render_auth(auth_store: AuthStore) -> None:
    st.subheader("Acceso")
    tab_login, tab_register = st.tabs(["Iniciar sesion", "Registrarse"])

    with tab_login:
        login_user = st.text_input("Usuario", key="login_user")
        login_pass = st.text_input("Contrasena", type="password", key="login_pass")
        if st.button("Entrar", type="primary"):
            ok, msg, role = auth_store.login(login_user, login_pass)
            if ok:
                st.session_state.is_authenticated = True
                st.session_state.current_user = login_user.strip().lower()
                st.session_state.current_role = role
                if role == "player":
                    saved = auth_store.load_progress(st.session_state.current_user)
                    if saved:
                        load_progress_to_state(saved)
                st.success("Sesion iniciada.")
                st.rerun()
            else:
                st.error(msg)

    with tab_register:
        reg_user = st.text_input("Nuevo usuario", key="reg_user")
        reg_pass = st.text_input("Nueva contrasena", type="password", key="reg_pass")
        reg_role = st.selectbox("Rol", ["player", "admin"])
        if st.button("Crear cuenta"):
            ok, msg = auth_store.register_user(reg_user, reg_pass, role=reg_role)
            if ok:
                st.success(msg)
            else:
                st.error(msg)


def ensure_current_question(simulator: StudentSimulator) -> Question:
    if st.session_state.current_question is None:
        q = simulator.get_question_for_level(st.session_state.current_level)
        st.session_state.current_question = {
            "text": q.text,
            "answer": q.answer,
            "hint": q.hint,
            "level": q.level,
        }
        st.session_state.question_started_at = datetime.utcnow().isoformat()
        st.session_state.question_attempts = 0
        st.session_state.question_hints_used = 0
        st.session_state.show_hint = False

    cq = st.session_state.current_question
    return Question(text=cq["text"], answer=int(cq["answer"]), hint=cq["hint"], level=int(cq["level"]))


def render_player_interface(
    config: ProjectConfig,
    auth_store: AuthStore,
    loader: DataLoader,
    simulator: StudentSimulator,
    model: DifficultyModel,
) -> None:
    try:
        df = loader.load_dataset()
    except Exception as exc:
        st.error(f"No se pudo cargar dataset: {exc}")
        return

    st.subheader("Modo Jugador")
    left, right = st.columns([1, 2])
    with left:
        student_id = st.text_input("ID estudiante", value=st.session_state.current_user)
        if st.button("Guardar progreso"):
            auth_store.save_progress(st.session_state.current_user, serialize_progress())
            st.success("Progreso guardado.")
        if st.button("Cargar progreso"):
            saved = auth_store.load_progress(st.session_state.current_user)
            if saved:
                load_progress_to_state(saved)
                st.success("Progreso cargado.")
                st.rerun()
            else:
                st.warning("No hay progreso guardado.")
        if st.button("Reiniciar partida"):
            reset_player_state()
            auth_store.save_progress(st.session_state.current_user, serialize_progress())
            st.rerun()

    with right:
        stats = load_student_stats(df, student_id)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Ronda", st.session_state.round)
        c2.metric("Nivel", st.session_state.current_level)
        c3.metric("Puntaje", st.session_state.score)
        c4.metric("Racha", st.session_state.streak)
        c5.metric("Intentos pregunta", st.session_state.question_attempts)
        st.caption(
            f"Promedio historico -> errores {stats['avg_errors']:.2f}, "
            f"tiempo {stats['avg_time']:.2f}s, hints {stats['avg_hints']:.2f}"
        )
        st.caption(f"Banco actual de preguntas: {simulator.total_questions()} ejercicios.")

    question = ensure_current_question(simulator)
    st.markdown(f"### Ejercicio actual (nivel {st.session_state.current_level})")
    st.code(question.text, language="text")

    hint_col, change_col, _ = st.columns([1, 1, 2])
    with hint_col:
        if st.button("Pedir pista"):
            st.session_state.question_hints_used += 1
            st.session_state.show_hint = True
    with change_col:
        if st.button("Otra pregunta"):
            st.session_state.current_question = None
            st.session_state.question_started_at = None
            st.session_state.question_attempts = 0
            st.session_state.question_hints_used = 0
            st.session_state.show_hint = False
            st.info("Se cambio la pregunta actual por una nueva del mismo nivel.")
            st.rerun()
    if st.session_state.show_hint:
        st.info(f"Pista: {question.hint}")

    with st.form("answer_form", clear_on_submit=True):
        answer_input = st.text_input("Ingresa el valor de la variable (x, y o a segun ejercicio)")
        submit_answer = st.form_submit_button("Enviar respuesta")

    if not submit_answer:
        return

    started_at = datetime.fromisoformat(st.session_state.question_started_at)
    elapsed = max(1.0, (datetime.utcnow() - started_at).total_seconds())
    st.session_state.question_attempts += 1
    is_correct = simulator.is_correct_answer(answer_input, question.answer)
    attempt_number = st.session_state.question_attempts
    hints_used = st.session_state.question_hints_used

    if is_correct:
        incorrects = attempt_number - 1
        correct_first_attempt = 1 if attempt_number == 1 else 0
    else:
        incorrects = attempt_number
        correct_first_attempt = 0

    record = {
        "student_id": student_id,
        "step_name": question.text,
        "incorrects": int(incorrects),
        "hints": int(hints_used),
        "correct_first_attempt": int(correct_first_attempt),
        "step_duration_sec": float(elapsed),
        "level": int(st.session_state.current_level),
    }

    try:
        prediction = model.predict(record)
        st.success("Modelo utilizado: se calculo la dificultad para este intento.")
        st.toast("Prediccion generada por el modelo", icon="🤖")
    except Exception as exc:
        st.error(f"No se pudo usar el modelo. Ejecuta `py main_train.py`. Detalle: {exc}")
        return

    st.info(build_prediction_message(prediction))

    engine = AdaptiveEngine(initial_level=st.session_state.current_level)
    recommendation = engine.recommend_next_level(
        errors=int(record["incorrects"]),
        time_spent=float(record["step_duration_sec"]),
        average_time=float(stats["avg_time"]),
        correct_first_attempt=int(record["correct_first_attempt"]),
        predicted_difficulty=str(prediction["difficulty_label"]),
        predicted_probability=float(prediction["probability"]),
    )

    if is_correct:
        st.session_state.current_level = int(recommendation["next_level"])
        points = compute_points(
            is_correct=True,
            incorrects=int(record["incorrects"]),
            hints=int(record["hints"]),
            duration=float(record["step_duration_sec"]),
            action=recommendation["action"],
        )
        st.session_state.score += points
        st.session_state.streak += 1
        st.session_state.round += 1
        st.success("Respuesta correcta. Pasas a la siguiente pregunta.")
        st.caption(f"Ajuste de nivel: {recommendation['reason']}")
    else:
        points = 0
        st.session_state.streak = 0
        st.warning("Respuesta incorrecta. Intenta nuevamente la misma pregunta.")
        st.caption(f"Sugerencia IA/reglas: {recommendation['reason']}")

    st.session_state.history.append(
        {
            "ronda": st.session_state.round - 1,
            "pregunta": question.text,
            "respuesta_usuario": answer_input,
            "respuesta_correcta": question.answer,
            "es_correcta": int(is_correct),
            "nivel_antes": record["level"],
            "nivel_despues": st.session_state.current_level,
            "prediccion_modelo": prediction["difficulty_label"],
            "confianza_modelo": prediction["probability"],
            "accion_adaptativa": recommendation["action"],
            "puntaje_ronda": points,
        }
    )

    append_row(
        config.gameplay_log_path,
        {
            "timestamp": datetime.utcnow().isoformat(),
            "username": st.session_state.current_user,
            "student_id": record["student_id"],
            "question_text": question.text,
            "user_answer": answer_input,
            "expected_answer": question.answer,
            "is_correct": int(is_correct),
            "step_name": record["step_name"],
            "incorrects": record["incorrects"],
            "hints": record["hints"],
            "correct_first_attempt": record["correct_first_attempt"],
            "step_duration_sec": record["step_duration_sec"],
            "source_split": "user_gameplay",
            "model_input_incorrects": prediction["model_inputs"]["incorrects"],
            "model_input_hints": prediction["model_inputs"]["hints"],
            "model_input_step_duration_sec": prediction["model_inputs"]["step_duration_sec"],
            "model_input_correct_first_attempt": prediction["model_inputs"]["correct_first_attempt"],
            "model_input_error_rate": prediction["model_inputs"]["error_rate"],
            "model_input_time_efficiency": prediction["model_inputs"]["time_efficiency"],
            "model_input_difficulty_score": prediction["model_inputs"]["difficulty_score"],
            "predicted_difficulty": prediction["difficulty_label"],
            "predicted_probability": prediction["probability"],
            "recommended_action": recommendation["action"],
            "recommended_reason": recommendation["reason"],
            "game_level_before": record["level"],
            "game_level_after": st.session_state.current_level,
            "attempt_number": attempt_number,
            "question_status": "resolved" if is_correct else "retry",
        },
    )
    auth_store.save_progress(st.session_state.current_user, serialize_progress())
    st.info("Intento guardado. La prediccion del modelo y todos los parametros quedaron registrados.")

    r1, r2, r3 = st.columns(3)
    r1.metric("Dificultad predicha", prediction["difficulty_label"])
    r2.metric("Confianza", f"{prediction['probability']:.2f}")
    r3.metric("Puntos", points)
    st.progress(st.session_state.current_level / 5, text=f"Nivel actual: {st.session_state.current_level}/5")

    # Solo cambia de pregunta cuando la respuesta es correcta.
    if is_correct:
        st.session_state.current_question = None
        st.session_state.question_started_at = None
        st.session_state.question_attempts = 0
        st.session_state.question_hints_used = 0
        st.session_state.show_hint = False

    st.markdown("### Historial de partida")
    st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True, height=280)


def render_admin_manual_tester(model: DifficultyModel) -> None:
    st.markdown("#### Prueba manual del modelo")
    with st.form("manual_model_test", clear_on_submit=False):
        duration = st.number_input("Duracion (seg)", min_value=1.0, value=60.0, step=1.0)
        incorrects = st.number_input("Errores", min_value=0, value=1, step=1)
        hints = st.number_input("Hints", min_value=0, value=0, step=1)
        correct_first_attempt = st.selectbox("Correct First Attempt", [1, 0])
        step_name = st.text_input("Pregunta/step", value="2x + 3 = 11")
        run_test = st.form_submit_button("Probar modelo")

    if run_test:
        test_record = {
            "student_id": "admin_probe",
            "step_name": step_name,
            "incorrects": int(incorrects),
            "hints": int(hints),
            "correct_first_attempt": int(correct_first_attempt),
            "step_duration_sec": float(duration),
            "level": 3,
        }
        try:
            prediction = model.predict(test_record)
        except Exception as exc:
            st.error(f"No se pudo ejecutar prueba manual. Detalle: {exc}")
            return
        st.success("Prediccion generada.")
        st.json(
            {
                "predicted_difficulty": prediction["difficulty_label"],
                "probability": prediction["probability"],
                "model_inputs": prediction["model_inputs"],
            }
        )


def render_admin_interface(config: ProjectConfig, model: DifficultyModel) -> None:
    st.subheader("Modo Administrador")
    c1, c2 = st.columns([1, 2])
    with c1:
        if st.button("Reentrenar modelo", type="primary"):
            with st.spinner("Entrenando..."):
                run_training_pipeline()
            st.success("Modelo actualizado.")
        st.caption("Reentrena cuando haya nuevos logs de juego.")
    with c2:
        if config.leaderboard_path.exists():
            leaderboard = pd.read_csv(config.leaderboard_path)
            st.markdown("#### Rendimiento de modelos")
            st.dataframe(leaderboard, use_container_width=True, height=220)
        else:
            st.warning("No existe leaderboard aun.")

    render_admin_manual_tester(model)

    st.markdown("#### Seguimiento de preguntas y predicciones")
    if config.gameplay_log_path.exists():
        logs = read_csv_safe(config.gameplay_log_path)
        st.write(f"Total de intentos registrados: **{len(logs)}**")
        cols = [
            "timestamp",
            "username",
            "question_text",
            "user_answer",
            "expected_answer",
            "is_correct",
            "model_input_incorrects",
            "model_input_hints",
            "model_input_step_duration_sec",
            "model_input_correct_first_attempt",
            "model_input_error_rate",
            "model_input_time_efficiency",
            "model_input_difficulty_score",
            "predicted_difficulty",
            "predicted_probability",
            "recommended_action",
            "recommended_reason",
            "game_level_before",
            "game_level_after",
        ]
        available_cols = [col for col in cols if col in logs.columns]
        st.dataframe(logs[available_cols].tail(200), use_container_width=True, height=320)
    else:
        st.info("Aun no hay intentos registrados.")


def main() -> None:
    st.set_page_config(page_title="Juego adaptativo", page_icon="📘", layout="wide")
    st.title("Juego adaptativo con seguimiento del modelo")
    st.caption("Jugador: responde preguntas reales. Administrador: monitorea y prueba el modelo.")

    init_state()
    config = ProjectConfig()
    auth_store = AuthStore(config)

    if not st.session_state.is_authenticated:
        render_auth(auth_store)
        return

    t1, t2 = st.columns([3, 1])
    with t1:
        st.write(
            f"Usuario activo: **{st.session_state.current_user}** | "
            f"Rol: **{st.session_state.current_role}**"
        )
    with t2:
        if st.button("Cerrar sesion"):
            logout()
            st.rerun()

    loader = DataLoader(config)
    simulator = StudentSimulator(seed=42)
    model = DifficultyModel(config)

    if st.session_state.current_role == "admin":
        render_admin_interface(config, model)
    else:
        render_player_interface(config, auth_store, loader, simulator, model)


if __name__ == "__main__":
    main()
