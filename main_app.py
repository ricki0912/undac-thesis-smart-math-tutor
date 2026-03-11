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


def inject_player_styles() -> None:
    st.markdown(
        """
        <link
          rel="stylesheet"
          href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css"
          integrity="sha384-QWTKZyjpPEjISv5WaRU9OFeRpok6YctnYmDr5pNlyT2bRjXh0JMhjY6hW+ALEwIH"
          crossorigin="anonymous"
        />
        <style>
        .main .block-container {
            max-width: 980px;
            padding-top: 1rem;
        }
        .stButton > button, .stForm button {
            background-color: #0d6efd !important;
            color: #fff !important;
            border: 1px solid #0b5ed7 !important;
            border-radius: .5rem !important;
            font-weight: 600 !important;
        }
        .stButton > button:hover, .stForm button:hover {
            background-color: #0b5ed7 !important;
        }
        div[data-testid="stMetric"] {
            background: #0b1220;
            border: 1px solid #1f2937;
            border-radius: .5rem;
            padding: .6rem .8rem;
        }
        div[data-testid="stMetricLabel"] {
            color: #cbd5e1 !important;
        }
        div[data-testid="stMetricValue"] {
            color: #f8fafc !important;
        }
        .ia-banner {
            background: #0d6efd;
            color: #fff;
            border-radius: .5rem;
            padding: .75rem 1rem;
            text-align: center;
            font-weight: 700;
            margin-bottom: .75rem;
        }
        .game-title {
            border: 1px solid #ced4da;
            background: #f8f9fa;
            border-radius: .5rem;
            text-align: center;
            font-size: 1.9rem;
            font-weight: 700;
            padding: .45rem .75rem;
            margin-bottom: .75rem;
        }
        .game-shell {
            background: #eef5fb;
            border: 1px solid #60a5fa;
            border-radius: 18px;
            padding: 14px;
            margin-bottom: 12px;
        }
        .section-chip {
            background: #0d6efd;
            color: white;
            border-radius: .5rem;
            text-align: center;
            font-weight: 700;
            padding: .5rem .75rem;
            margin: .75rem 0;
        }
        .hint-box {
            background: #198754;
            color: white;
            border-radius: .4rem;
            padding: .55rem .75rem;
            margin-top: .4rem;
            margin-bottom: .4rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


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
        "latest_prediction": None,
        "latest_points": 0,
        "latest_recommendation": None,
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
    st.session_state.latest_prediction = None
    st.session_state.latest_points = 0
    st.session_state.latest_recommendation = None


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
        "latest_prediction": st.session_state.latest_prediction,
        "latest_points": st.session_state.latest_points,
        "latest_recommendation": st.session_state.latest_recommendation,
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
        raise ValueError("No hay pregunta activa.")
    cq = st.session_state.current_question
    return Question(text=cq["text"], answer=int(cq["answer"]), hint=cq["hint"], level=int(cq["level"]))


def load_next_question(simulator: StudentSimulator) -> None:
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


@st.dialog("Detalle del intento")
def show_attempt_detail_modal(detail: dict[str, Any]) -> None:
    st.markdown("#### Resumen")
    st.json(
        {
            "pregunta": detail.get("pregunta"),
            "respuesta_usuario": detail.get("respuesta_usuario"),
            "respuesta_correcta": detail.get("respuesta_correcta"),
            "prediccion_modelo": detail.get("prediccion_modelo"),
            "confianza_modelo": detail.get("confianza_modelo"),
            "accion_adaptativa": detail.get("accion_adaptativa"),
            "puntaje_ronda": detail.get("puntaje_ronda"),
        }
    )
    st.markdown("#### Parametros de entrada al modelo")
    st.json(detail.get("model_inputs", {}))
    st.markdown("#### Salida del modelo")
    st.json(detail.get("model_output", {}))


def render_player_interface(
    config: ProjectConfig,
    auth_store: AuthStore,
    loader: DataLoader,
    simulator: StudentSimulator,
    model: DifficultyModel,
) -> None:
    inject_player_styles()
    try:
        df = loader.load_dataset()
    except Exception as exc:
        st.error(f"No se pudo cargar dataset: {exc}")
        return

    st.subheader("Modo Jugador")
    student_id = st.text_input("ID estudiante", value=st.session_state.current_user)
    stats = load_student_stats(df, student_id)

    st.caption(
        f"Promedio historico -> errores {stats['avg_errors']:.2f}, "
        f"tiempo {stats['avg_time']:.2f}s, hints {stats['avg_hints']:.2f}"
    )
    st.caption(f"Banco actual de preguntas: {simulator.total_questions()} ejercicios.")

    st.markdown("<div class='game-shell'>", unsafe_allow_html=True)
    if st.session_state.get("latest_prediction"):
        st.markdown(
            f"<div class='ia-banner'>{build_prediction_message(st.session_state['latest_prediction'])}</div>",
            unsafe_allow_html=True,
        )
    st.markdown("<div class='game-title'>Pitágoras Aprende</div>", unsafe_allow_html=True)

    option_col, hint_col, next_col = st.columns([1, 1, 1])
    with option_col:
        with st.popover("OPCIONES"):
            if st.button("Guardar progreso", key="opt_save"):
                auth_store.save_progress(st.session_state.current_user, serialize_progress())
                st.success("Progreso guardado.")
            if st.button("Cargar progreso", key="opt_load"):
                saved = auth_store.load_progress(st.session_state.current_user)
                if saved:
                    load_progress_to_state(saved)
                    st.success("Progreso cargado.")
                    st.rerun()
                else:
                    st.warning("No hay progreso guardado.")
            if st.button("Reiniciar", key="opt_reset"):
                reset_player_state()
                auth_store.save_progress(st.session_state.current_user, serialize_progress())
                st.rerun()
            if st.button("Cerrar sesion", key="opt_logout"):
                logout()
                st.rerun()

    with hint_col:
        if st.button("PISTA", disabled=st.session_state.current_question is None):
            st.session_state.question_hints_used += 1
            st.session_state.show_hint = True

    with next_col:
        if st.button(">", help="Otra pregunta o siguiente"):
            load_next_question(simulator)
            st.info("Nueva pregunta cargada.")
            st.rerun()

    if st.session_state.current_question is None:
        st.info("Presiona > para cargar la siguiente pregunta.")
        question = None
    else:
        question = ensure_current_question(simulator)
        st.markdown(f"### Ejercicio actual (nivel {st.session_state.current_level})")
        st.code(question.text, language="text")

    if st.session_state.show_hint and question is not None:
        st.markdown(
            f"<div class='hint-box'>Pista: {question.hint}</div>",
            unsafe_allow_html=True,
        )

    if question is None:
        submit_answer = False
        answer_input = ""
    else:
        with st.form("answer_form", clear_on_submit=True):
            answer_input = st.text_input("Respuesta")
            submit_answer = st.form_submit_button("Responder", use_container_width=True)

    if not submit_answer or question is None:
        st.markdown("</div>", unsafe_allow_html=True)
        render_player_dashboard()
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
            "model_inputs": prediction["model_inputs"],
            "model_output": {
                "difficulty_level": prediction["difficulty_level"],
                "difficulty_label": prediction["difficulty_label"],
                "probability": prediction["probability"],
            },
        }
    )
    st.session_state.latest_prediction = prediction
    st.session_state.latest_points = points
    st.session_state.latest_recommendation = recommendation

    event_row = {
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
        "predicted_difficulty": prediction["difficulty_label"],
        "predicted_probability": prediction["probability"],
        "recommended_action": recommendation["action"],
        "recommended_reason": recommendation["reason"],
        "game_level_before": record["level"],
        "game_level_after": st.session_state.current_level,
        "attempt_number": attempt_number,
        "question_status": "resolved" if is_correct else "retry",
    }
    for key, value in prediction["model_inputs"].items():
        event_row[f"model_input_{key}"] = value
    append_row(config.gameplay_log_path, event_row)
    auth_store.save_progress(st.session_state.current_user, serialize_progress())
    st.info("Intento guardado. La prediccion del modelo y todos los parametros quedaron registrados.")

    # Solo libera para siguiente cuando la respuesta es correcta.
    if is_correct:
        st.session_state.current_question = None
        st.session_state.question_started_at = None
        st.session_state.question_attempts = 0
        st.session_state.question_hints_used = 0
        st.session_state.show_hint = False

    st.markdown("</div>", unsafe_allow_html=True)
    render_player_dashboard()


def render_player_dashboard() -> None:
    st.markdown("---")
    st.markdown("<div class='section-chip'>Prediccion Actual</div>", unsafe_allow_html=True)
    pred = st.session_state.get("latest_prediction")
    rec = st.session_state.get("latest_recommendation")
    points = st.session_state.get("latest_points", 0)

    p1, p2, p3 = st.columns(3)
    if pred:
        p1.metric("Dificultad predicha", str(pred.get("difficulty_label", "-")))
        p2.metric("Confianza", f"{float(pred.get('probability', 0.0)):.2f}")
    else:
        p1.metric("Dificultad predicha", "-")
        p2.metric("Confianza", "-")
    p3.metric("Puntos", points)
    st.progress(st.session_state.current_level / 5, text=f"Nivel actual: {st.session_state.current_level}/5")

    st.markdown("<div class='section-chip'>Avance</div>", unsafe_allow_html=True)
    a1, a2, a3, a4, a5 = st.columns(5)
    a1.metric("Ronda", st.session_state.round)
    a2.metric("Nivel", st.session_state.current_level)
    a3.metric("Puntaje", st.session_state.score)
    a4.metric("Racha", st.session_state.streak)
    a5.metric("Intentos pregunta", st.session_state.question_attempts)
    if rec:
        st.caption(f"Ajuste actual: {rec.get('action', '-')}. Motivo: {rec.get('reason', '-')}")

    history = st.session_state.get("history", [])
    if history:
        hist_df = pd.DataFrame(history)
        plot_df = hist_df.copy()
        plot_df["puntaje_acumulado"] = plot_df["puntaje_ronda"].cumsum()
        chart_df = plot_df[["ronda", "puntaje_acumulado", "nivel_despues"]].set_index("ronda")
        st.line_chart(chart_df, height=240)

    st.markdown("<div class='section-chip'>Historial</div>", unsafe_allow_html=True)
    if not history:
        st.info("Aun no hay intentos registrados.")
        return

    table_cols = [
        "ronda",
        "pregunta",
        "respuesta_usuario",
        "respuesta_correcta",
        "es_correcta",
        "nivel_antes",
        "nivel_despues",
        "prediccion_modelo",
        "confianza_modelo",
        "accion_adaptativa",
    ]
    st.dataframe(pd.DataFrame(history)[table_cols], use_container_width=True, height=260)

    st.markdown("### Ver detalles")
    recent = list(enumerate(history[-10:], start=max(0, len(history) - 10)))
    for idx, item in recent:
        c1, c2 = st.columns([4, 1])
        c1.write(f"Intento #{idx + 1}: {item.get('pregunta', '')}")
        if c2.button("Ver mas", key=f"view_more_{idx}"):
            show_attempt_detail_modal(item)


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
        st.success("Prediccion generada por IA.")
        st.info(build_prediction_message(prediction))
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
        summary_path = config.metrics_dir / "training_summary.json"
        if summary_path.exists():
            summary = pd.read_json(summary_path, typ="series")
            st.caption(
                f"Validacion usada: {summary.get('validation_type', 'no disponible')} | "
                f"Features: {summary.get('feature_count', 'n/a')}"
            )

    render_admin_manual_tester(model)

    st.markdown("#### Seguimiento de preguntas y predicciones")
    if config.gameplay_log_path.exists():
        logs = read_csv_safe(config.gameplay_log_path)
        st.write(f"Total de intentos registrados: **{len(logs)}**")
        base_cols = [
            "timestamp",
            "username",
            "question_text",
            "user_answer",
            "expected_answer",
            "is_correct",
            "predicted_difficulty",
            "predicted_probability",
            "recommended_action",
            "recommended_reason",
            "game_level_before",
            "game_level_after",
        ]
        input_cols = sorted([col for col in logs.columns if col.startswith("model_input_")])
        cols = base_cols[:6] + input_cols + base_cols[6:]
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

    st.write(
        f"Usuario activo: **{st.session_state.current_user}** | "
        f"Rol: **{st.session_state.current_role}**"
    )

    loader = DataLoader(config)
    simulator = StudentSimulator(seed=42)
    model = DifficultyModel(config)

    if st.session_state.current_role == "admin":
        if st.button("Cerrar sesion"):
            logout()
            st.rerun()
        render_admin_interface(config, model)
    else:
        render_player_interface(config, auth_store, loader, simulator, model)


if __name__ == "__main__":
    main()
