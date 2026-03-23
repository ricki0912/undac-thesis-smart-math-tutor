from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OverallMetrics:
    attempts: int
    unique_users: int
    unique_students: int
    accuracy: float
    avg_time_sec: float
    avg_hints: float
    avg_incorrects: float
    help_dependency: float
    avg_attempts_to_resolve: float
    oscillation_rate: float


def _safe_mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce")
    if values.dropna().empty:
        return 0.0
    return float(values.mean())


def _oscillation_rate(level_after: pd.Series) -> float:
    """
    % de veces que sube (+1 o mas) y luego baja (-1 o mas) en <=2 intentos.
    """
    vals = pd.to_numeric(level_after, errors="coerce").dropna().astype(float)
    if len(vals) < 4:
        return 0.0
    deltas = vals.diff()
    up = deltas > 0
    down_soon = (deltas.shift(-1) < 0) | (deltas.shift(-2) < 0)
    count_up = int(up.sum())
    if count_up == 0:
        return 0.0
    return float((up & down_soon).sum() / count_up)


def _half_delta(metric: pd.Series) -> float:
    vals = pd.to_numeric(metric, errors="coerce").dropna()
    if len(vals) < 6:
        return 0.0
    mid = max(1, len(vals) // 2)
    first = vals.iloc[:mid].mean()
    second = vals.iloc[mid:].mean()
    return float(second - first)


def compute_learning_metrics_from_logs(log_path: Path) -> dict[str, Any]:
    if not log_path.exists():
        return {"overall": None, "per_user": []}

    try:
        df = pd.read_csv(log_path, on_bad_lines="skip", engine="python")
    except Exception:
        return {"overall": None, "per_user": []}

    if df.empty:
        return {"overall": None, "per_user": []}

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df.get("timestamp"), errors="coerce")
    df["username"] = df.get("username", "").fillna("").astype(str).str.lower()
    df["student_id"] = df.get("student_id", "").fillna("").astype(str)
    df["is_correct"] = pd.to_numeric(df.get("is_correct"), errors="coerce").fillna(0).astype(int)

    df["incorrects"] = pd.to_numeric(df.get("incorrects"), errors="coerce").fillna(0).clip(lower=0)
    df["hints"] = pd.to_numeric(df.get("hints"), errors="coerce").fillna(0).clip(lower=0)
    df["step_duration_sec"] = pd.to_numeric(df.get("step_duration_sec"), errors="coerce").fillna(0).clip(lower=0)
    df["attempt_number"] = pd.to_numeric(df.get("attempt_number"), errors="coerce").fillna(1).clip(lower=1)
    df["question_status"] = df.get("question_status", "").fillna("").astype(str)
    df["game_level_after"] = pd.to_numeric(df.get("game_level_after"), errors="coerce")

    attempts = int(len(df))
    unique_users = int(df["username"].replace("", np.nan).nunique(dropna=True))
    unique_students = int(df["student_id"].replace("", np.nan).nunique(dropna=True))
    accuracy = _safe_mean(df["is_correct"])
    avg_time = _safe_mean(df["step_duration_sec"])
    avg_hints = _safe_mean(df["hints"])
    avg_incorrects = _safe_mean(df["incorrects"])

    correct_df = df[df["is_correct"] == 1]
    help_dependency = 0.0
    if not correct_df.empty:
        help_dependency = float((correct_df["hints"] > 0).mean())

    resolved = df[df["question_status"].astype(str).str.lower() == "resolved"]
    avg_attempts_to_resolve = _safe_mean(resolved["attempt_number"]) if not resolved.empty else 0.0

    oscillation_rate = 0.0
    if df["game_level_after"].notna().any():
        oscillation_rate = _oscillation_rate(df.sort_values("timestamp")["game_level_after"])

    overall = OverallMetrics(
        attempts=attempts,
        unique_users=unique_users,
        unique_students=unique_students,
        accuracy=accuracy,
        avg_time_sec=avg_time,
        avg_hints=avg_hints,
        avg_incorrects=avg_incorrects,
        help_dependency=help_dependency,
        avg_attempts_to_resolve=avg_attempts_to_resolve,
        oscillation_rate=oscillation_rate,
    )

    per_user: list[dict[str, Any]] = []
    for username, group in df.groupby("username"):
        if not username:
            continue
        g = group.sort_values("timestamp")
        user_accuracy = _safe_mean(g["is_correct"])
        user_avg_time = _safe_mean(g["step_duration_sec"])
        user_avg_hints = _safe_mean(g["hints"])
        user_avg_incorrects = _safe_mean(g["incorrects"])
        user_help_dependency = 0.0
        correct_g = g[g["is_correct"] == 1]
        if not correct_g.empty:
            user_help_dependency = float((correct_g["hints"] > 0).mean())

        per_user.append(
            {
                "username": username,
                "attempts": int(len(g)),
                "accuracy": user_accuracy,
                "avg_time_sec": user_avg_time,
                "avg_hints": user_avg_hints,
                "avg_incorrects": user_avg_incorrects,
                "help_dependency": user_help_dependency,
                "avg_attempts_to_resolve": _safe_mean(
                    g[g["question_status"].astype(str).str.lower() == "resolved"]["attempt_number"]
                ),
                "oscillation_rate": _oscillation_rate(g["game_level_after"]),
                "delta_accuracy_2half": _half_delta(g["is_correct"]),
                "delta_time_2half": _half_delta(g["step_duration_sec"]),
                "delta_hints_2half": _half_delta(g["hints"]),
                "delta_incorrects_2half": _half_delta(g["incorrects"]),
            }
        )

    per_user.sort(key=lambda row: (-float(row["attempts"]), row["username"]))
    per_user = per_user[:50]

    return {"overall": asdict(overall), "per_user": per_user}

