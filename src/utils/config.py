from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectConfig:
    """Configuracion central del proyecto."""

    base_dir: Path = Path(__file__).resolve().parents[2]
    data_path: Path = base_dir / "data" / "dataset.csv"
    external_data_dir: Path = base_dir / "data" / "external"
    model_path: Path = base_dir / "models" / "model.pkl"
    leaderboard_path: Path = base_dir / "reports" / "metrics" / "model_leaderboard.csv"
    processed_path: Path = base_dir / "reports" / "metrics" / "processed_dataset.csv"
    figures_dir: Path = base_dir / "reports" / "figures"
    metrics_dir: Path = base_dir / "reports" / "metrics"
    logs_dir: Path = base_dir / "data" / "raw"
    gameplay_log_path: Path = logs_dir / "gameplay_logs.csv"
    assessment_log_path: Path = logs_dir / "assessment_logs.csv"
    users_path: Path = logs_dir / "users.csv"
    user_progress_path: Path = logs_dir / "user_progress.json"
    random_state: int = 42
    test_size: float = 0.2
