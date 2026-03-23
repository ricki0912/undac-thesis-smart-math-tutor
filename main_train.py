from __future__ import annotations

import argparse
import json

from src.data_processing.data_loader import DataLoader
from src.data_processing.feature_engineering import FeatureEngineer
from src.evaluation.plots import PlotGenerator
from src.models.model_trainer import ModelTrainer
from src.utils.config import ProjectConfig
from src.utils.logger import Logger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Entrena el modelo de dificultad (pipeline del tutor)."
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Limita filas para pruebas rapidas (ej: 50000).",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Desactiva la generacion de figuras (mas rapido).",
    )
    return parser


def main(max_rows: int | None = None, generate_plots: bool = True) -> None:
    Logger.print("Iniciando pipeline de entrenamiento...")
    config = ProjectConfig()
    loader = DataLoader(config)
    engineer = FeatureEngineer()
    trainer = ModelTrainer(config)
    plotter = PlotGenerator(config.figures_dir) if generate_plots else None

    raw_df = loader.load_dataset()
    Logger.print(f"Dataset cargado: {len(raw_df)} filas.")
    if max_rows is not None and max_rows > 0 and len(raw_df) > max_rows:
        raw_df = raw_df.head(max_rows).copy()
        Logger.print(f"Aplicado --max-rows={max_rows}.")
    dataset = engineer.transform(raw_df)
    Logger.print("Feature engineering completado.")
    x, y = engineer.split_features_target(dataset)
    split_series = dataset["source_split"] if "source_split" in dataset.columns else None
    time_series = dataset["event_order"] if "event_order" in dataset.columns else None
    groups = dataset["student_id"] if "student_id" in dataset.columns else None

    training_output = trainer.train(
        x,
        y,
        split_series=split_series,
        time_series=time_series,
        groups=groups,
    )
    leaderboard = training_output["leaderboard"]
    importances = training_output["feature_importance"]

    config.metrics_dir.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(config.processed_path, index=False)

    if plotter is not None:
        plotter.plot_error_distribution(dataset)
        plotter.plot_avg_time_by_step(dataset)
        plotter.plot_correlation_matrix(dataset)
        plotter.plot_feature_importance(engineer.feature_columns, importances)
        plotter.plot_training_evolution(leaderboard)
        plotter.plot_model_metric_comparison(leaderboard)
        plotter.plot_per_model_diagnostics(training_output["evaluation_payloads"])

    report = {
        "best_model": training_output["best_model_name"],
        "validation_type": training_output["validation_type"],
        "metrics_path": str(config.leaderboard_path),
        "figures_dir": str(config.figures_dir) if plotter is not None else None,
        "model_path": str(config.model_path),
        "feature_count": len(engineer.feature_columns),
    }
    summary_path = config.metrics_dir / "training_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    Logger.print("Entrenamiento completado.")
    Logger.print(f"Mejor modelo: {training_output['best_model_name']}")
    Logger.print(f"Modelo guardado en: {config.model_path}")
    Logger.print(f"Metricas guardadas en: {config.leaderboard_path}")
    if plotter is not None:
        Logger.print(f"Figuras guardadas en: {config.figures_dir}")


if __name__ == "__main__":
    args = build_parser().parse_args()
    main(max_rows=args.max_rows, generate_plots=not args.no_plots)
