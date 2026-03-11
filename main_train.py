from __future__ import annotations

import json

from src.data_processing.data_loader import DataLoader
from src.data_processing.feature_engineering import FeatureEngineer
from src.evaluation.plots import PlotGenerator
from src.models.model_trainer import ModelTrainer
from src.utils.config import ProjectConfig


def main() -> None:
    config = ProjectConfig()
    loader = DataLoader(config)
    engineer = FeatureEngineer()
    trainer = ModelTrainer(config)
    plotter = PlotGenerator(config.figures_dir)

    raw_df = loader.load_dataset()
    dataset = engineer.transform(raw_df)
    x, y = engineer.split_features_target(dataset)
    split_series = dataset["source_split"] if "source_split" in dataset.columns else None
    time_series = dataset["event_order"] if "event_order" in dataset.columns else None

    training_output = trainer.train(x, y, split_series=split_series, time_series=time_series)
    leaderboard = training_output["leaderboard"]
    importances = training_output["feature_importance"]

    config.metrics_dir.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(config.processed_path, index=False)

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
        "figures_dir": str(config.figures_dir),
        "model_path": str(config.model_path),
        "feature_count": len(engineer.feature_columns),
    }
    summary_path = config.metrics_dir / "training_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print("Entrenamiento completado.")
    print(f"Mejor modelo: {training_output['best_model_name']}")
    print(f"Modelo guardado en: {config.model_path}")
    print(f"Metricas guardadas en: {config.leaderboard_path}")
    print(f"Figuras guardadas en: {config.figures_dir}")


if __name__ == "__main__":
    main()
