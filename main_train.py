from __future__ import annotations

import argparse
import json
import time

from sklearn.metrics import classification_report, confusion_matrix

from src.data_processing.data_loader import DataLoader
from src.data_processing.feature_engineering import FeatureEngineer
from src.evaluation.plots import PlotGenerator
from src.models.model_trainer import ModelTrainer
from src.utils.config import ProjectConfig
from src.utils.logger import Logger

#pPARA PRESENTAR INFORMACIÓN 
import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", 0)

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
    start_time = time.time()

    Logger.print("========================================")
    Logger.print("INICIO -> Iniciando pipeline de entrenamiento...")
    config = ProjectConfig()
    loader = DataLoader(config)
    engineer = FeatureEngineer()
    trainer = ModelTrainer(config)
    plotter = PlotGenerator(config.figures_dir) if generate_plots else None

    # Carga inicial de datos
    raw_df = loader.load_dataset()
    Logger.print(f"Dataset crudo cargado: {len(raw_df)} filas, {len(raw_df.columns)} columnas.")
    Logger.print(f"Columnas del dataset crudo: {list(raw_df.columns)}")

    # Estadísticas iniciales detalladas
    Logger.print("=== ESTADÍSTICAS DATASET CRUDO ===")
    Logger.print(f"Valores nulos por columna:\n{raw_df.isnull().sum().to_string()}")
    Logger.print(f"Tipos de datos:\n{raw_df.dtypes.to_string()}")
    Logger.print(f"Estadísticas numéricas:\n{raw_df.describe().to_string()}")

    if max_rows is not None and max_rows > 0 and len(raw_df) > max_rows:
        raw_df = raw_df.head(max_rows).copy()
        Logger.print(f"Aplicado --max-rows={max_rows}. Dataset reducido a {len(raw_df)} filas.")

    # Feature engineering
    fe_start = time.time()
    dataset = engineer.transform(raw_df)
    fe_time = time.time() - fe_start

    Logger.print(f"Feature engineering completado en {fe_time:.2f} segundos.")
    Logger.print(f"Cambio en dataset: filas {len(raw_df)} -> {len(dataset)}, columnas {len(raw_df.columns)} -> {len(dataset.columns)}")

    # Estadísticas target detalladas
    Logger.print("=== ESTADÍSTICAS TARGET (effort_score) ===")
    target_stats = dataset['effort_score'].describe()
    Logger.print(f"Estadísticas target:\n{target_stats.to_string()}")
    Logger.print(f"Distribución target (value_counts):\n{dataset['effort_score'].value_counts().sort_index().to_string()}")

    # Estadísticas features
    Logger.print(f"Features creadas: {len(engineer.feature_columns)}")
    Logger.print(f"Lista de features: {engineer.feature_columns}")

    x, y = engineer.split_features_target(dataset)
    Logger.print(f"Features y target separados. Shape X: {x.shape}, Shape y: {y.shape}")

    # Correlación features-target
    correlations = x.corrwith(y).sort_values(ascending=False)
    Logger.print("=== CORRELACIONES FEATURES-TARGET (TOP 10) ===")
    Logger.print(f"Correlaciones positivas:\n{correlations.head(10).to_string()}")
    Logger.print(f"Correlaciones negativas:\n{correlations.tail(10).to_string()}")

    # Estadísticas detalladas de features
    Logger.print("=== ESTADÍSTICAS FEATURES ===")
    Logger.print(f"Estadísticas features:\n{x.describe().to_string()}")
    Logger.print(f"Valores nulos en features: {x.isnull().sum().sum()}")
    Logger.print(f"Duplicados en dataset: {dataset.duplicated().sum()}")

    # Split information
    split_series = dataset["source_split"] if "source_split" in dataset.columns else None
    time_series = dataset["event_order"] if "event_order" in dataset.columns else None
    groups = dataset["student_id"] if "student_id" in dataset.columns else None

    Logger.print("=== INFORMACIÓN DE SPLIT ===")
    Logger.print(f"Split series detectada: {'source_split' in dataset.columns}")
    if split_series is not None:
        Logger.print(f"Distribución split: {split_series.value_counts().to_string()}")

    Logger.print(f"Time series detectada: {'event_order' in dataset.columns}")
    Logger.print(f"Groups detectados: {'student_id' in dataset.columns}")
    if groups is not None:
        Logger.print(f"Número de grupos únicos: {groups.nunique()}")

    # Entrenamiento
    train_start = time.time()
    training_output = trainer.train(x, y, split_series=split_series, time_series=time_series, groups=groups)
    train_time = time.time() - train_start

    Logger.print(f"Entrenamiento completado en {train_time:.2f} segundos.")

    # Resultados del entrenamiento
    leaderboard = training_output["leaderboard"]
    importances = training_output["feature_importance"]

    Logger.print("=== LEADERBOARD DE MODELOS ===")
    Logger.print(f"Leaderboard completo:\n{leaderboard.to_string()}")

    # Métricas detalladas del mejor modelo
    best_model_name = training_output["best_model_name"]
    Logger.print(f"Mejor modelo: {best_model_name}")

    # Buscar el payload del mejor modelo
    best_payload = None
    for payload in training_output["evaluation_payloads"]:
        if payload["model_name"] == best_model_name:
            best_payload = payload
            break

    if best_payload is not None:
        from sklearn.metrics import classification_report, confusion_matrix
        y_true = best_payload["y_true"]
        y_pred = best_payload["y_pred"]

        Logger.print("=== MÉTRICAS DETALLADAS DEL MEJOR MODELO ===")
        Logger.print("Classification Report:")
        report = classification_report(y_true, y_pred, output_dict=False)
        Logger.print(report)

        cm = confusion_matrix(y_true, y_pred)
        Logger.print(f"Matriz de confusión:\n{cm}")

    # Importancia de features
    importances_df = pd.DataFrame({'feature': engineer.feature_columns, 'importance': importances}).sort_values('importance', ascending=False)
    Logger.print("=== IMPORTANCIA DE FEATURES (TOP 20) ===")
    Logger.print(f"Importancia completa:\n{importances_df.to_string()}")

    # Hiperparámetros del mejor modelo
    best_model = training_output["best_model"]
    Logger.print("=== HIPERPARÁMETROS DEL MEJOR MODELO ===")
    if hasattr(best_model, 'get_params'):
        params = best_model.get_params()
        Logger.print(f"Hiperparámetros: {params}")

    # Información adicional para tesis
    Logger.print("=== INFORMACIÓN PARA TESIS ===")
    Logger.print(f"Tiempo total pipeline: {time.time() - start_time:.2f} segundos")
    Logger.print(f"Ratio train/test: {len(x) / len(y) if 'source_split' in dataset.columns else 'N/A'}")
    Logger.print(f"Número de features finales: {len(engineer.feature_columns)}")
    Logger.print(f"Target discretizado: {training_output.get('label_thresholds') is not None}")
    if training_output.get('label_thresholds'):
        Logger.print(f"Umbrales de discretización: {training_output['label_thresholds']}")

    # Guardar dataset procesado
    config.metrics_dir.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(config.processed_path, index=False)
    Logger.print(f"Dataset procesado guardado en: {config.processed_path}")

    # Generar plots si corresponde
    if plotter is not None:
        Logger.print("Generando visualizaciones...")
        plotter.plot_error_distribution(dataset)
        plotter.plot_avg_time_by_step(dataset)
        plotter.plot_correlation_matrix(dataset)
        plotter.plot_feature_importance(engineer.feature_columns, importances)
        plotter.plot_training_evolution(leaderboard)
        plotter.plot_model_metric_comparison(leaderboard)
        plotter.plot_per_model_diagnostics(training_output["evaluation_payloads"])
        Logger.print(f"Figuras guardadas en: {config.figures_dir}")

    # Reporte final
    report = {
        "best_model": training_output["best_model_name"],
        "validation_type": training_output["validation_type"],
        "metrics_path": str(config.leaderboard_path),
        "figures_dir": str(config.figures_dir) if plotter is not None else None,
        "model_path": str(config.model_path),
        "feature_count": len(engineer.feature_columns),
        "training_time_seconds": train_time,
        "total_time_seconds": time.time() - start_time,
        "dataset_stats": {
            "original_rows": len(raw_df),
            "processed_rows": len(dataset),
            "features_count": len(engineer.feature_columns),
            "target_mean": float(y.mean()),
            "target_std": float(y.std())
        }
    }
    summary_path = config.metrics_dir / "training_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    Logger.print("Pipeline completado exitosamente.")
    Logger.print(f"Mejor modelo: {training_output['best_model_name']}")
    Logger.print(f"Modelo guardado en: {config.model_path}")
    Logger.print(f"Métricas guardadas en: {config.leaderboard_path}")
    if plotter is not None:
        Logger.print(f"Figuras guardadas en: {config.figures_dir}")
    Logger.print(f"Resumen guardado en: {summary_path}")


if __name__ == "__main__":
    args = build_parser().parse_args()
    main(max_rows=args.max_rows, generate_plots=not args.no_plots)
