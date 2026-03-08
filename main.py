from __future__ import annotations

import argparse
import sys

import uvicorn

from src.data.clean_dataset import clean_kdd_dataset
from src.models.train_models import train_and_select_models


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pipeline del tutor inteligente de matematicas."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    clean_parser = subparsers.add_parser("clean", help="Limpiar dataset KDD.")
    clean_parser.add_argument(
        "--input-dir",
        default="data/external",
        help="Carpeta de entrada con archivos train/test/master.",
    )
    clean_parser.add_argument(
        "--output",
        default="data/processed/clean_dataset.csv",
        help="Ruta de salida CSV limpio.",
    )

    train_parser = subparsers.add_parser(
        "train", help="Entrenar modelos y guardar mejor modelo."
    )
    train_parser.add_argument(
        "--data-path",
        default="data/processed/clean_dataset.csv",
        help="Ruta del dataset limpio.",
    )
    train_parser.add_argument(
        "--metric",
        default="accuracy",
        choices=["accuracy", "precision", "recall", "f1", "auc"],
        help="Metrica usada para seleccionar el mejor modelo.",
    )
    train_parser.add_argument(
        "--models-dir",
        default="models",
        help="Directorio para guardar modelos y metadatos.",
    )

    serve_parser = subparsers.add_parser("serve", help="Levantar API FastAPI.")
    serve_parser.add_argument("--host", default="127.0.0.1", help="Host del servidor.")
    serve_parser.add_argument("--port", type=int, default=8000, help="Puerto del servidor.")
    serve_parser.add_argument("--reload", action="store_true", help="Activar autoreload.")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "clean":
        output_path = clean_kdd_dataset(input_dir=args.input_dir, output_path=args.output)
        print(f"Dataset limpio guardado en: {output_path}")
        return 0

    if args.command == "train":
        summary = train_and_select_models(
            data_path=args.data_path,
            models_dir=args.models_dir,
            selection_metric=args.metric,
        )
        print("Entrenamiento completado.")
        print(f"Mejor modelo: {summary['best_model']}")
        print(f"Metrica ({args.metric}): {summary['best_metric_value']:.4f}")
        print(f"Resultados: {summary['results_path']}")
        return 0

    if args.command == "serve":
        print(f"Levantando API en http://{args.host}:{args.port} ...")
        uvicorn.run("src.api.main:app", host=args.host, port=args.port, reload=args.reload)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
