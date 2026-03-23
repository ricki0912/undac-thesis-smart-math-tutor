from __future__ import annotations

import argparse
import sys

import uvicorn

from main_train import main as run_tutor_training
from src.utils.logger import Logger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pipeline del tutor inteligente de matematicas."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser(
        "train",
        help="Entrenar el modelo del tutor (difficulty_level) y generar reportes.",
    )
    train_parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Limita filas para pruebas rapidas (ej: 50000).",
    )
    train_parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Desactiva la generacion de figuras (mas rapido).",
    )

    serve_parser = subparsers.add_parser("serve", help="Levantar API FastAPI.")
    serve_parser.add_argument("--host", default="127.0.0.1", help="Host del servidor.")
    serve_parser.add_argument("--port", type=int, default=8000, help="Puerto del servidor.")
    serve_parser.add_argument("--reload", action="store_true", help="Activar autoreload.")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        run_tutor_training(max_rows=args.max_rows, generate_plots=not args.no_plots)
        return 0

    if args.command == "serve":
        Logger.print(f"Levantando API en http://{args.host}:{args.port} ...")
        uvicorn.run("src.api.main:app", host=args.host, port=args.port, reload=args.reload)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
