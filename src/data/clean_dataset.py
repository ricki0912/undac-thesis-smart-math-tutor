from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

REQUIRED_FEATURES = [
    "Step Duration (sec)",
    "Incorrects",
    "Hints",
    "Corrects",
    "Skill",
    "Opportunity(SubSkills)",
]
TARGET_COLUMN = "Correct First Attempt"


def _find_input_files(input_dir: Path) -> list[Path]:
    txt_files = sorted(input_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(
            f"No se encontraron archivos .txt en {input_dir}. "
            "Coloca archivos train/test/master del KDD Cup 2010."
        )

    master_files = [path for path in txt_files if "master" in path.name.lower()]
    return master_files if master_files else txt_files


def _read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", low_memory=False)


def _resolve_skill_column(columns: Iterable[str]) -> str | None:
    for candidate in ("KC(SubSkills)", "KC(Default)", "KC(Rules)"):
        if candidate in columns:
            return candidate
    return None


def _prepare_skill_column(df: pd.DataFrame) -> pd.Series:
    skill_col = _resolve_skill_column(df.columns)
    if skill_col is None:
        return pd.Series(["sin_habilidad"] * len(df), index=df.index, dtype="object")

    base = df[skill_col].fillna("sin_habilidad").astype(str)
    # Baseline monolabel: si hay varias habilidades separadas por '~~', tomamos la primera.
    first_skill = base.str.split("~~").str[0].str.strip()
    first_skill = first_skill.replace("", "sin_habilidad")

    # Para expansion multilabel (opcional), usar por ejemplo:
    # expanded = base.str.split("~~").explode().str.strip()
    return first_skill


def _coerce_numeric(series: pd.Series, default_value: float = 0.0) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.fillna(default_value)


def clean_kdd_dataset(
    input_dir: str | Path = "data/external",
    output_path: str | Path = "data/processed/clean_dataset.csv",
) -> str:
    input_path = Path(input_dir)
    output_csv = Path(output_path)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    for file_path in _find_input_files(input_path):
        frame = _read_tsv(file_path)
        frame["Skill"] = _prepare_skill_column(frame)
        frames.append(frame)

    df = pd.concat(frames, axis=0, ignore_index=True)

    for column in ("Step Duration (sec)", "Incorrects", "Hints", "Corrects", "Opportunity(SubSkills)"):
        if column not in df.columns:
            df[column] = 0

    cleaned = pd.DataFrame()
    cleaned["Step Duration (sec)"] = _coerce_numeric(df["Step Duration (sec)"])
    cleaned["Incorrects"] = _coerce_numeric(df["Incorrects"])
    cleaned["Hints"] = _coerce_numeric(df["Hints"])
    cleaned["Corrects"] = _coerce_numeric(df["Corrects"])
    cleaned["Skill"] = df["Skill"].fillna("sin_habilidad").astype(str)
    cleaned["Opportunity(SubSkills)"] = _coerce_numeric(df["Opportunity(SubSkills)"])

    if TARGET_COLUMN in df.columns:
        target_numeric = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
        cleaned[TARGET_COLUMN] = target_numeric.fillna(0).astype(int).clip(lower=0, upper=1)

    cleaned.to_csv(output_csv, index=False)
    return str(output_csv)


if __name__ == "__main__":
    result = clean_kdd_dataset()
    print(f"Dataset limpio guardado en: {result}")
