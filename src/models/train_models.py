from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

FEATURE_COLUMNS = [
    "Step Duration (sec)",
    "Incorrects",
    "Hints",
    "Corrects",
    "Skill",
    "Opportunity(SubSkills)",
]
TARGET_COLUMN = "Correct First Attempt"


def _load_data(data_path: str | Path) -> pd.DataFrame:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontro {path}. Ejecuta primero: python main.py clean"
        )
    df = pd.read_csv(path)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            "El dataset limpio no tiene la etiqueta 'Correct First Attempt'. "
            "Usa archivos master para entrenamiento supervisado."
        )
    return df


def _build_models() -> dict[str, Pipeline]:
    try:
        skill_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        skill_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocess = ColumnTransformer(
        transformers=[
            ("cat_skill", skill_encoder, ["Skill"]),
            ("num", "passthrough", ["Step Duration (sec)", "Incorrects", "Hints", "Corrects", "Opportunity(SubSkills)"]),
        ],
        remainder="drop",
    )

    return {
        "logistic_regression": Pipeline(
            steps=[
                ("preprocess", preprocess),
                ("model", LogisticRegression(max_iter=1000, solver="liblinear", random_state=42)),
            ]
        ),
        "decision_tree": Pipeline(
            steps=[
                ("preprocess", preprocess),
                ("model", DecisionTreeClassifier(random_state=42)),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocess", preprocess),
                ("model", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
            ]
        ),
        "gradient_boosting": Pipeline(
            steps=[
                ("preprocess", preprocess),
                ("model", GradientBoostingClassifier(random_state=42)),
            ]
        ),
        "svm": Pipeline(
            steps=[
                ("preprocess", preprocess),
                ("model", SVC(probability=True, random_state=42)),
            ]
        ),
        "gaussian_nb": Pipeline(
            steps=[
                ("preprocess", preprocess),
                ("model", GaussianNB()),
            ]
        ),
    }


def _predict_scores(model: Pipeline, x_test: pd.DataFrame) -> np.ndarray | None:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_test)[:, 1]
    if hasattr(model, "decision_function"):
        raw = model.decision_function(x_test)
        raw = np.asarray(raw, dtype=float)
        if raw.ndim > 1:
            raw = raw[:, 0]
        min_val = np.min(raw)
        max_val = np.max(raw)
        if max_val == min_val:
            return np.zeros_like(raw)
        return (raw - min_val) / (max_val - min_val)
    return None


def _safe_auc(y_true: np.ndarray, y_scores: np.ndarray | None) -> float:
    if y_scores is None:
        return float("nan")
    unique_values = np.unique(y_true)
    if len(unique_values) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_scores))


def train_and_select_models(
    data_path: str | Path = "data/processed/clean_dataset.csv",
    models_dir: str | Path = "models",
    selection_metric: str = "accuracy",
) -> dict[str, object]:
    df = _load_data(data_path)
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0 if col != "Skill" else "sin_habilidad"

    x = df[FEATURE_COLUMNS].copy()
    x["Skill"] = x["Skill"].fillna("sin_habilidad").astype(str)
    for num_col in [c for c in FEATURE_COLUMNS if c != "Skill"]:
        x[num_col] = pd.to_numeric(x[num_col], errors="coerce").fillna(0.0)

    y_raw = pd.to_numeric(df[TARGET_COLUMN], errors="coerce").fillna(0).astype(int).clip(lower=0, upper=1)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    if len(np.unique(y)) < 2:
        raise ValueError(
            "La etiqueta tiene una sola clase. Verifica que el dataset master contenga ejemplos de 0 y 1."
        )

    try:
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=42, stratify=None
        )

    models = _build_models()
    model_results: list[dict[str, float | str]] = []

    output_dir = Path(models_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        y_scores = _predict_scores(model, x_test)

        metrics = {
            "model": model_name,
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "auc": _safe_auc(y_test, y_scores),
        }
        model_results.append(metrics)

        model_path = output_dir / f"model_{model_name}.pkl"
        joblib.dump(model, model_path)

    results_df = pd.DataFrame(model_results).sort_values(by=selection_metric, ascending=False)
    best_row = results_df.iloc[0]
    best_name = str(best_row["model"])

    meta_path = output_dir / "meta.pkl"
    joblib.dump({"best": best_name, "le": label_encoder}, meta_path)

    results_path = output_dir / "results.txt"
    with results_path.open("w", encoding="utf-8") as f:
        f.write("Resultados de entrenamiento (KDD Cup 2010)\n")
        f.write(f"Metrica de seleccion: {selection_metric}\n")
        f.write("=" * 80 + "\n")
        for _, row in results_df.iterrows():
            f.write(
                f"Modelo: {row['model']}\n"
                f"  accuracy={row['accuracy']:.4f}\n"
                f"  precision={row['precision']:.4f}\n"
                f"  recall={row['recall']:.4f}\n"
                f"  f1={row['f1']:.4f}\n"
                f"  auc={row['auc']:.4f}\n"
            )
        f.write("=" * 80 + "\n")
        f.write(f"Mejor modelo: {best_name}\n")

    return {
        "best_model": best_name,
        "best_metric_value": float(best_row[selection_metric]),
        "results_path": str(results_path),
    }


if __name__ == "__main__":
    info = train_and_select_models()
    print(
        f"Entrenamiento finalizado. Mejor modelo: {info['best_model']} "
        f"({info['best_metric_value']:.4f})"
    )
