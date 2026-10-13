"""Entrenamiento y evaluación del clasificador CONFIRMED vs FALSE POSITIVE.

Pasos:

1. Carga ``data/features_dataset_v2.csv`` por defecto (una fila por estrella);
   ``--dataset`` permite elegir otro CSV (p. ej. ``data/features_dataset_v1.csv``).
2. Usa SOLO las features calculadas desde la curva de luz. ``kepid`` no es
   feature (es un identificador) y las columnas ``koi_*`` del catálogo NASA
   tampoco (data leakage: son el resultado del pipeline que generó la etiqueta).
3. Compara Random Forest contra dos baselines (regresión logística y un
   clasificador "tonto") con validación cruzada estratificada de 5 folds.
4. Evalúa el Random Forest en un split estratificado 80/20 y guarda figuras en
   ``docs/``, las estrellas mal clasificadas en ``data/misclassified.csv`` y el
   pipeline completo en ``data/model.pkl``.

Uso:
    source venv/bin/activate.fish
    python src/train_model.py
    python src/train_model.py --dataset data/features_dataset_v1.csv

Para comparar v1 vs v2 con CV repetida y folds pareados, ver ``compare_versions.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = PROJECT_ROOT / "docs"
DATASET_PATH = DATA_DIR / "features_dataset_v2.csv"
MODEL_PATH = DATA_DIR / "model.pkl"
MISCLASSIFIED_PATH = DATA_DIR / "misclassified.csv"

# v1: snr/power crudos, 11 features.
FEATURE_COLUMNS_V1 = [
    "period",
    "depth",
    "duration",
    "snr",
    "power",
    "n_transits",
    "odd_even_diff",
    "secondary_depth",
    "depth_duration_ratio",
    "secondary_ratio",
    "flux_std",
]
# v2: log10(1+x) en snr/power y duration_period_ratio, 12 features.
FEATURE_COLUMNS = [
    "period",
    "depth",
    "duration",
    "log_snr",
    "log_power",
    "duration_period_ratio",
    "n_transits",
    "odd_even_diff",
    "secondary_depth",
    "depth_duration_ratio",
    "secondary_ratio",
    "flux_std",
]
LABEL_MAP = {"CONFIRMED": 1, "FALSE POSITIVE": 0}
CLASS_NAMES = ["FALSE POSITIVE", "CONFIRMED"]  # índice = valor de la etiqueta

SEED = 42
TEST_SIZE = 0.2
N_FOLDS = 5
MIN_ROWS_RECOMMENDED = 400

METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]


# --------------------------------------------------------------------------- #
# Datos
# --------------------------------------------------------------------------- #
def load_dataset(path: Path = DATASET_PATH) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Devuelve ``(X, y, kepid)`` con las features, la etiqueta 0/1 y el id.

    Detecta la versión del CSV: si trae ``log_snr`` usa las columnas v2; si no,
    las v1 (para poder reentrenar sobre el dataset anterior).
    """
    df = pd.read_csv(path)
    df = df[df["label"].isin(LABEL_MAP)].reset_index(drop=True)
    X = df[FEATURE_COLUMNS if "log_snr" in df.columns else FEATURE_COLUMNS_V1]
    y = df["label"].map(LABEL_MAP).astype(int)
    return X, y, df["kepid"]


# --------------------------------------------------------------------------- #
# Modelos
# --------------------------------------------------------------------------- #
def build_models(seed: int = SEED) -> dict[str, Pipeline]:
    """Los tres modelos. Todos llevan SimpleImputer(mediana) dentro del Pipeline,
    de modo que la mediana se calcula solo con los datos de entrenamiento de
    cada fold (sin filtrar información del test)."""
    impute = lambda: SimpleImputer(strategy="median")  # noqa: E731
    return {
        "Random Forest": Pipeline([
            ("imputer", impute()),
            ("clf", RandomForestClassifier(
                n_estimators=300, class_weight="balanced", random_state=seed)),
        ]),
        "Regresión logística": Pipeline([
            ("imputer", impute()),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
        ]),
        "Dummy (clase mayoritaria)": Pipeline([
            ("imputer", impute()),
            ("clf", DummyClassifier(strategy="most_frequent")),
        ]),
    }


# --------------------------------------------------------------------------- #
# Evaluación
# --------------------------------------------------------------------------- #
def cross_validate_models(models: dict[str, Pipeline], X, y, seed: int = SEED) -> pd.DataFrame:
    """CV estratificada de 5 folds; devuelve media ± std de cada métrica."""
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
        "roc_auc": "roc_auc",
    }
    rows = {}
    for name, model in models.items():
        scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=1)
        rows[name] = {
            m: (scores[f"test_{m}"].mean(), scores[f"test_{m}"].std()) for m in METRICS
        }
    return pd.DataFrame(rows).T


def test_metrics(model: Pipeline, X_test, y_test) -> dict[str, float]:
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba),
    }


def format_cv_table(cv_table: pd.DataFrame, test_table: pd.DataFrame) -> str:
    lines = [f"{'Modelo':<28}" + "".join(f"{m:>16}" for m in METRICS)]
    lines.append("-" * len(lines[0]))
    lines.append("Validación cruzada (media ± std, 5 folds, todo el dataset)")
    for name, row in cv_table.iterrows():
        lines.append(f"{name:<28}" + "".join(
            f"{f'{mean:.3f} ± {std:.3f}':>16}" for mean, std in row))
    lines.append("")
    lines.append("Test set 20% (una sola partición)")
    for name, row in test_table.iterrows():
        lines.append(f"{name:<28}" + "".join(f"{row[m]:>16.3f}" for m in METRICS))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Figuras
# --------------------------------------------------------------------------- #
def plot_confusion_matrix(model, X_test, y_test, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, display_labels=CLASS_NAMES, cmap="Blues", ax=ax)
    ax.set_title("Random Forest: matriz de confusión (test set)")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curve(model, X_test, y_test, path: Path) -> float:
    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    RocCurveDisplay.from_predictions(
        y_test, proba, ax=ax, name=f"Random Forest (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Azar (AUC = 0.5)")
    ax.set_title("Random Forest: curva ROC (test set)")
    ax.legend(loc="lower right")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return auc


def plot_feature_importance(importances: pd.Series, path: Path) -> None:
    ordered = importances.sort_values()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(ordered.index, ordered.values, color="steelblue")
    ax.set_xlabel("Importancia (reducción media de impureza de Gini)")
    ax.set_title("Random Forest: importancia de features")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(dataset_path: Path = DATASET_PATH) -> None:
    plt.switch_backend("Agg")  # solo se guardan figuras, no se muestran
    X, y, kepid = load_dataset(dataset_path)
    columns = list(X.columns)
    print(f"Dataset: {len(X)} estrellas, {y.sum()} CONFIRMED / {(1 - y).sum()} FALSE POSITIVE")
    if len(X) < MIN_ROWS_RECOMMENDED:
        print(f"AVISO: el dataset tiene menos de {MIN_ROWS_RECOMMENDED} filas; "
              "los resultados tendrán bastante varianza.")
    print(f"NaN por feature (los imputa la mediana del train):\n{X.isna().sum()[X.isna().any()]}\n")

    models = build_models()

    # --- Validación cruzada sobre todo el dataset ---
    cv_table = cross_validate_models(models, X, y)

    # --- Split estratificado 80/20 ---
    X_train, X_test, y_train, y_test, _, id_test = train_test_split(
        X, y, kepid, test_size=TEST_SIZE, stratify=y, random_state=SEED)
    test_rows = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        test_rows[name] = test_metrics(model, X_test, y_test)
    test_table = pd.DataFrame(test_rows).T

    print(format_cv_table(cv_table, test_table))

    rf = models["Random Forest"]
    lr_auc = cv_table.loc["Regresión logística", "roc_auc"][0]
    rf_auc = cv_table.loc["Random Forest", "roc_auc"][0]
    lr_std = cv_table.loc["Regresión logística", "roc_auc"][1]
    rf_std = cv_table.loc["Random Forest", "roc_auc"][1]
    gap = rf_auc - lr_auc
    if gap > max(rf_std, lr_std) * 2 and gap > 0.02:
        print(f"\nEl Random Forest le gana a la regresión logística (ΔAUC CV = {gap:+.3f}).")
    else:
        print(f"\nEl Random Forest NO le gana claramente a la regresión logística "
              f"(ΔAUC CV = {gap:+.3f}, std RF {rf_std:.3f} / LR {lr_std:.3f}).")

    # --- Resultados del RF en test ---
    DOCS_DIR.mkdir(exist_ok=True)
    plot_confusion_matrix(rf, X_test, y_test, DOCS_DIR / "confusion_matrix.png")
    auc = plot_roc_curve(rf, X_test, y_test, DOCS_DIR / "roc_curve.png")

    importances = pd.Series(rf.named_steps["clf"].feature_importances_, index=columns)
    plot_feature_importance(importances, DOCS_DIR / "feature_importance.png")
    print("\nImportancia de features (Gini, RF entrenado con el 80%):")
    print(importances.sort_values(ascending=False).round(3).to_string())

    # La importancia de Gini favorece features continuas/de alta cardinalidad;
    # la de permutación (sobre test) es un contraste más honesto.
    perm = permutation_importance(
        rf, X_test, y_test, scoring="roc_auc", n_repeats=20, random_state=SEED)
    perm_s = pd.Series(perm.importances_mean, index=columns).sort_values(ascending=False)
    print("\nImportancia por permutación (caída de AUC en test):")
    print(perm_s.round(3).to_string())

    proba = rf.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    wrong = pred != y_test.values
    mis = pd.DataFrame({
        "kepid": id_test.values[wrong],
        "true_label": [CLASS_NAMES[v] for v in y_test.values[wrong]],
        "predicted_proba_confirmed": proba[wrong].round(4),
    }).sort_values("predicted_proba_confirmed", ascending=False)
    mis.to_csv(MISCLASSIFIED_PATH, index=False)
    print(f"\nMal clasificadas en test: {wrong.sum()} de {len(y_test)} (AUC test = {auc:.3f})")
    print(mis.to_string(index=False))

    # --- Pipeline final guardado ---
    # Se guarda el RF entrenado con el 80% de train, que es el que se evaluó:
    # así las métricas, el CSV de errores y el notebook describen el mismo modelo.
    joblib.dump(rf, MODEL_PATH)
    print(f"\nModelo guardado en {MODEL_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrena y evalúa el clasificador.")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH,
                        help="CSV de features (v1 o v2).")
    sys.exit(main(parser.parse_args().dataset))
