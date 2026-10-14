"""Entrenamiento y evaluación del clasificador CONFIRMED vs FALSE POSITIVE (v3).

Unidad de análisis: un KOI por fila (``data/features_dataset_v3.csv``), con las 9
features de ``koi_features.FEATURE_COLUMNS_V3``. Ninguna columna ``koi_*`` del
catálogo entra como feature (data leakage).

Pasos de ``main()``:

1. Resultado principal: Random Forest (imputer dentro del Pipeline) con
   ``StratifiedGroupKFold`` por ``kepid``, 5 folds x 5 repeticiones (los mismos
   25 folds de ``compare_v2_v3.py``). Se reporta media y desvío de 5 métricas.
2. Split held-out agrupado por ``kepid`` (~80/20, semilla fija): ninguna
   estrella aparece en train y test. Con ese test se calculan las métricas, las
   figuras de ``docs/`` (matriz de confusión, ROC, importancia por permutación)
   y ``data/misclassified.csv``.
3. Se guarda el Pipeline entrenado con el 80% en ``data/model.pkl`` y la lista
   de features en ``data/model_features.json``.

Uso:
    source venv/bin/activate.fish
    python src/train_model.py

``build_models``, ``load_dataset`` y las constantes v1/v2 se conservan porque las
usan ``compare_versions.py`` y ``compare_v2_v3.py``.
"""

from __future__ import annotations

import json
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
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = PROJECT_ROOT / "docs"
DATASET_PATH = DATA_DIR / "features_dataset_v2.csv"  # usado por compare_versions.py
DATASET_V3_PATH = DATA_DIR / "features_dataset_v3.csv"
MODEL_PATH = DATA_DIR / "model.pkl"
MISCLASSIFIED_PATH = DATA_DIR / "misclassified.csv"
FEATURES_PATH = DATA_DIR / "model_features.json"

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


def load_dataset_v3(path: Path = DATASET_V3_PATH) -> pd.DataFrame:
    """Tabla v3 (una fila por KOI) con ``y`` (0/1) agregada; sin KOIs CANDIDATE."""
    df = pd.read_csv(path)
    df = df[df["label"].isin(LABEL_MAP)].reset_index(drop=True)
    df["y"] = df["label"].map(LABEL_MAP).astype(int)
    return df


def heldout_split(df: pd.DataFrame, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """Índices (train, test) ~80/20, estratificado y agrupado por ``kepid``.

    Es el primer fold de un ``StratifiedGroupKFold(5)``: ninguna estrella tiene
    KOIs en train y test a la vez. Lo usan el entrenamiento y el notebook.
    """
    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    return next(cv.split(np.zeros(len(df)), df["y"], df["kepid"]))


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


def plot_feature_importance(importances: pd.Series, path: Path,
                            xlabel: str = "Importancia (reducción media de impureza de Gini)") -> None:
    ordered = importances.sort_values()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(ordered.index, ordered.values, color="steelblue")
    ax.set_xlabel(xlabel)
    ax.set_title("Random Forest: importancia de features")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def grouped_cv_table(model: Pipeline, X, y, groups) -> pd.DataFrame:
    """CV 5x5 agrupada por kepid; devuelve media y desvío de cada métrica."""
    from compare_v2_v3 import grouped_splits  # import tardío: ese módulo importa este

    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
        "roc_auc": "roc_auc",
    }
    res = cross_validate(model, X, y, cv=grouped_splits(y, groups), scoring=scoring, n_jobs=-1)
    return pd.DataFrame({m: {"mean": res[f"test_{m}"].mean(), "std": res[f"test_{m}"].std()}
                         for m in METRICS}).T


def main() -> None:
    from compare_v2_v3 import build_models as build_v3_models  # mismo RF que la comparación
    from koi_features import FEATURE_COLUMNS_V3, _assert_no_leakage

    plt.switch_backend("Agg")  # solo se guardan figuras, no se muestran
    _assert_no_leakage(FEATURE_COLUMNS_V3)
    df = load_dataset_v3()
    X, y, groups = df[FEATURE_COLUMNS_V3], df["y"], df["kepid"]
    print(f"Dataset v3: {len(df)} KOIs en {groups.nunique()} estrellas, "
          f"{y.sum()} CONFIRMED / {(1 - y).sum()} FALSE POSITIVE")
    rf = build_v3_models()["Random Forest"]

    # --- Resultado principal: CV agrupada por estrella, 5x5 ---
    cv = grouped_cv_table(rf, X, y, groups)
    print("\nCV agrupada por kepid, 5 folds x 5 repeticiones (media ± std):")
    print(cv.apply(lambda r: f"{r['mean']:.3f} ± {r['std']:.3f}", axis=1).to_string())

    # --- Split held-out agrupado: figuras, errores y modelo guardado ---
    tr, te = heldout_split(df)
    assert not set(groups.iloc[tr]) & set(groups.iloc[te])
    rf.fit(X.iloc[tr], y.iloc[tr])
    X_test, y_test = X.iloc[te], y.iloc[te]
    test = test_metrics(rf, X_test, y_test)
    print(f"\nTest held-out ({len(te)} KOIs, {groups.iloc[te].nunique()} estrellas):")
    print(pd.Series(test).round(3).to_string())

    DOCS_DIR.mkdir(exist_ok=True)
    plot_confusion_matrix(rf, X_test, y_test, DOCS_DIR / "confusion_matrix.png")
    plot_roc_curve(rf, X_test, y_test, DOCS_DIR / "roc_curve.png")
    perm = permutation_importance(
        rf, X_test, y_test, scoring="roc_auc", n_repeats=20, random_state=SEED)
    perm_s = pd.Series(perm.importances_mean, index=FEATURE_COLUMNS_V3)
    plot_feature_importance(perm_s, DOCS_DIR / "feature_importance.png",
                            xlabel="Importancia por permutación (caída de AUC en test)")
    print("\nImportancia por permutación (caída de AUC en test):")
    print(perm_s.sort_values(ascending=False).round(3).to_string())

    proba = rf.predict_proba(X_test)[:, 1]
    wrong = (proba >= 0.5).astype(int) != y_test.values
    mis = pd.DataFrame({
        "kepoi_name": df["kepoi_name"].iloc[te].values[wrong],
        "kepid": groups.iloc[te].values[wrong],
        "true_label": [CLASS_NAMES[v] for v in y_test.values[wrong]],
        "predicted_proba_confirmed": proba[wrong].round(4),
    }).sort_values("predicted_proba_confirmed", ascending=False)
    mis.to_csv(MISCLASSIFIED_PATH, index=False)
    print(f"\nMal clasificados en test: {wrong.sum()} de {len(te)}")

    # Se guarda el RF entrenado con el 80%: el que se evaluó y el que muestra el notebook.
    joblib.dump(rf, MODEL_PATH)
    FEATURES_PATH.write_text(json.dumps(FEATURE_COLUMNS_V3, indent=2))
    print(f"Modelo: {MODEL_PATH}\nFeatures: {FEATURES_PATH}")


if __name__ == "__main__":
    main()
