"""Comparación v1 vs v2 del dataset de features, con CV repetida y folds PAREADOS.

- Mismas 399 estrellas y misma semilla en ambas versiones (se verifica).
- ``RepeatedStratifiedKFold`` (5 folds x 5 repeticiones = 25 folds) con la misma
  semilla: como las etiquetas están en el mismo orden, los folds son idénticos
  para v1 y v2, lo que permite comparar el AUC fold por fold.
- Importancia por permutación calculada sobre los folds de validación de la CV
  (no sobre un único test de 80 estrellas).
- Además se reporta el test set 80/20 de ``train_model.py`` (una sola partición).

Salidas: ``docs/comparison_v1_v2.csv`` y ``docs/permutation_importance_cv.csv``.

Uso:
    source venv/bin/activate.fish
    python -u src/compare_versions.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_model import (  # noqa: E402
    DATA_DIR,
    DOCS_DIR,
    METRICS,
    SEED,
    TEST_SIZE,
    build_models,
    load_dataset,
    test_metrics,
)

V1_PATH = DATA_DIR / "features_dataset_v1.csv"
V2_PATH = DATA_DIR / "features_dataset_v2.csv"
N_FOLDS, N_REPEATS = 5, 5
MODELS = ["Random Forest", "Regresión logística"]
SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc",
}


def cv_permutation_importance(model, X, y, cv, n_repeats: int = 5) -> pd.DataFrame:
    """Importancia por permutación (caída de AUC) en cada fold de validación.

    Devuelve un DataFrame (folds x features); cada celda es la caída media de
    AUC al permutar esa feature en el fold de validación, con el modelo
    entrenado solo con el fold de entrenamiento.
    """
    def one_fold(train_idx, val_idx):
        warnings.filterwarnings("ignore")
        m = clone(model).fit(X.iloc[train_idx], y.iloc[train_idx])
        p = permutation_importance(m, X.iloc[val_idx], y.iloc[val_idx],
                                   scoring="roc_auc", n_repeats=n_repeats,
                                   random_state=SEED)
        return p.importances_mean

    rows = Parallel(n_jobs=-1)(delayed(one_fold)(tr, va) for tr, va in cv.split(X, y))
    return pd.DataFrame(rows, columns=X.columns)


def main() -> None:
    warnings.filterwarnings("ignore")
    X1, y1, id1 = load_dataset(V1_PATH)
    X2, y2, id2 = load_dataset(V2_PATH)
    assert (id1.values == id2.values).all() and (y1.values == y2.values).all(), \
        "v1 y v2 deben tener las mismas estrellas en el mismo orden"
    y, kepid = y1, id1
    print(f"{len(y)} estrellas ({y.sum()} CONFIRMED / {(1 - y).sum()} FALSE POSITIVE), "
          f"v1: {X1.shape[1]} features, v2: {X2.shape[1]} features", flush=True)

    cv = RepeatedStratifiedKFold(n_splits=N_FOLDS, n_repeats=N_REPEATS, random_state=SEED)
    sets = {"v1": X1, "v2": X2}

    # --- CV repetida (mismos folds) ---
    cv_scores: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for ver, X in sets.items():
        models = build_models()
        for name in MODELS:
            res = cross_validate(models[name], X, y, cv=cv, scoring=SCORING, n_jobs=-1)
            cv_scores[(ver, name)] = {m: res[f"test_{m}"] for m in METRICS}
            print(f"  CV {ver} {name}: AUC={res['test_roc_auc'].mean():.3f}", flush=True)

    # --- Test set 80/20 (misma partición para ambas versiones) ---
    tr_idx, te_idx = train_test_split(np.arange(len(y)), test_size=TEST_SIZE,
                                      stratify=y, random_state=SEED)
    test = {}
    for ver, X in sets.items():
        models = build_models()
        for name in MODELS:
            models[name].fit(X.iloc[tr_idx], y.iloc[tr_idx])
            test[(ver, name)] = test_metrics(models[name], X.iloc[te_idx], y.iloc[te_idx])

    # --- Tabla ---
    rows = []
    for name in MODELS:
        for ver in sets:
            r = {"model": name, "version": ver}
            for m in METRICS:
                s = cv_scores[(ver, name)][m]
                r[f"cv_{m}"] = f"{s.mean():.3f} ± {s.std():.3f}"
            for m in METRICS:
                r[f"test_{m}"] = round(test[(ver, name)][m], 3)
            rows.append(r)
    table = pd.DataFrame(rows)
    DOCS_DIR.mkdir(exist_ok=True)
    table.to_csv(DOCS_DIR / "comparison_v1_v2.csv", index=False)
    print("\n=== v1 vs v2: CV 5x5 (media ± std sobre 25 folds) y test 80/20 ===")
    print(table.to_string(index=False), flush=True)

    # --- Diferencias pareadas de AUC ---
    print("\n=== Diferencia pareada de AUC (v2 - v1) por fold ===")
    for name in MODELS:
        d = cv_scores[("v2", name)]["roc_auc"] - cv_scores[("v1", name)]["roc_auc"]
        se = d.std(ddof=1) / np.sqrt(len(d))
        print(f"{name}: media={d.mean():+.3f}, std={d.std(ddof=1):.3f}, "
              f"error estándar≈{se:.3f}, v2 mejora en {(d > 0).sum()}/{len(d)} folds "
              f"(empata en {(d == 0).sum()})")
    print("(Los 25 folds no son independientes: las 5 repeticiones reutilizan las mismas "
          "estrellas, así que el error estándar es optimista.)", flush=True)

    # --- Importancia por permutación sobre la CV ---
    print("\n=== Importancia por permutación en CV (caída de AUC, Random Forest) ===")
    perm_out = {}
    for ver, X in sets.items():
        imp = cv_permutation_importance(build_models()["Random Forest"], X, y, cv)
        summary = pd.DataFrame({"mean": imp.mean(), "std": imp.std()}).sort_values(
            "mean", ascending=False)
        summary["version"] = ver
        perm_out[ver] = summary
        print(f"\n{ver}:")
        print(summary[["mean", "std"]].round(4).to_string(), flush=True)
    pd.concat(perm_out.values()).rename_axis("feature").to_csv(
        DOCS_DIR / "permutation_importance_cv.csv")


if __name__ == "__main__":
    main()
