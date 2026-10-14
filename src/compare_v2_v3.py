"""Comparación v2 vs v3 a nivel KOI, con CV agrupada por estrella y folds PAREADOS.

Unidad de análisis: una fila por KOI (dataset v3). Para que la comparación sea
pareada, v2 se evalúa sobre EXACTAMENTE las mismas filas: a cada KOI se le
asignan las features v2 de su estrella (BLS, una fila por kepid). Así todos los
conjuntos de features comparten filas, etiquetas y folds; solo cambian las
columnas.

- Folds: ``StratifiedGroupKFold`` (5 folds) agrupando por ``kepid``, repetido
  5 veces con semillas distintas = 25 folds. Ninguna estrella tiene KOIs en
  train y en test a la vez.
- Conjuntos: v2 (12 features BLS), v3 (9 features nuevas), v2+v3 (21).
- Modelos: Random Forest y regresión logística con ``QuantileTransformer``
  (todas las features llevadas a una gaussiana, ajustado dentro de cada fold),
  que neutraliza las colas pesadas. Se reporta también la logística con solo
  StandardScaler como referencia.

Salidas: ``docs/comparison_v2_v3.csv``, ``docs/comparison_v2_v3_folds.csv`` y
``docs/permutation_importance_cv_v3.csv``.

Uso:
    source venv/bin/activate.fish
    python -u src/compare_v2_v3.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

from koi_features import DATASET_V3_PATH, FEATURE_COLUMNS_V3, _assert_no_leakage  # noqa: E402
from train_model import DATA_DIR, DOCS_DIR, FEATURE_COLUMNS, LABEL_MAP, SEED  # noqa: E402

V2_PATH = DATA_DIR / "features_dataset_v2.csv"
N_FOLDS, N_REPEATS = 5, 5
FEATURE_SETS = {
    "v2": FEATURE_COLUMNS,
    "v3": FEATURE_COLUMNS_V3,
    "v2+v3": FEATURE_COLUMNS + FEATURE_COLUMNS_V3,
}
SCORING = ["roc_auc", "accuracy", "f1"]


def build_models() -> dict[str, Pipeline]:
    impute = lambda: SimpleImputer(strategy="median")  # noqa: E731
    return {
        "Random Forest": Pipeline([
            ("imputer", impute()),
            ("clf", RandomForestClassifier(
                n_estimators=300, class_weight="balanced", random_state=SEED)),
        ]),
        "LogReg + quantile": Pipeline([
            ("imputer", impute()),
            ("qt", QuantileTransformer(n_quantiles=100, output_distribution="normal",
                                       random_state=SEED)),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]),
        "LogReg (solo scaler)": Pipeline([
            ("imputer", impute()),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]),
    }


def load_koi_table() -> pd.DataFrame:
    """Filas v3 (una por KOI) con las features v2 de su estrella."""
    v3 = pd.read_csv(DATASET_V3_PATH)
    v2 = pd.read_csv(V2_PATH).drop(columns="label")
    df = v3.merge(v2, on="kepid", how="inner", validate="many_to_one")
    assert len(df) == len(v3), "faltan features v2 para algún kepid"
    for cols in FEATURE_SETS.values():
        _assert_no_leakage(cols)
    return df.sort_values("kepoi_name").reset_index(drop=True)


def grouped_splits(y: pd.Series, groups: pd.Series) -> list[tuple[np.ndarray, np.ndarray]]:
    splits = []
    for r in range(N_REPEATS):
        cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED + r)
        for tr, te in cv.split(np.zeros(len(y)), y, groups):
            assert not set(groups.iloc[tr]) & set(groups.iloc[te])
            splits.append((tr, te))
    return splits


def cv_permutation_importance(model, X, y, splits, n_repeats: int = 5) -> pd.DataFrame:
    def one_fold(tr, te):
        warnings.filterwarnings("ignore")
        m = clone(model).fit(X.iloc[tr], y.iloc[tr])
        return permutation_importance(m, X.iloc[te], y.iloc[te], scoring="roc_auc",
                                      n_repeats=n_repeats, random_state=SEED).importances_mean

    rows = Parallel(n_jobs=-1)(delayed(one_fold)(tr, te) for tr, te in splits)
    return pd.DataFrame(rows, columns=X.columns)


def main() -> None:
    warnings.filterwarnings("ignore")
    df = load_koi_table()
    y = df["label"].map(LABEL_MAP).astype(int)
    groups = df["kepid"]
    print(f"{len(df)} KOIs ({y.sum()} CONFIRMED / {(1 - y).sum()} FALSE POSITIVE) "
          f"en {groups.nunique()} estrellas", flush=True)
    splits = grouped_splits(y, groups)

    # --- CV con los mismos 25 folds para todo ---
    per_fold = {}
    for fs_name, cols in FEATURE_SETS.items():
        for model_name, model in build_models().items():
            res = cross_validate(model, df[cols], y, cv=splits, scoring=SCORING, n_jobs=-1)
            per_fold[(model_name, fs_name)] = {m: res[f"test_{m}"] for m in SCORING}
            print(f"  {model_name:<22} {fs_name:<6} AUC={res['test_roc_auc'].mean():.3f}",
                  flush=True)

    rows = []
    for (model_name, fs_name), sc in per_fold.items():
        r = {"model": model_name, "features": fs_name, "n_features": len(FEATURE_SETS[fs_name])}
        for m in SCORING:
            r[f"{m}_mean"] = round(sc[m].mean(), 3)
            r[f"{m}_std"] = round(sc[m].std(), 3)
        rows.append(r)
    table = pd.DataFrame(rows).sort_values(["model", "features"])
    DOCS_DIR.mkdir(exist_ok=True)
    table.to_csv(DOCS_DIR / "comparison_v2_v3.csv", index=False)
    pd.DataFrame({f"{m}|{f}": sc["roc_auc"] for (m, f), sc in per_fold.items()}).to_csv(
        DOCS_DIR / "comparison_v2_v3_folds.csv", index_label="fold")
    print("\n=== CV agrupada por estrella, 5x5 folds (media ± std) ===")
    print(table.to_string(index=False))

    # --- Diferencias pareadas de AUC ---
    print("\n=== Diferencia pareada de AUC por fold (25 folds) ===")
    pairs = [("v3", "v2"), ("v2+v3", "v2"), ("v2+v3", "v3")]
    for model_name in build_models():
        for a, b in pairs:
            d = per_fold[(model_name, a)]["roc_auc"] - per_fold[(model_name, b)]["roc_auc"]
            print(f"{model_name:<22} {a:>6} - {b:<6}: media={d.mean():+.3f}, "
                  f"std={d.std(ddof=1):.3f}, gana en {(d > 0).sum()}/{len(d)} folds")
    print("(Los 25 folds reutilizan las mismas estrellas en cada repetición: "
          "no son independientes.)", flush=True)

    # --- Importancia por permutación sobre la CV (Random Forest) ---
    rf = build_models()["Random Forest"]
    out = []
    for fs_name in ("v3", "v2+v3"):
        imp = cv_permutation_importance(rf, df[FEATURE_SETS[fs_name]], y, splits)
        s = pd.DataFrame({"mean": imp.mean(), "std": imp.std(),
                          "frac_folds_pos": (imp > 0).mean()}).sort_values("mean",
                                                                         ascending=False)
        s["features"] = fs_name
        out.append(s)
        print(f"\n=== Importancia por permutación en CV, RF, {fs_name} (caída de AUC) ===")
        print(s[["mean", "std", "frac_folds_pos"]].round(4).to_string(), flush=True)
    pd.concat(out).rename_axis("feature").to_csv(DOCS_DIR / "permutation_importance_cv_v3.csv")


if __name__ == "__main__":
    main()
