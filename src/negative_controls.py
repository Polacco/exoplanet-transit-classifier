"""Controles negativos del dataset v3 (no modifica datos ni archivos de v3).

Mismo protocolo que ``compare_v2_v3.py``: StratifiedGroupKFold por kepid,
5 folds x 5 repeticiones (los MISMOS 25 folds), Random Forest.

1. Etiquetas mezcladas: se permutan las etiquetas solo en el fold de
   entrenamiento; el test se evalúa con las etiquetas reales. Esperado: AUC ~0.5.
2. Efeméride aleatoria: para cada KOI se sortea un período de la distribución
   de períodos del catálogo y una época uniforme dentro de ese período; la
   duración se mantiene. El detrend es idéntico al de v3 (máscara con la
   efeméride REAL), pero las features se calculan doblando con la falsa.
3. Solo forma: v3 sin las features que no dependen de la forma del tránsito.

Salida: ``docs/negative_controls.csv``.

Uso:
    source venv/bin/activate.fish
    python -u src/negative_controls.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_validate

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_v2_v3 import build_models, cv_permutation_importance, grouped_splits  # noqa: E402
from koi_features import (  # noqa: E402
    DATASET_V3_PATH,
    FEATURE_COLUMNS_V3,
    LC_CACHE_DIR,
    _other_mask,
    cached_kepids,
    koi_features,
    load_catalog,
    prepare_star,
    select_kois,
)
from train_model import DOCS_DIR, LABEL_MAP  # noqa: E402

OUT_CSV = DOCS_DIR / "negative_controls.csv"
SHUFFLE_SEEDS = [0, 1, 2, 3, 4]
EPHEMERIS_SEEDS = [0, 1, 2]
MAX_DRAWS = 50
SHAPE_ONLY = [
    "shape_uv_ratio", "log_asymmetry_sigma", "log_secondary_snr",
    "log_odd_even_sigma", "log_depth_chi2",
]


def rf():
    return build_models()["Random Forest"]


def cv_auc(X: pd.DataFrame, y: pd.Series, splits) -> np.ndarray:
    return cross_validate(rf(), X, y, cv=splits, scoring="roc_auc", n_jobs=-1)["test_score"]


# --------------------------------------------------------------------------- #
# 1. Etiquetas mezcladas en train
# --------------------------------------------------------------------------- #
def shuffled_label_auc(X, y, splits, seed: int) -> np.ndarray:
    def one(i, tr, te):
        warnings.filterwarnings("ignore")
        rng = np.random.default_rng([seed, i])
        y_tr = rng.permutation(y.values[tr])
        m = clone(rf()).fit(X.iloc[tr], y_tr)
        return roc_auc_score(y.values[te], m.predict_proba(X.iloc[te])[:, 1])

    return np.array(Parallel(n_jobs=-1)(
        delayed(one)(i, tr, te) for i, (tr, te) in enumerate(splits)))


# --------------------------------------------------------------------------- #
# 2. Efeméride aleatoria
# --------------------------------------------------------------------------- #
def _fake_star(kepid, targets, star_kois, period_pool, seed):
    """Features v3 de una estrella con efemérides sorteadas (detrend real)."""
    warnings.filterwarnings("ignore")
    raw = pd.read_csv(LC_CACHE_DIR / f"KIC_{kepid}_lc.csv")
    lc, _ = prepare_star(raw, star_kois)  # misma máscara que v3 (efeméride real)
    time, flux = lc["time"].values, lc["flux"].values
    rows = []
    for _, koi in targets.iterrows():
        exclude = _other_mask(time, koi, star_kois)
        D = koi["koi_duration"] / 24.0
        rng = np.random.default_rng([seed, int(koi["kepoi_name"][1:6]), int(koi["kepoi_name"][-2:])])
        # Se resortea si la efeméride falsa no es usable (P < 4D o sin tránsitos
        # con datos), para conservar exactamente las mismas filas que v3.
        for draw in range(1, MAX_DRAWS + 1):
            P = rng.choice(period_pool)
            t0 = rng.uniform(time.min(), time.min() + P)
            if P < 4 * D:
                continue
            try:
                f = koi_features(time, flux, P, t0, D, exclude)
            except ValueError:
                continue
            if f["n_transits_observed"] > 0:
                break
        else:
            raise RuntimeError(f"{koi['kepoi_name']}: sin efeméride falsa usable")
        f.update(kepoi_name=koi["kepoi_name"], fake_period=P, draws=draw)
        rows.append(f)
    return rows


def fake_ephemeris_dataset(keep: pd.Series, seed: int) -> pd.DataFrame:
    catalog = load_catalog()
    targets, all_kois = select_kois(catalog, cached_kepids())
    targets = targets[targets["kepoi_name"].isin(keep)]
    period_pool = catalog["koi_period"].dropna().values  # todo el catálogo KOI
    results = Parallel(n_jobs=-1)(
        delayed(_fake_star)(kepid, t, all_kois[all_kois["kepid"] == kepid], period_pool, seed)
        for kepid, t in targets.groupby("kepid"))
    return pd.DataFrame([r for rs in results for r in rs])


# --------------------------------------------------------------------------- #
def main() -> None:
    warnings.filterwarnings("ignore")
    v3 = pd.read_csv(DATASET_V3_PATH).sort_values("kepoi_name").reset_index(drop=True)
    y = v3["label"].map(LABEL_MAP).astype(int)
    splits = grouped_splits(y, v3["kepid"])
    X = v3[FEATURE_COLUMNS_V3]
    print(f"{len(v3)} KOIs, {v3['kepid'].nunique()} estrellas, {len(splits)} folds", flush=True)

    rows = []

    def add(control, detail, seed, aucs):
        rows.append({"control": control, "detail": detail, "seed": seed,
                     "metric": "roc_auc", "value": round(aucs.mean(), 3), "std": round(aucs.std(), 3),
                     "n_folds": len(aucs)})
        print(f"  {control:<22} {detail:<40} seed={seed}: AUC={aucs.mean():.3f} ± "
              f"{aucs.std():.3f}", flush=True)

    # Referencia
    add("referencia", "v3 completo (9 features)", "", cv_auc(X, y, splits))

    # 1. Etiquetas mezcladas
    print("\n1. Etiquetas mezcladas en train", flush=True)
    all_shuffled = []
    for s in SHUFFLE_SEEDS:
        a = shuffled_label_auc(X, y, splits, s)
        all_shuffled.append(a)
        add("etiquetas_mezcladas", "v3 completo, y permutada en train", s, a)
    add("etiquetas_mezcladas", "todas las semillas juntas", "all", np.concatenate(all_shuffled))

    # 2. Efeméride aleatoria
    print("\n2. Efeméride aleatoria", flush=True)
    importances = []
    for s in EPHEMERIS_SEEDS:
        fake = fake_ephemeris_dataset(v3["kepoi_name"], s)
        fake = v3[["kepoi_name"]].merge(fake, on="kepoi_name", how="left", validate="one_to_one")
        assert len(fake) == len(v3) and fake["kepoi_name"].equals(v3["kepoi_name"])
        print(f"  seed={s}: resorteos medios={fake['draws'].mean():.2f}, "
              f"tránsitos observados (mediana)={fake['n_transits_observed'].median():.0f}",
              flush=True)
        Xf = fake[FEATURE_COLUMNS_V3]
        add("efemeride_aleatoria", "v3 recalculado con P y t0 falsos", s, cv_auc(Xf, y, splits))
        imp = cv_permutation_importance(rf(), Xf, y, splits)
        importances.append(imp.mean())
        # Ablación dentro del control: ¿cuánto queda sin las de "ruido de la estrella"?
        add("efemeride_aleatoria", "  solo forma (5 features)", s,
            cv_auc(Xf[SHAPE_ONLY], y, splits))
    imp = pd.concat(importances, axis=1)
    imp_summary = pd.DataFrame({"mean": imp.mean(axis=1), "min_seed": imp.min(axis=1),
                                "max_seed": imp.max(axis=1)}).sort_values("mean",
                                                                          ascending=False)
    print("\n  Importancia por permutación (caída de AUC), promedio de semillas:")
    print(imp_summary.round(4).to_string(), flush=True)
    for feat, r in imp_summary.head(3).iterrows():
        rows.append({"control": "efemeride_aleatoria_importancia", "detail": feat,
                     "seed": "mean", "metric": "perm_importance_auc_drop",
                     "value": round(r["mean"], 4),
                     "std": np.nan, "n_folds": len(splits)})

    # 3. Solo forma (datos reales)
    print("\n3. Solo forma y consistencia", flush=True)
    add("solo_forma", "+".join(SHAPE_ONLY), "", cv_auc(X[SHAPE_ONLY], y, splits))
    add("solo_forma", "  sin shape_uv_ratio",
        "", cv_auc(X[[c for c in SHAPE_ONLY if c != "shape_uv_ratio"]], y, splits))

    table = pd.DataFrame(rows)
    table.to_csv(OUT_CSV, index=False)
    print(f"\nTabla: {OUT_CSV}")


if __name__ == "__main__":
    main()
