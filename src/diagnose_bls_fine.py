"""Barrido fino de ``window_length`` de detrend con tope de duración BLS fijo en 0.30 d.

Misma submuestra (100 estrellas, semilla 42) y misma metodología que
``diagnose_bls.py``, pero paralelizando por estrella con joblib.
Resultado en ``docs/bls_diagnostic_fine.csv``.

Uso:
    source venv/bin/activate.fish
    python -u src/diagnose_bls_fine.py
"""

from __future__ import annotations

import sys
import time as _time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))

from diagnose_bls import (  # noqa: E402
    DOCS_DIR,
    FEATURES,
    N_FOLDS,
    N_REPEATS,
    SEED,
    features_from,
    pick_sample,
    prepare,
    run_bls,
)
from features import LC_CACHE_DIR  # noqa: E402

WINDOWS = [51, 75, 101, 151]
MAX_DURATION = 0.3
OUT_CSV = DOCS_DIR / "bls_diagnostic_fine.csv"


def process_star(kepid: int, label: str) -> list[dict]:
    """Extrae features de una estrella para cada ventana (corre en un worker)."""
    warnings.filterwarnings("ignore")
    raw = pd.read_csv(LC_CACHE_DIR / f"KIC_{kepid}_lc.csv")
    rows = []
    for window in WINDOWS:
        try:
            lc = prepare(raw, window)
            r = run_bls(lc, MAX_DURATION)
        except Exception as exc:  # noqa: BLE001
            print(f"  [fail] KIC {kepid} w={window}: {exc}", flush=True)
            continue
        f = features_from(lc, r)
        f.update(kepid=kepid, window=window, label=int(label == "CONFIRMED"),
                 at_cap=bool(r["duration"] >= MAX_DURATION - 1e-9))
        rows.append(f)
    print(f"  [ok] KIC {kepid} ({label})", flush=True)
    return rows


def main() -> None:
    t0 = _time.time()
    sample = pick_sample()
    print(f"Submuestra: {sample['koi_disposition'].value_counts().to_dict()}", flush=True)
    print(f"Ventanas: {WINDOWS}, tope de duración: {MAX_DURATION} d", flush=True)

    results = Parallel(n_jobs=-1)(
        delayed(process_star)(int(k), lab)
        for k, lab in zip(sample["kepid"], sample["koi_disposition"])
    )
    all_rows = pd.DataFrame([r for star in results for r in star])
    print(f"Extracción lista en {_time.time() - t0:.0f} s. Calculando CV...", flush=True)

    cv = RepeatedStratifiedKFold(n_splits=N_FOLDS, n_repeats=N_REPEATS, random_state=SEED)
    out = []
    for window in WINDOWS:
        df = all_rows[all_rows["window"] == window]
        clf = make_pipeline(SimpleImputer(strategy="median"),
                            RandomForestClassifier(n_estimators=200, random_state=SEED))
        X = df[FEATURES].replace([np.inf, -np.inf], np.nan)
        scores = cross_val_score(clf, X, df["label"], cv=cv, scoring="roc_auc", n_jobs=-1)
        out.append({
            "window_length": window,
            "window_days": round(window * 29.4 / 60 / 24, 1),
            "max_duration": MAX_DURATION,
            "n_stars": len(df),
            "frac_at_cap": round(df["at_cap"].mean(), 3),
            "median_snr": round(df["snr"].median(), 2),
            "median_duration": round(df["duration"].median(), 3),
            "cv_auc_mean": round(scores.mean(), 3),
            "cv_auc_std": round(scores.std(), 3),
        })
        print(f"  w={window}: AUC={scores.mean():.3f} ± {scores.std():.3f}", flush=True)

    table = pd.DataFrame(out)
    table.to_csv(OUT_CSV, index=False)
    print("\n" + table.to_string(index=False), flush=True)
    print(f"\nTabla guardada en {OUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
