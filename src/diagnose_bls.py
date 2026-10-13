"""Diagnóstico de la extracción de features: ventana de detrend vs tope de duración BLS.

Usa SOLO curvas ya cacheadas en ``data/lc_cache/`` (sin descargas). Para una
submuestra balanceada (semilla fija) prueba todas las combinaciones de
``window_length`` (detrend) y tope de duración de BLS, y reporta por cada una:

(a) fracción de estrellas cuya duración detectada queda en el tope de la grilla,
(b) mediana de SNR,
(c) ROC-AUC en validación cruzada estratificada de 5 folds (RandomForest simple
    + SimpleImputer). Solo CV: no se usa ningún test set para elegir parámetros.

Salidas: ``docs/bls_diagnostic.csv`` y ``docs/bls_diagnostic_stars.png``.

Uso:
    source venv/bin/activate.fish
    python src/diagnose_bls.py
"""

from __future__ import annotations

import sys
import warnings
from itertools import product
from pathlib import Path

import astropy.units as u
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.timeseries import BoxLeastSquares
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))

from features import (  # noqa: E402
    CATALOG_PATH,
    LC_CACHE_DIR,
    _transit_shape_features,
)
from preprocessing import detrend, normalize_flux, remove_outliers  # noqa: E402

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"
OUT_CSV = DOCS_DIR / "bls_diagnostic.csv"
OUT_FIG = DOCS_DIR / "bls_diagnostic_stars.png"

WINDOWS = [101, 301, 501, 1001]
MAX_DURATIONS = [0.15, 0.3]
PERIOD_RANGE = (0.5, 20.0)
MIN_DURATION = 0.01
N_PERIODS = 1500
N_DURATIONS = 20
N_PER_CLASS = 50
N_STARS_FIG = 6
SEED = 42
N_FOLDS = 5
N_REPEATS = 3  # repeticiones del CV de 5 folds: con ~100 estrellas un solo CV es muy ruidoso

FEATURES = [
    "period", "depth", "duration", "snr", "power", "n_transits",
    "odd_even_diff", "secondary_depth", "depth_duration_ratio",
    "secondary_ratio", "flux_std",
]


def pick_sample() -> pd.DataFrame:
    """Submuestra balanceada de estrellas que tengan curva en caché."""
    catalog = pd.read_csv(CATALOG_PATH).dropna(subset=["kepid"])
    catalog["kepid"] = catalog["kepid"].astype(int)
    catalog = catalog.drop_duplicates("kepid")
    cached = {int(p.stem.split("_")[1]) for p in LC_CACHE_DIR.glob("KIC_*_lc.csv")}
    catalog = catalog[catalog["kepid"].isin(cached)]
    parts = []
    for disp in ("CONFIRMED", "FALSE POSITIVE"):
        sub = catalog[catalog["koi_disposition"] == disp]
        parts.append(sub.sample(n=min(N_PER_CLASS, len(sub)), random_state=SEED))
    return pd.concat(parts).reset_index(drop=True)


def run_bls(lc: pd.DataFrame, max_duration: float):
    """BLS (método rápido) con grilla de duraciones acotada a ``max_duration``."""
    # Requisito de astropy: toda duración debe ser menor que el período mínimo.
    assert max_duration < PERIOD_RANGE[0], "la duración máxima debe ser < período mínimo"
    model = BoxLeastSquares(lc["time"].values * u.day, lc["flux"].values,
                            dy=lc["flux_err"].values)
    periods = np.linspace(*PERIOD_RANGE, N_PERIODS)
    durations = np.linspace(MIN_DURATION, max_duration, N_DURATIONS) * u.day
    res = model.power(periods, durations, method="fast", oversample=10)
    i = int(np.argmax(res.power))
    err = res.depth_err[i]
    return {
        "period": float(res.period[i].value),
        "depth": float(res.depth[i]),
        "duration": float(res.duration[i].value),
        "t0": float(res.transit_time[i].value),
        "snr": float(res.depth[i] / err) if err != 0 else np.inf,
        "power": float(res.power[i]),
    }


def features_from(lc: pd.DataFrame, r: dict) -> dict:
    time, flux = lc["time"].values, lc["flux"].values
    odd_even, sec = _transit_shape_features(time, flux, r["period"], r["duration"], r["t0"])
    depth, dur = r["depth"], r["duration"]
    return {
        "period": r["period"], "depth": depth, "duration": dur, "snr": r["snr"],
        "power": r["power"],
        "n_transits": int(np.floor((time.max() - time.min()) / r["period"])),
        "odd_even_diff": odd_even, "secondary_depth": sec,
        "depth_duration_ratio": depth / dur if dur > 0 else np.nan,
        "secondary_ratio": sec / depth if depth > 0 else np.nan,
        "flux_std": float(np.std(flux)),
    }


def prepare(raw: pd.DataFrame, window: int) -> pd.DataFrame:
    assert window % 2 == 1, "window_length debe ser impar"
    return normalize_flux(detrend(remove_outliers(raw, sigma=5), window_length=window))


def main() -> None:
    sample = pick_sample()
    print(f"Submuestra: {sample['koi_disposition'].value_counts().to_dict()}")
    raws = {int(k): pd.read_csv(LC_CACHE_DIR / f"KIC_{int(k)}_lc.csv") for k in sample["kepid"]}

    rows_by_cfg: dict[tuple, list] = {cfg: [] for cfg in product(WINDOWS, MAX_DURATIONS)}
    lcs_for_fig: dict[int, dict[int, pd.DataFrame]] = {}
    fig_ids = list(
        sample.groupby("koi_disposition").head(N_STARS_FIG // 2)["kepid"].astype(int)
    )

    for n, (kepid, label) in enumerate(zip(sample["kepid"], sample["koi_disposition"]), 1):
        kepid = int(kepid)
        for window in WINDOWS:
            try:
                lc = prepare(raws[kepid], window)
            except Exception as exc:  # noqa: BLE001
                print(f"  [fail detrend] KIC {kepid} w={window}: {exc}")
                continue
            if kepid in fig_ids:
                lcs_for_fig.setdefault(kepid, {})[window] = lc
            for max_dur in MAX_DURATIONS:
                r = run_bls(lc, max_dur)
                f = features_from(lc, r)
                f.update(kepid=kepid, label=int(label == "CONFIRMED"),
                         at_cap=bool(r["duration"] >= max_dur - 1e-9))
                rows_by_cfg[(window, max_dur)].append(f)
        if n % 10 == 0:
            print(f"  {n}/{len(sample)} estrellas procesadas")

    # ---- Tabla comparativa ----
    cv = RepeatedStratifiedKFold(n_splits=N_FOLDS, n_repeats=N_REPEATS, random_state=SEED)
    out = []
    for (window, max_dur), rows in rows_by_cfg.items():
        df = pd.DataFrame(rows)
        clf = make_pipeline(SimpleImputer(strategy="median"),
                            RandomForestClassifier(n_estimators=200, random_state=SEED))
        X = df[FEATURES].replace([np.inf, -np.inf], np.nan)
        scores = cross_val_score(clf, X, df["label"], cv=cv, scoring="roc_auc")
        out.append({
            "window_length": window,
            "window_days": round(window * 29.4 / 60 / 24, 1),
            "max_duration": max_dur,
            "n_stars": len(df),
            "frac_at_cap": round(df["at_cap"].mean(), 3),
            "median_snr": round(df["snr"].median(), 2),
            "median_duration": round(df["duration"].median(), 3),
            "cv_auc_mean": round(scores.mean(), 3),
            "cv_auc_std": round(scores.std(), 3),
        })
    table = pd.DataFrame(out)
    DOCS_DIR.mkdir(exist_ok=True)
    table.to_csv(OUT_CSV, index=False)
    print("\n" + table.to_string(index=False))
    print(f"\nTabla guardada en {OUT_CSV}")

    make_figure(lcs_for_fig, sample)


def make_figure(lcs: dict[int, dict[int, pd.DataFrame]], sample: pd.DataFrame) -> None:
    """Por estrella: fila superior = curva detrendeada, fila inferior = doblada en fase."""
    ids = list(lcs)
    labels = dict(zip(sample["kepid"].astype(int), sample["koi_disposition"]))
    max_dur = min(MAX_DURATIONS)
    fig, axes = plt.subplots(2 * len(ids), len(WINDOWS),
                             figsize=(4.2 * len(WINDOWS), 3.0 * 2 * len(ids)))
    for i, kepid in enumerate(ids):
        for j, window in enumerate(WINDOWS):
            ax_t, ax_f = axes[2 * i, j], axes[2 * i + 1, j]
            lc = lcs[kepid].get(window)
            if lc is None:
                continue
            r = run_bls(lc, max_dur)
            ax_t.plot(lc["time"], lc["flux"], ".", ms=1, color="k", alpha=0.5)
            ax_t.set_title(f"KIC {kepid} ({labels[kepid][:5]}) | w={window} "
                           f"(~{window * 29.4 / 60 / 24:.0f} d)", fontsize=8)
            phase = ((lc["time"].values - r["t0"]) / r["period"]) % 1.0
            phase[phase > 0.5] -= 1.0
            ax_f.plot(phase, lc["flux"], ".", ms=1, color="k", alpha=0.4)
            ax_f.set_title(f"P={r['period']:.3f} d, dur={r['duration']:.3f} d, "
                           f"SNR={r['snr']:.1f}", fontsize=8)
            lo, hi = np.percentile(lc["flux"], [0.5, 99.5])
            ax_t.set_ylim(lo, hi)
            ax_f.set_ylim(lo, hi)
            if i == len(ids) - 1:
                ax_f.set_xlabel("fase")
    fig.suptitle(f"Detrend (filas impares) y curva doblada (filas pares); BLS con tope {max_dur} d",
                 y=1.0)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=80, bbox_inches="tight")
    plt.close(fig)
    print(f"Figura guardada en {OUT_FIG}")


if __name__ == "__main__":
    main()
