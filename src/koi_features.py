"""Dataset v3: un registro por KOI, doblando la curva con la efeméride del catálogo.

A diferencia de v1/v2 (una fila por estrella, período buscado con BLS), acá cada
fila es un candidato (KOI) y la curva se dobla con ``koi_period`` y
``koi_time0bk`` del catálogo. ``koi_duration`` define las ventanas de tránsito.

Fuga de información: esas tres columnas (``EPHEMERIS_COLUMNS``) se usan SOLO
para doblar y enmascarar; ninguna columna ``koi_*`` entra como feature. Se
verifica al importar el módulo (``_assert_no_leakage``) y en los tests.

Pipeline por estrella (solo curvas ya cacheadas en ``data/lc_cache/``):

1. Máscara con los tránsitos de TODOS los KOIs con efeméride de la estrella
   (incluidos CANDIDATE), con margen: |t - t_tránsito| < MASK_HALF_WIDTH * D.
2. Detrend (Savitzky-Golay vía lightkurve, ``window_length=101`` ~2.1 d) sin
   usar los puntos enmascarados; normalización; sigma-clipping a 5σ fuera de
   la máscara (dentro solo se recortan outliers hacia arriba, para no borrar
   eclipses profundos).
3. Por cada KOI CONFIRMED / FALSE POSITIVE: se excluyen los tránsitos de los
   OTROS KOIs de la estrella y se calculan las features de forma.

Uso:
    source venv/bin/activate.fish
    python -u src/koi_features.py --build    # escribe data/features_dataset_v3.csv
    python -u src/koi_features.py --plot     # docs/koi_folded_examples.png
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))

from preprocessing import detrend, normalize_flux  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = PROJECT_ROOT / "docs"
LC_CACHE_DIR = DATA_DIR / "lc_cache"
CATALOG_V3_PATH = DATA_DIR / "koi_catalog_v3.csv"
DATASET_V3_PATH = DATA_DIR / "features_dataset_v3.csv"
FIG_PATH = DOCS_DIR / "koi_folded_examples.png"

CATALOG_COLUMNS = [
    "kepoi_name", "kepid", "koi_disposition", "koi_period", "koi_time0bk",
    "koi_duration", "koi_tce_plnt_num",
]
# Solo para doblar y enmascarar. NUNCA como features.
EPHEMERIS_COLUMNS = ["koi_period", "koi_time0bk", "koi_duration"]
FORBIDDEN_FEATURE_PREFIX = "koi_"
LABELS = ("CONFIRMED", "FALSE POSITIVE")

CADENCE_DAYS = 29.4244 / 60 / 24  # cadencia larga de Kepler
DETREND_WINDOW = 101               # puntos (~2.1 d), elegido en docs/bls_diagnostic_*.csv
MASK_HALF_WIDTH = 1.0              # en unidades de D: enmascara 2D (margen D/2 a cada lado)
SIGMA_CLIP = 5.0
MIN_OOT_POINTS = 10
SHAPE_MIN_SNR = 5.0                # por debajo, el cociente U/V es puro ruido -> NaN
SEED = 42

FEATURE_COLUMNS_V3 = [
    "log_depth_ppm",            # profundidad (mediana fuera - mediana dentro), con signo
    "log_depth_err_ppm",        # su incertidumbre
    "log_transit_snr",          # depth / depth_err, con signo
    "log_n_transits_observed",  # tránsitos con datos reales
    "log_odd_even_sigma",       # |prof. impares - pares| / incertidumbre
    "log_secondary_snr",        # profundidad en fase 0.5 / incertidumbre, con signo
    "log_asymmetry_sigma",      # |mitad de entrada - mitad de salida| / incertidumbre
    "shape_uv_ratio",           # prof. tercio central / prof. media (U ~1, V ~1.7)
    "log_depth_chi2",           # chi2 reducido de las profundidades individuales
]


def _assert_no_leakage(columns) -> None:
    bad = [c for c in columns if c.startswith(FORBIDDEN_FEATURE_PREFIX)]
    if bad:
        raise AssertionError(f"Columnas del catálogo usadas como features: {bad}")


_assert_no_leakage(FEATURE_COLUMNS_V3)


# --------------------------------------------------------------------------- #
# Utilidades numéricas
# --------------------------------------------------------------------------- #
def slog10(x: float) -> float:
    """log10(1 + |x|) conservando el signo (para features que pueden ser < 0)."""
    return float(np.sign(x) * np.log10(1.0 + abs(x))) if np.isfinite(x) else np.nan


def log1p10(x: float) -> float:
    return float(np.log10(1.0 + x)) if np.isfinite(x) and x >= 0 else np.nan


def robust_std(x: np.ndarray) -> float:
    """Desviación estándar estimada con la MAD (insensible a outliers)."""
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def time_from_midtransit(time: np.ndarray, period: float, t0: float) -> np.ndarray:
    """Tiempo (días) al centro de tránsito más cercano, en [-P/2, P/2)."""
    return ((time - t0 + 0.5 * period) % period) - 0.5 * period


def transits_mask(time: np.ndarray, kois: pd.DataFrame,
                  half_width: float = MASK_HALF_WIDTH) -> np.ndarray:
    """Unión de ventanas |dt| < half_width * D para todos los KOIs dados."""
    mask = np.zeros(len(time), dtype=bool)
    for _, k in kois.iterrows():
        dt = time_from_midtransit(time, k["koi_period"], k["koi_time0bk"])
        mask |= np.abs(dt) < half_width * k["koi_duration"] / 24.0
    return mask


# --------------------------------------------------------------------------- #
# Catálogo
# --------------------------------------------------------------------------- #
def load_catalog(force_download: bool = False) -> pd.DataFrame:
    """Tabla cumulative de KOI con efemérides (caché en data/koi_catalog_v3.csv)."""
    if CATALOG_V3_PATH.exists() and not force_download:
        return pd.read_csv(CATALOG_V3_PATH)
    from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive

    table = NasaExoplanetArchive.query_criteria(
        table="cumulative", select=",".join(CATALOG_COLUMNS))
    df = table.to_pandas()
    df.to_csv(CATALOG_V3_PATH, index=False)
    return df


def cached_kepids() -> set[int]:
    return {int(p.stem.split("_")[1]) for p in LC_CACHE_DIR.glob("KIC_*_lc.csv")}


def select_kois(catalog: pd.DataFrame, kepids: set[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve (targets, all_kois).

    - ``targets``: KOIs CONFIRMED / FALSE POSITIVE con efeméride completa, de
      estrellas cacheadas. Una fila por KOI.
    - ``all_kois``: TODOS los KOIs con efeméride de esas estrellas (cualquier
      disposición), usados solo para enmascarar tránsitos.
    """
    cat = catalog.dropna(subset=["kepid"] + EPHEMERIS_COLUMNS).copy()
    cat["kepid"] = cat["kepid"].astype(int)
    cat = cat[cat["kepid"].isin(kepids) & (cat["koi_duration"] > 0)
              & (cat["koi_period"] > 0)]
    targets = cat[cat["koi_disposition"].isin(LABELS)].reset_index(drop=True)
    return targets, cat.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Preprocesamiento por estrella
# --------------------------------------------------------------------------- #
def prepare_star(raw: pd.DataFrame, star_kois: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Detrend con máscara de tránsitos + normalización + clipping asimétrico.

    La ventana es ``DETREND_WINDOW`` salvo que algún tránsito sea tan largo que
    la zona enmascarada ocupe más de 1/3 de la ventana: ahí se agranda para que
    el filtro tenga datos a ambos lados del hueco.
    """
    raw = raw[np.isfinite(raw["time"]) & np.isfinite(raw["flux"])].reset_index(drop=True)
    time = raw["time"].values
    mask = transits_mask(time, star_kois)
    if mask.mean() > 0.8:
        raise ValueError("más del 80% de la curva cae en ventanas de tránsito")

    masked_pts = 2 * MASK_HALF_WIDTH * star_kois["koi_duration"].max() / 24 / CADENCE_DAYS
    window = max(DETREND_WINDOW, int(np.ceil(3 * masked_pts)))
    window += 1 - window % 2  # impar

    lc = normalize_flux(detrend(raw, window_length=window, mask=mask))
    resid = lc["flux"].values - np.median(lc["flux"].values[~mask])
    s = robust_std(lc["flux"].values[~mask])
    bad = (~mask & (np.abs(resid) > SIGMA_CLIP * s)) | (mask & (resid > SIGMA_CLIP * s))
    return lc.loc[~bad].reset_index(drop=True), window


# --------------------------------------------------------------------------- #
# Features de un KOI
# --------------------------------------------------------------------------- #
def koi_features(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
    duration_days: float,
    exclude: np.ndarray | None = None,
) -> dict:
    """Features de forma de un KOI a partir de la curva doblada con su efeméride.

    ``exclude`` marca puntos a ignorar (tránsitos de otros KOIs de la estrella).
    Los errores de una mediana de n puntos se estiman como 1.2533·σ/√n, con σ
    la dispersión robusta fuera de tránsito.
    """
    D = duration_days
    if exclude is not None:
        time, flux = time[~exclude], flux[~exclude]
    dt = time_from_midtransit(time, period, t0)
    epoch = np.round((time - t0) / period).astype(int)
    half_p = 0.5 * period

    in_tr = np.abs(dt) < D / 2
    outer = min(D + max(2 * D, 0.5), half_p - D)
    oot = (np.abs(dt) > D) & (np.abs(dt) < outer)
    if oot.sum() < MIN_OOT_POINTS:
        oot = np.abs(dt) > D
    if in_tr.sum() == 0 or oot.sum() < MIN_OOT_POINTS:
        raise ValueError("sin puntos suficientes dentro o fuera de tránsito")

    base = np.median(flux[oot])
    sigma = robust_std(flux[oot])
    n_oot = int(oot.sum())

    def med_depth(sel: np.ndarray, base_=base, n_ref=n_oot) -> tuple[float, float]:
        n = int(sel.sum())
        if n == 0 or sigma == 0:
            return np.nan, np.nan
        return base_ - np.median(flux[sel]), 1.2533 * sigma * np.sqrt(1 / n + 1 / n_ref)

    def sigma_diff(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = int(a.sum()), int(b.sum())
        if na == 0 or nb == 0 or sigma == 0:
            return np.nan
        err = 1.2533 * sigma * np.sqrt(1 / na + 1 / nb)
        return abs(np.median(flux[a]) - np.median(flux[b])) / err

    # Profundidad y SNR
    depth, depth_err = med_depth(in_tr)
    snr = depth / depth_err if depth_err > 0 else np.nan

    # Tránsitos con datos reales: al menos la mitad de los puntos esperados.
    min_pts = max(1, int(0.5 * D / CADENCE_DAYS))
    epochs, counts = np.unique(epoch[in_tr], return_counts=True)
    observed = epochs[counts >= min_pts]
    n_obs = len(observed)

    # Impares vs pares (el signo no importa: cuál es "impar" es arbitrario)
    odd_even = sigma_diff(in_tr & (epoch % 2 == 1), in_tr & (epoch % 2 == 0))

    # Eclipse secundario en fase 0.5 (con su propia línea de base local)
    dt2 = time_from_midtransit(time, period, t0 + half_p)
    not_primary = np.abs(dt) > D
    in_sec = (np.abs(dt2) < D / 2) & not_primary
    oot_sec = (np.abs(dt2) > D) & (np.abs(dt2) < outer) & not_primary
    sec_snr = np.nan
    if oot_sec.sum() >= MIN_OOT_POINTS:
        d2, e2 = med_depth(in_sec, np.median(flux[oot_sec]), int(oot_sec.sum()))
        sec_snr = d2 / e2 if e2 and e2 > 0 else np.nan

    # Simetría: mitad de entrada vs mitad de salida
    asym = sigma_diff(in_tr & (dt < 0), in_tr & (dt > 0))

    # Forma U vs V: caja ~1.0, planeta con oscurecimiento al limbo ~1.1-1.3,
    # triángulo (binaria rasante) ~1.7. Solo si la señal es significativa.
    center = np.abs(dt) < D / 6
    shape = np.nan
    if center.sum() >= 3 and np.isfinite(snr) and snr >= SHAPE_MIN_SNR:
        d_mean = base - flux[in_tr].mean()
        if d_mean > 0:
            shape = float(np.clip((base - flux[center].mean()) / d_mean, 0, 3))

    # Consistencia de las profundidades individuales (chi2 reducido)
    chi2 = np.nan
    if n_obs >= 3:
        d_k, e_k = [], []
        for k in observed:
            sel = in_tr & (epoch == k)
            d_k.append(base - np.median(flux[sel]))
            e_k.append(1.2533 * sigma / np.sqrt(sel.sum()))
        d_k, e_k = np.array(d_k), np.array(e_k)
        w = 1 / e_k**2
        mean_w = np.sum(w * d_k) / np.sum(w)
        chi2 = float(np.sum(((d_k - mean_w) / e_k) ** 2) / (n_obs - 1))

    return {
        "log_depth_ppm": slog10(depth * 1e6),
        "log_depth_err_ppm": log1p10(depth_err * 1e6),
        "log_transit_snr": slog10(snr),
        "log_n_transits_observed": log1p10(n_obs),
        "log_odd_even_sigma": log1p10(odd_even),
        "log_secondary_snr": slog10(sec_snr),
        "log_asymmetry_sigma": log1p10(asym),
        "shape_uv_ratio": shape,
        "log_depth_chi2": log1p10(chi2),
        # Diagnósticos (no son features)
        "n_transits_observed": n_obs,
    }


def _other_mask(time: np.ndarray, koi: pd.Series, star_kois: pd.DataFrame) -> np.ndarray:
    """Tránsitos de los otros KOIs de la estrella, salvo donde pisan los del propio KOI."""
    others = star_kois[star_kois["kepoi_name"] != koi["kepoi_name"]]
    if others.empty:
        return np.zeros(len(time), dtype=bool)
    own = transits_mask(time, koi.to_frame().T)
    return transits_mask(time, others) & ~own


# --------------------------------------------------------------------------- #
# Construcción del dataset
# --------------------------------------------------------------------------- #
def process_star(kepid: int, targets: pd.DataFrame, star_kois: pd.DataFrame):
    warnings.filterwarnings("ignore")
    rows, fails = [], []
    try:
        raw = pd.read_csv(LC_CACHE_DIR / f"KIC_{kepid}_lc.csv")
        lc, window = prepare_star(raw, star_kois)
    except Exception as exc:  # noqa: BLE001
        return [], [(k, f"{type(exc).__name__}: {exc}") for k in targets["kepoi_name"]]
    time, flux = lc["time"].values, lc["flux"].values
    for _, koi in targets.iterrows():
        try:
            f = koi_features(time, flux, koi["koi_period"], koi["koi_time0bk"],
                             koi["koi_duration"] / 24.0, _other_mask(time, koi, star_kois))
            if f["n_transits_observed"] == 0:
                raise ValueError("ningún tránsito con datos en los quarters cacheados")
            f.update(kepoi_name=koi["kepoi_name"], kepid=kepid,
                     label=koi["koi_disposition"], detrend_window=window)
            rows.append(f)
        except Exception as exc:  # noqa: BLE001
            fails.append((koi["kepoi_name"], f"{type(exc).__name__}: {exc}"))
    return rows, fails


def build_dataset(n_jobs: int = -1) -> pd.DataFrame:
    targets, all_kois = select_kois(load_catalog(), cached_kepids())
    print(f"KOIs candidatos: {len(targets)} "
          f"{targets['koi_disposition'].value_counts().to_dict()}, "
          f"{targets['kepid'].nunique()} estrellas", flush=True)

    groups = targets.groupby("kepid")
    results = Parallel(n_jobs=n_jobs, verbose=2)(
        delayed(process_star)(kepid, t, all_kois[all_kois["kepid"] == kepid])
        for kepid, t in groups
    )
    rows = [r for rs, _ in results for r in rs]
    fails = [f for _, fs in results for f in fs]

    df = pd.DataFrame(rows)
    meta = ["kepoi_name", "kepid", "label"]
    _assert_no_leakage(FEATURE_COLUMNS_V3)
    df = df.sort_values("kepoi_name")
    df[meta + FEATURE_COLUMNS_V3].to_csv(DATASET_V3_PATH, index=False)

    per_star = df.groupby("kepid").size()
    print(f"\nFilas v3: {len(df)} {df['label'].value_counts().to_dict()}")
    print(f"Estrellas: {df['kepid'].nunique()}, con más de un KOI: {(per_star > 1).sum()}")
    print(f"Ventana de detrend agrandada (tránsitos largos) en "
          f"{(df['detrend_window'] > DETREND_WINDOW).sum()} KOIs")
    print(f"Descartados: {len(fails)}")
    for name, msg in fails:
        print(f"  {name}: {msg}")
    print(f"NaN por feature:\n{df[FEATURE_COLUMNS_V3].isna().sum().to_string()}")
    print(f"\nDataset: {DATASET_V3_PATH}")
    return df


# --------------------------------------------------------------------------- #
# Figura de control
# --------------------------------------------------------------------------- #
def plot_examples(n_per_class: int = 3) -> None:
    """Curvas dobladas con la efeméride del catálogo: tránsito y fase 0.5."""
    data = pd.read_csv(DATASET_V3_PATH)
    _, all_kois = select_kois(load_catalog(), cached_kepids())
    eligible = data[data["log_n_transits_observed"] >= np.log10(1 + 2)]
    picks = pd.concat([
        eligible[eligible["label"] == lab].sample(n_per_class, random_state=SEED)
        for lab in LABELS
    ])

    fig, axes = plt.subplots(len(picks), 2, figsize=(11, 2.6 * len(picks)))
    for row_axes, (_, row) in zip(axes, picks.iterrows()):
        star_kois = all_kois[all_kois["kepid"] == row["kepid"]]
        koi = star_kois[star_kois["kepoi_name"] == row["kepoi_name"]].iloc[0]
        lc, _ = prepare_star(pd.read_csv(LC_CACHE_DIR / f"KIC_{row['kepid']}_lc.csv"),
                             star_kois)
        keep = ~_other_mask(lc["time"].values, koi, star_kois)
        t, f = lc["time"].values[keep], lc["flux"].values[keep]
        P, t0, D = koi["koi_period"], koi["koi_time0bk"], koi["koi_duration"] / 24
        lo, hi = np.percentile(f, [0.2, 99.8])
        for ax, center, name in ((row_axes[0], t0, "tránsito"),
                                 (row_axes[1], t0 + P / 2, "fase 0.5")):
            dt_h = time_from_midtransit(t, P, center) * 24
            sel = np.abs(dt_h) < 3 * D * 24
            ax.plot(dt_h[sel], f[sel], ".", ms=2, color="0.6")
            edges = np.linspace(-3 * D * 24, 3 * D * 24, 37)
            idx = np.digitize(dt_h[sel], edges)
            med = [np.median(f[sel][idx == i]) if (idx == i).any() else np.nan
                   for i in range(1, len(edges))]
            ax.plot((edges[:-1] + edges[1:]) / 2, med, "r-", lw=1.5)
            ax.axvspan(-D * 12, D * 12, color="C0", alpha=0.1)
            ax.set_ylim(min(lo, np.nanmin(med)) - 1e-4, hi)
            ax.set_xlabel(f"horas desde {name}")
            ax.ticklabel_format(axis="y", useOffset=False)
        row_axes[0].set_title(
            f"{row['kepoi_name']} ({row['label']}) P={P:.3f} d D={D * 24:.1f} h | "
            f"SNR={np.sign(row['log_transit_snr']) * (10 ** abs(row['log_transit_snr']) - 1):.0f}, "
            f"U/V={row['shape_uv_ratio']:.2f}", fontsize=8)
        row_axes[1].set_title(
            f"odd/even log={row['log_odd_even_sigma']:.2f}, "
            f"secundario log={row['log_secondary_snr']:.2f}", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=90)
    plt.close(fig)
    print(f"Figura: {FIG_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset v3 (una fila por KOI).")
    parser.add_argument("--build", action="store_true", help="Construye el dataset v3.")
    parser.add_argument("--plot", action="store_true", help="Figura con 6 curvas dobladas.")
    args = parser.parse_args()
    if args.build:
        build_dataset()
    if args.plot:
        plot_examples()
    if not (args.build or args.plot):
        parser.print_help()
