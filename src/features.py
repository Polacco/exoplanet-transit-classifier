"""Construcción de un dataset etiquetado de features para clasificación de exoplanetas.

Pipeline por objetivo:

1. La muestra se obtiene del catálogo "cumulative" de KOI (NasaExoplanetArchive),
   con clases balanceadas CONFIRMED / FALSE POSITIVE (semilla fija).
2. Para cada ``kepid`` se descarga su curva de luz de Kepler (cadencia larga,
   primeros 4 quarters) con lightkurve y se cachea localmente.
3. Se aplica el pipeline: remove_outliers -> detrend -> normalize_flux ->
   find_transits (BLS).
4. Se calculan features propias a partir de la curva de luz (NUNCA columnas
   ``koi_*`` del catálogo, que son resultados del pipeline de la NASA y
   causarían data leakage).
5. Los resultados se guardan de forma incremental en
   ``data/features_dataset_v2.csv`` y los fallos en ``data/failed_targets.log``,
   de modo que una ejecución interrumpida se pueda retomar sin repetir
   objetivos ya procesados.

Uso:
    source venv/bin/activate
    python src/features.py --n-per-class 5
    python src/features.py --recompute      # reprocesa todo data/lc_cache/ -> v2

Versiones del dataset:
- v1: ``window_length=1001`` en detrend, ``snr``/``power`` crudos (11 features).
- v2: ``window_length=101`` (ver ``docs/bls_diagnostic_*.csv``), ``log_snr`` y
  ``log_power`` (log10(1+x)) y ``duration_period_ratio`` (12 features).
"""

from __future__ import annotations

import argparse
import re
import signal
import sys
from pathlib import Path

import lightkurve as lk
from joblib import Parallel, delayed
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from preprocessing import detrend, normalize_flux, remove_outliers
from transit_detection import find_transits

# Timeouts de red para las consultas a MAST (evita bloqueos indefinidos).
from astroquery.mast import conf as mast_conf

mast_conf.timeout = 120  # segundos

# Timeout total por objetivo (descarga + procesamiento), en segundos.
TARGET_TIMEOUT = 600


class TargetTimeoutError(TimeoutError):
    """Se excedió el tiempo máximo permitido para procesar un objetivo."""


def _alarm_handler(signum, frame):  # noqa: ARG001
    raise TargetTimeoutError(f"Timeout de {TARGET_TIMEOUT} s excedido")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LC_CACHE_DIR = DATA_DIR / "lc_cache"
CATALOG_PATH = DATA_DIR / "koi_catalog.csv"
DATASET_PATH = DATA_DIR / "features_dataset_v2.csv"
DATASET_V1_PATH = DATA_DIR / "features_dataset_v1.csv"
LEGACY_DATASET_PATH = DATA_DIR / "features_dataset.csv"  # nombre previo a las versiones

# Ventana (en puntos, ~2.1 d en cadencia larga de Kepler) del filtro de detrend.
# Elegida por CV en docs/bls_diagnostic_step1.csv / bls_diagnostic_fine.csv:
# la ventana larga (1001, ~20 d) dejaba variabilidad estelar lenta que BLS
# confundía con tránsitos. NOTA: está calibrada para cadencia larga; con
# cadencia corta (1 min) habría que pasar una ventana mucho mayor.
DETREND_WINDOW = 101
FAILED_LOG = DATA_DIR / "failed_targets.log"

FEATURE_COLUMNS = [
    "kepid",
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
    "label",
]


# --------------------------------------------------------------------------- #
# 1. Catálogo KOI
# --------------------------------------------------------------------------- #
def get_koi_catalog(force_download: bool = False) -> pd.DataFrame:
    """Descarga (o carga desde caché) la tabla "cumulative" de KOI.

    Solo conserva las columnas ``kepid`` (identificador de estrella) y
    ``koi_disposition`` (etiqueta: CONFIRMED / FALSE POSITIVE / CANDIDATE).
    """
    if CATALOG_PATH.exists() and not force_download:
        return pd.read_csv(CATALOG_PATH)

    from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive

    table = NasaExoplanetArchive.query_criteria(
        table="cumulative", select="kepid,koi_disposition"
    )
    df = table.to_pandas()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CATALOG_PATH, index=False)
    return df


# --------------------------------------------------------------------------- #
# 2. Muestreo balanceado
# --------------------------------------------------------------------------- #
def build_sample(
    n_per_class: int,
    seed: int = 42,
    catalog: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Construye una muestra balanceada de objetivos KOI.

    Toma ``n_per_class`` objetivos CONFIRMED y ``n_per_class`` FALSE POSITIVE
    (los CANDIDATE se ignoran), con semilla fija para reproducibilidad.
    Si una estrella aparece varias veces en el catálogo se conserva una sola
    fila por ``kepid``. La selección usa una permutación fija, por lo que una
    muestra de tamaño N es siempre un subconjunto de una muestra de tamaño
    M > N (útil para escalar n_per_class incrementalmente).
    """
    if catalog is None:
        catalog = get_koi_catalog()

    catalog = catalog.dropna(subset=["kepid"]).copy()
    catalog["kepid"] = catalog["kepid"].astype(int)
    catalog = catalog.drop_duplicates(subset="kepid", keep="first")

    parts = []
    for disposition in ("CONFIRMED", "FALSE POSITIVE"):
        subset = catalog[catalog["koi_disposition"] == disposition]
        shuffled = subset.sample(frac=1.0, random_state=seed)
        if len(shuffled) < n_per_class:
            raise ValueError(
                f"Solo hay {len(shuffled)} objetivos {disposition} disponibles, "
                f"se pidieron {n_per_class}."
            )
        parts.append(shuffled.head(n_per_class))

    sample = pd.concat(parts)
    return sample.sample(frac=1.0, random_state=seed).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 3. Descarga de curvas de luz (con caché local)
# --------------------------------------------------------------------------- #
def download_lc_for_kepid(kepid: int, max_quarters: int = 4) -> pd.DataFrame:
    """Descarga la curva de luz de Kepler (cadencia larga) de un ``kepid``.

    Limita la descarga a los primeros ``max_quarters`` quarters para que el
    procesamiento no tarde una eternidad. Cachea el resultado como CSV en
    ``data/lc_cache/`` para no volver a descargar.
    """
    LC_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = LC_CACHE_DIR / f"KIC_{kepid}_lc.csv"
    if cache_file.exists():
        return pd.read_csv(cache_file)

    search = lk.search_lightcurve(f"KIC {kepid}", mission="Kepler", cadence="long")
    if len(search) == 0:
        raise ValueError(f"Sin curvas de luz de Kepler para KIC {kepid}")

    # Ordenar por número de quarter ("Kepler Quarter 02" -> 2) y quedarnos
    # con los primeros max_quarters.
    quarters = [int(str(m).split()[-1]) for m in search.table["mission"]]
    order = np.argsort(quarters)[:max_quarters]

    lcc = None
    for attempt in (1, 2):
        try:
            lcc = search[order].download_all()
            break
        except Exception as exc:
            # Si una descarga previa quedó interrumpida, lightkurve guarda un
            # FITS corrupto en su caché global. Lo borramos y reintentamos.
            msg = str(exc)
            match = re.search(r"(/\S+\.fits)", msg)
            if "corrupt" in msg and match and attempt == 1:
                Path(match.group(1)).unlink(missing_ok=True)
                continue
            raise

    if lcc is None or len(lcc) == 0:
        raise ValueError(f"La descarga falló para KIC {kepid}")

    lc = lcc.stitch()
    df = pd.DataFrame({
        "time": np.asarray(lc.time.value, dtype=float),
        "flux": np.asarray(lc.flux.value, dtype=float),
        "flux_err": np.asarray(lc.flux_err.value, dtype=float),
    }).dropna().reset_index(drop=True)

    if len(df) < 100:
        raise ValueError(f"Curva demasiado corta para KIC {kepid} ({len(df)} puntos)")

    df.to_csv(cache_file, index=False)
    return df


# --------------------------------------------------------------------------- #
# 4. Extracción de features desde la curva de luz
# --------------------------------------------------------------------------- #
def _transit_shape_features(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    duration: float,
    t0: float,
) -> tuple[float, float]:
    """Calcula odd_even_diff y secondary_depth desde la curva normalizada.

    - ``odd_even_diff``: diferencia entre la profundidad media de los
      tránsitos pares e impares, normalizada por su incertidumbre. Una
      diferencia significativa es indicio de binaria eclipsante (eclipses
      primario y secundario de distinta profundidad confundidos con un único
      tránsito al doble de período).
    - ``secondary_depth``: profundidad medida en fase 0.5 (eclipse
      secundario). Los planetas casi no lo tienen; las binarias eclipsantes
      sí.
    """
    half_width = (duration / period) / 2.0  # semiancho del tránsito en unidades de fase

    # --- Profundidad de cada tránsito individual, agrupada por paridad ---
    k_min = int(np.floor((time.min() - t0) / period))
    k_max = int(np.ceil((time.max() - t0) / period))
    depths = []
    parities = []
    for k in range(k_min, k_max + 1):
        transit_time = t0 + k * period
        mask = np.abs(time - transit_time) < duration / 2.0
        if mask.sum() >= 3:
            depths.append(1.0 - np.median(flux[mask]))
            parities.append(k % 2)

    depths = np.asarray(depths)
    parities = np.asarray(parities)

    odd_even_diff = np.nan
    if (parities == 0).sum() >= 2 and (parities == 1).sum() >= 2:
        even = depths[parities == 0]
        odd = depths[parities == 1]
        diff = even.mean() - odd.mean()
        err = np.sqrt(even.std(ddof=1) ** 2 / len(even)
                      + odd.std(ddof=1) ** 2 / len(odd))
        odd_even_diff = float(diff / err) if err > 0 else 0.0

    # --- Profundidad en fase 0.5 (eclipse secundario) ---
    phase = ((time - t0) / period) % 1.0
    phase[phase > 0.5] -= 1.0
    secondary_mask = np.abs(np.abs(phase) - 0.5) < half_width
    secondary_depth = (
        float(1.0 - np.median(flux[secondary_mask]))
        if secondary_mask.sum() >= 3
        else np.nan
    )

    return odd_even_diff, secondary_depth


def extract_features(
    lc: pd.DataFrame,
    period_range: tuple[float, float] = (0.5, 20.0),
    window_length: int = DETREND_WINDOW,
) -> dict:
    """Aplica el pipeline completo a una curva de luz y extrae features.

    Todas las features se derivan de la curva de luz (no del catálogo), para
    evitar data leakage. ``snr`` y ``power`` de BLS tienen colas muy pesadas,
    por lo que se devuelven como ``log10(1 + x)``.
    """
    lc_clean = remove_outliers(lc, sigma=5)
    lc_detrended = detrend(lc_clean, window_length=window_length)
    lc_norm = normalize_flux(lc_detrended)

    result = find_transits(
        lc_norm,
        period_range=period_range,
        output_path=None,
        n_periods=1500,
        n_durations=20,
        bls_method="fast",
    )

    period = result["period"]
    depth = result["depth"]
    duration = result["duration"]
    t0 = result["t0"]

    time = lc_norm["time"].values
    flux = lc_norm["flux"].values

    n_transits = int(np.floor((time.max() - time.min()) / period))

    odd_even_diff, secondary_depth = _transit_shape_features(
        time, flux, period, duration, t0
    )

    return {
        "period": float(period),
        "depth": float(depth),
        "duration": float(duration),
        "log_snr": float(np.log10(1.0 + max(result["snr"], 0.0))),
        "log_power": float(np.log10(1.0 + max(result["power"], 0.0))),
        "duration_period_ratio": float(duration / period),
        "n_transits": n_transits,
        "odd_even_diff": float(odd_even_diff),
        "secondary_depth": float(secondary_depth),
        "depth_duration_ratio": float(depth / duration) if duration > 0 else np.nan,
        "secondary_ratio": float(secondary_depth / depth) if depth > 0 else np.nan,
        "flux_std": float(np.std(flux)),
    }


# --------------------------------------------------------------------------- #
# 5. Reprocesamiento completo desde la caché (--recompute)
# --------------------------------------------------------------------------- #
def _process_cached(kepid: int, label: str) -> dict:
    lc = pd.read_csv(LC_CACHE_DIR / f"KIC_{kepid}_lc.csv")
    features = extract_features(lc)
    features["kepid"] = kepid
    features["label"] = label
    return features


def recompute(n_jobs: int = -1) -> None:
    """Reprocesa todas las curvas de ``data/lc_cache/`` y escribe el dataset v2.

    El dataset anterior (``features_dataset.csv``) se renombra a
    ``features_dataset_v1.csv`` (si v1 todavía no existe) para poder comparar.
    No descarga nada. El orden de las filas sigue al de v1 cuando existe.
    """
    if LEGACY_DATASET_PATH.exists():
        if DATASET_V1_PATH.exists():
            raise FileExistsError(
                f"{DATASET_V1_PATH.name} ya existe; revisá {LEGACY_DATASET_PATH.name} a mano."
            )
        LEGACY_DATASET_PATH.rename(DATASET_V1_PATH)
        print(f"Renombrado {LEGACY_DATASET_PATH.name} -> {DATASET_V1_PATH.name}")

    catalog = get_koi_catalog().dropna(subset=["kepid"]).copy()
    catalog["kepid"] = catalog["kepid"].astype(int)
    # Misma regla que build_sample: una fila por kepid, la primera del catálogo.
    catalog = catalog.drop_duplicates(subset="kepid", keep="first")
    labels = dict(zip(catalog["kepid"], catalog["koi_disposition"]))
    cached = sorted(int(p.stem.split("_")[1]) for p in LC_CACHE_DIR.glob("KIC_*_lc.csv"))
    if DATASET_V1_PATH.exists():
        order = pd.read_csv(DATASET_V1_PATH)["kepid"].astype(int).tolist()
        cached = [k for k in order if k in set(cached)]
    targets = [(k, labels[k]) for k in cached
               if labels.get(k) in ("CONFIRMED", "FALSE POSITIVE")]
    print(f"Reprocesando {len(targets)} curvas cacheadas (n_jobs={n_jobs})...", flush=True)

    def safe(kepid: int, label: str):
        try:
            return _process_cached(kepid, label), None
        except Exception as exc:  # noqa: BLE001
            return None, (kepid, label, f"{type(exc).__name__}: {exc}")

    results = Parallel(n_jobs=n_jobs, verbose=5)(delayed(safe)(k, l) for k, l in targets)
    rows = [r for r, _ in results if r is not None]
    failures = [e for _, e in results if e is not None]

    pd.DataFrame(rows)[FEATURE_COLUMNS].to_csv(DATASET_PATH, index=False)
    with open(FAILED_LOG, "a") as fh:
        for kepid, label, msg in failures:
            fh.write(f"{kepid}\t{label}\t[recompute] {msg}\n")
    print(f"Listo. OK: {len(rows)}, fallidos: {len(failures)}. Dataset: {DATASET_PATH}")


# --------------------------------------------------------------------------- #
# 6. Loop principal con guardado incremental
# --------------------------------------------------------------------------- #
def main(n_per_class: int, seed: int = 42) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    processed: set[int] = set()
    if DATASET_PATH.exists():
        existing = pd.read_csv(DATASET_PATH)
        processed = set(existing["kepid"].astype(int))
        print(f"Retomando: {len(processed)} objetivos ya procesados.")

    sample = build_sample(n_per_class, seed=seed)
    print(f"Muestra: {n_per_class} CONFIRMED + {n_per_class} FALSE POSITIVE.\n")

    n_ok = 0
    n_fail = 0
    for _, row in sample.iterrows():
        kepid = int(row["kepid"])
        label = row["koi_disposition"]

        if kepid in processed:
            print(f"[skip] KIC {kepid} ya estaba procesado")
            continue

        try:
            signal.signal(signal.SIGALRM, _alarm_handler)
            signal.alarm(TARGET_TIMEOUT)

            lc = download_lc_for_kepid(kepid)
            features = extract_features(lc)
            features["kepid"] = kepid
            features["label"] = label

            df_row = pd.DataFrame([features])[FEATURE_COLUMNS]
            write_header = not DATASET_PATH.exists()
            df_row.to_csv(DATASET_PATH, mode="a", header=write_header, index=False)
            processed.add(kepid)
            n_ok += 1
            print(
                f"[ok]   KIC {kepid} ({label}): "
                f"P={features['period']:.4f} d, depth={features['depth']:.2e}, "
                f"log10(1+SNR)={features['log_snr']:.2f}"
            )
        except Exception as exc:  # noqa: BLE001 - queremos máxima robustez
            n_fail += 1
            with open(FAILED_LOG, "a") as fh:
                fh.write(f"{kepid}\t{label}\t{type(exc).__name__}: {exc}\n")
            print(f"[fail] KIC {kepid} ({label}): {type(exc).__name__}: {exc}")
        finally:
            signal.alarm(0)

    print(f"\nTerminado. OK: {n_ok}, fallidos: {n_fail}.")
    print(f"Dataset: {DATASET_PATH}")
    if n_fail:
        print(f"Log de fallos: {FAILED_LOG}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Construye el dataset de features a partir de KOI."
    )
    parser.add_argument(
        "--n-per-class",
        type=int,
        default=5,
        help="Cantidad de objetivos por clase (CONFIRMED y FALSE POSITIVE).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria.")
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Reprocesa todo data/lc_cache/ en paralelo y escribe features_dataset_v2.csv "
        "(renombra el CSV previo a features_dataset_v1.csv).",
    )
    args = parser.parse_args()

    if args.recompute:
        recompute()
        sys.exit(0)

    main(n_per_class=args.n_per_class, seed=args.seed)
