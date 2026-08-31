"""Carga y cacheo de curvas de luz desde MAST usando lightkurve."""

from __future__ import annotations

import re
from pathlib import Path

import lightkurve as lk
import numpy as np
import pandas as pd

# Ubicación por defecto del cache local, relativa a la raíz del proyecto.
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "data"


def _sanitize_filename(name: str) -> str:
    """Convierte un string en un nombre de archivo seguro."""
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", str(name))


def download_light_curve(
    target_name: str,
    mission: str = "TESS",
    cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Busca, descarga (o carga desde caché) y devuelve una curva de luz.

    Parameters
    ----------
    target_name : str
        Nombre del objetivo (p. ej. ``"TOI-700"``, ``"Kepler-10"``,
        ``"TIC 123456789"``, ``"KIC 1234567"``) o identificador de catálogo.
    mission : str, optional
        Misión de la que obtener datos. Por defecto ``"TESS"``.
        Otros valores comunes: ``"Kepler"``, ``"K2"``.
    cache_dir : str | Path | None, optional
        Directorio donde cachear los archivos FITS descargados.
        Si es ``None`` se usa ``<raíz_del_proyecto>/data``.

    Returns
    -------
    pd.DataFrame
        DataFrame con las columnas ``time``, ``flux`` y ``flux_err``.

    Raises
    ------
    ValueError
        Si no se encuentran curvas de luz para el objetivo/misión dados.
    RuntimeError
        Si la descarga desde MAST falla.
    """
    cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    search_result = lk.search_lightcurve(target_name, mission=mission)
    if len(search_result) == 0:
        raise ValueError(
            f"No se encontraron curvas de luz para '{target_name}' "
            f"en la misión '{mission}'."
        )

    # Seleccionamos el primer resultado (generalmente el de mayor calidad o
    # el primer sector/cuarto disponible).
    first_result = search_result[0]

    # Construimos un nombre de caché único a partir del nombre del producto FITS
    # (p. ej. kplr011904151-2009231120729_slc.fits) u otro identificador único.
    table = first_result.table
    product_id = None
    for key in ("productFilename", "obs_id", "obsid"):
        try:
            product_id = table[key][0]
            break
        except KeyError:
            continue

    if product_id is None:
        product_id = "unknown"
    safe_product = _sanitize_filename(Path(str(product_id)).stem)

    safe_target = _sanitize_filename(target_name)
    safe_mission = _sanitize_filename(mission)
    cache_file = cache_dir / f"{safe_target}_{safe_mission}_{safe_product}.fits"

    if cache_file.exists():
        lc = lk.read(cache_file)
    else:
        lc = first_result.download()
        if lc is None:
            raise RuntimeError(
                f"No se pudo descargar la curva de luz para '{target_name}' "
                f"({mission})."
            )
        lc.to_fits(cache_file, overwrite=True)

    # Normalizamos la salida a un DataFrame con las columnas requeridas.
    # Forzamos tipos little-endian nativos porque pandas no acepta buffers
    # big-endian en arquitecturas little-endian.
    time = np.asarray(lc.time.value, dtype=float)
    flux = np.asarray(lc.flux.value, dtype=float)

    if hasattr(lc, "flux_err") and lc.flux_err is not None:
        flux_err = np.asarray(lc.flux_err.value, dtype=float)
    else:
        flux_err = np.full_like(flux, np.nan, dtype=float)

    df = pd.DataFrame({
        "time": time,
        "flux": flux,
        "flux_err": flux_err,
    })

    # Descartamos filas sin las magnitudes esenciales.
    return df.dropna(subset=["time", "flux"]).reset_index(drop=True)
