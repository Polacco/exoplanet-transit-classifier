"""Funciones de preprocesamiento de curvas de luz."""

from __future__ import annotations

import lightkurve as lk
import numpy as np
import pandas as pd
from astropy.stats import sigma_clip


REQUIRED_COLUMNS = {"time", "flux", "flux_err"}


def _validate_lc(lc: pd.DataFrame) -> None:
    """Verifica que el DataFrame tenga las columnas mínimas necesarias."""
    if not isinstance(lc, pd.DataFrame):
        raise TypeError("lc debe ser un pandas.DataFrame")
    missing = REQUIRED_COLUMNS - set(lc.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")


def remove_outliers(lc: pd.DataFrame, sigma: float = 5) -> pd.DataFrame:
    """Elimina outliers de la curva de luz mediante sigma-clipping.

    Parameters
    ----------
    lc : pd.DataFrame
        Curva de luz con columnas ``time``, ``flux`` y ``flux_err``.
    sigma : float, optional
        Umbral en desviaciones estándar para considerar un punto outlier.
        Por defecto 5.

    Returns
    -------
    pd.DataFrame
        Curva de luz sin los puntos rechazados por sigma-clipping.
    """
    _validate_lc(lc)
    clipped = sigma_clip(lc["flux"].values, sigma=sigma, masked=True)
    mask = ~clipped.mask
    return lc.loc[mask].reset_index(drop=True)


def detrend(lc: pd.DataFrame, window_length: int) -> pd.DataFrame:
    """Remueve la tendencia de largo plazo usando ``lightkurve.LightCurve.flatten``.

    ``flatten`` aplica internamente un filtro Savitzky-Golay para estimar
    la tendencia y dividir la curva de luz por ella.

    Parameters
    ----------
    lc : pd.DataFrame
        Curva de luz con columnas ``time``, ``flux`` y ``flux_err``.
    window_length : int
        Longitud de la ventana del filtro Savitzky-Golay. Debe ser impar y
        menor que la cantidad de puntos; si no lo es, se ajusta
        automáticamente.

    Returns
    -------
    pd.DataFrame
        Curva de luz detrended con columnas ``time``, ``flux`` y ``flux_err``.
    """
    _validate_lc(lc)

    n = len(lc)
    if n < 3:
        raise ValueError("La curva de luz debe tener al menos 3 puntos para detrending.")

    window_length = int(window_length)
    if window_length % 2 == 0:
        window_length += 1
    if window_length > n:
        window_length = n if n % 2 == 1 else n - 1
    if window_length < 3:
        window_length = 3

    lk_lc = lk.LightCurve(
        time=lc["time"].values,
        flux=lc["flux"].values,
        flux_err=lc["flux_err"].values,
    )
    flat = lk_lc.flatten(window_length=window_length)

    return pd.DataFrame({
        "time": np.asarray(flat.time.value, dtype=float),
        "flux": np.asarray(flat.flux.value, dtype=float),
        "flux_err": np.asarray(flat.flux_err.value, dtype=float),
    })


def normalize_flux(lc: pd.DataFrame) -> pd.DataFrame:
    """Normaliza el flujo para que oscile alrededor de 1.0.

    Divide ``flux`` y ``flux_err`` por la mediana del flujo original.

    Parameters
    ----------
    lc : pd.DataFrame
        Curva de luz con columnas ``time``, ``flux`` y ``flux_err``.

    Returns
    -------
    pd.DataFrame
        Curva de luz normalizada.
    """
    _validate_lc(lc)
    median = lc["flux"].median()
    if median == 0:
        raise ValueError("La mediana del flujo es cero; no se puede normalizar.")

    normalized = lc.copy()
    normalized["flux"] = normalized["flux"] / median
    normalized["flux_err"] = normalized["flux_err"] / median
    return normalized
