"""Detección de tránsitos planetarios mediante Box Least Squares (BLS)."""

from __future__ import annotations

from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.timeseries import BoxLeastSquares


REQUIRED_COLUMNS = {"time", "flux", "flux_err"}


def _validate_lc(lc: pd.DataFrame) -> None:
    """Verifica que el DataFrame tenga las columnas mínimas necesarias."""
    if not isinstance(lc, pd.DataFrame):
        raise TypeError("lc debe ser un pandas.DataFrame")
    missing = REQUIRED_COLUMNS - set(lc.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")


def find_transits(
    lc: pd.DataFrame,
    period_range: tuple[float, float] = (0.5, 20.0),
    output_path: str | Path | None = None,
    n_periods: int = 5000,
    n_durations: int = 40,
) -> dict:
    """Detecta tránsitos planetarios con Box Least Squares (BLS).

    La función asume que la curva de luz ya fue preprocesada. Al usar este
    módulo en un pipeline completo (descarga → remove_outliers → detrend →
    normalize_flux → find_transits), es importante que el parámetro
    ``window_length`` de ``detrend`` sea claramente mayor que la duración
    esperada del tránsito (al menos 3 veces la duración del tránsito) para
    evitar que el filtro Savitzky-Golay elimine la señal que se quiere
    detectar.

    Parameters
    ----------
    lc : pd.DataFrame
        Curva de luz preprocesada con columnas ``time``, ``flux`` y
        ``flux_err``. Se recomienda que el flujo esté normalizado alrededor
        de 1.0.
    period_range : tuple[float, float], optional
        Rango de períodos a explorar en días. Por defecto (0.5, 20.0).
    output_path : str | Path | None, optional
        Ruta donde guardar el gráfico de la curva de luz doblada en fase.
        Si es ``None`` se guarda en ``docs/transit_folded.png``.
    n_periods : int, optional
        Cantidad de períodos a muestrear en el rango. Por defecto 5000.
    n_durations : int, optional
        Cantidad de duraciones de tránsito a muestrear. Por defecto 40.

    Returns
    -------
    dict
        Diccionario con las claves ``period``, ``depth``, ``duration``,
        ``t0``, ``snr`` y ``power`` del mejor candidato detectado.
    """
    _validate_lc(lc)

    if output_path is None:
        output_path = Path("docs") / "transit_folded.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Descartamos valores no finales para evitar advertencias de Astropy.
    clean = lc[np.isfinite(lc["time"]) & np.isfinite(lc["flux"])].copy()
    time = clean["time"].values
    flux = clean["flux"].values
    flux_err = clean["flux_err"].values

    model = BoxLeastSquares(time * u.day, flux, dy=flux_err)

    periods = np.linspace(period_range[0], period_range[1], n_periods)
    # Duraciones de tránsito entre ~15 min y ~0.3 días.
    durations = np.linspace(0.01, 0.3, n_durations) * u.day

    results = model.power(periods, durations)
    best_idx = np.argmax(results.power)

    best_period = results.period[best_idx]
    best_duration = results.duration[best_idx]
    best_t0 = results.transit_time[best_idx]
    best_depth = results.depth[best_idx]
    best_depth_err = results.depth_err[best_idx]
    snr = float(best_depth / best_depth_err) if best_depth_err != 0 else np.inf

    # --- Gráfico phase-folded ---
    phase = ((time - best_t0.value) / best_period.value) % 1.0
    phase[phase > 0.5] -= 1.0

    phase_model = np.linspace(-0.5, 0.5, 1000)
    t_model = phase_model * best_period.value + best_t0.value
    model_flux = model.model(t_model * u.day, best_period, best_duration, best_t0)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(phase, flux, s=2, alpha=0.4, c="black", label="Datos")
    ax.plot(phase_model, model_flux, "r-", lw=2, label="Modelo BLS")
    ax.set_xlabel("Fase")
    ax.set_ylabel("Flujo normalizado")
    ax.set_title(
        f"Curva de luz doblada en fase\n"
        f"P = {best_period.value:.4f} d, duración = {best_duration.value:.4f} d, SNR = {snr:.1f}"
    )
    ax.legend()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "period": float(best_period.value),
        "depth": float(best_depth),
        "duration": float(best_duration.value),
        "t0": float(best_t0.value),
        "snr": float(snr),
        "power": float(results.power[best_idx]),
        "periods": np.asarray(results.period),
        "powers": np.asarray(results.power),
    }
