"""Pruebas para el módulo de preprocesamiento de curvas de luz."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data_loader import download_light_curve
from preprocessing import detrend, normalize_flux, remove_outliers


@pytest.fixture(scope="module")
def real_lc():
    """Curva de luz real de Kepler-10. Requiere internet la primera vez."""
    return download_light_curve("Kepler-10", mission="Kepler")


def test_normalize_flux_median_is_one(real_lc):
    """Después de normalizar, la mediana del flujo debe ser ~1.0."""
    lc_norm = normalize_flux(real_lc)
    assert abs(lc_norm["flux"].median() - 1.0) < 1e-6


def test_remove_outliers_reduces_points(real_lc):
    """Inyecta outliers sintéticos en una curva real y verifica que se eliminan."""
    n_outliers = 10
    outlier_lc = real_lc.copy()

    # Creamos filas outlier con un flujo muy por encima de la distribución.
    outlier_rows = outlier_lc.head(n_outliers).copy()
    outlier_rows["flux"] = outlier_rows["flux"] + 10 * real_lc["flux"].std()
    outlier_lc = pd.concat([outlier_lc, outlier_rows], ignore_index=True)

    lc_clean = remove_outliers(outlier_lc, sigma=5)

    assert len(lc_clean) < len(outlier_lc)

    # Ninguno de los outliers inyectados debe permanecer.
    high_threshold = real_lc["flux"].median() + 5 * real_lc["flux"].std()
    assert (lc_clean["flux"] > high_threshold).sum() == 0


def test_detrend_preserves_shape_and_columns(real_lc):
    """El detrending debe conservar las columnas y la cantidad de puntos."""
    lc_detrended = detrend(real_lc, window_length=101)
    assert {"time", "flux", "flux_err"}.issubset(lc_detrended.columns)
    assert len(lc_detrended) == len(real_lc)
    assert np.isfinite(lc_detrended["flux"].values).all()
