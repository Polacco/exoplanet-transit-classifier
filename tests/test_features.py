"""Pruebas para el módulo de extracción de features."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data_loader import download_light_curve
from features import extract_features

EXPECTED_FEATURE_KEYS = [
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
]


@pytest.fixture(scope="module")
def kepler10_lc():
    """Curva de luz de Kepler-10. Requiere internet en la primera ejecución."""
    return download_light_curve("Kepler-10", mission="Kepler")


# Este test requiere conexión a internet la primera vez que se ejecuta
# (luego usa el cache local en data/).
def test_extract_features_returns_expected_columns(kepler10_lc):
    """extract_features devuelve todas las features esperadas, sin NaN en las claves."""
    # Recortamos a ~10 días para que el test sea ágil.
    t0 = kepler10_lc["time"].min()
    lc = kepler10_lc[kepler10_lc["time"] < t0 + 10].copy()

    # El archivo de Kepler-10 es de cadencia corta (1 min): el default del
    # pipeline (101 puntos, calibrado para cadencia larga de 30 min) sería una
    # ventana de ~1.7 h y borraría el tránsito. 1001 puntos ~ 0.7 d.
    features = extract_features(lc, period_range=(0.5, 1.5), window_length=1001)

    for key in EXPECTED_FEATURE_KEYS:
        assert key in features, f"Falta la feature '{key}'"

    assert not np.isnan(features["period"])
    assert not np.isnan(features["depth"])
    assert not np.isnan(features["log_snr"])
    assert features["log_snr"] >= 0 and features["log_power"] >= 0
    assert features["duration_period_ratio"] == pytest.approx(
        features["duration"] / features["period"]
    )

    # El período de Kepler-10b (~0.84 d) debería ser detectado.
    assert abs(features["period"] - 0.837495) / 0.837495 < 0.05
