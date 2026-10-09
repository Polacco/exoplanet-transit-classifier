"""Pruebas end-to-end de detección de tránsitos con BLS."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data_loader import download_light_curve
from preprocessing import detrend, normalize_flux, remove_outliers
from transit_detection import find_transits


@pytest.fixture(scope="module")
def kepler10_lc():
    """Curva de luz de Kepler-10. Requiere internet en la primera ejecución."""
    return download_light_curve("Kepler-10", mission="Kepler")


# Este test requiere conexión a internet la primera vez que se ejecuta
# (luego usa el cache local en data/) y corre el pipeline completo.
def test_pipeline_detects_kepler10b_period(kepler10_lc):
    """Pipeline completo: descarga → limpieza → detrend → BLS.

    Kepler-10b tiene un período orbital publicado de ~0.84 días.
    Verificamos que BLS lo detecta dentro del 5%.
    """
    # Trabajamos con los primeros ~10 días para mantener el test ágil.
    t0 = kepler10_lc["time"].min()
    lc = kepler10_lc[kepler10_lc["time"] < t0 + 10].copy()

    lc_clean = remove_outliers(lc, sigma=5)

    # La duración de un tránsito típico de Kepler-10b es de ~1.8 h (~0.075 d).
    # Elegimos una ventana mucho mayor (1001 puntos, ~0.7 d) para no borrar la señal.
    lc_detrended = detrend(lc_clean, window_length=1001)

    lc_norm = normalize_flux(lc_detrended)

    output = Path("docs") / "kepler10_folded.png"
    result = find_transits(
        lc_norm,
        period_range=(0.5, 1.5),
        output_path=output,
    )

    published_period = 0.837495  # días (Kepler-10b)
    detected_period = result["period"]

    assert abs(detected_period - published_period) / published_period < 0.05
    assert result["snr"] > 5
    assert output.exists()
