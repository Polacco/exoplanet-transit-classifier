"""Pruebas para el módulo de carga de curvas de luz."""

import sys
from pathlib import Path

import pytest

# Aseguramos que `src/` esté disponible en sys.path al ejecutar pytest.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data_loader import download_light_curve


# Este test requiere conexión a internet para descargar datos de MAST.
# La segunda ejecución debería ser más rápida gracias al cache local en data/.
def test_download_light_curve_kepler10():
    """Descarga Kepler-10 y verifica que el DataFrame tiene datos y columnas esperadas."""
    df = download_light_curve("Kepler-10", mission="Kepler")

    assert not df.empty, "El DataFrame de la curva de luz está vacío"
    assert "time" in df.columns
    assert "flux" in df.columns
    assert "flux_err" in df.columns

    # Sanity check de tipos y contenido mínimo.
    assert df["time"].notna().any()
    assert df["flux"].notna().any()
