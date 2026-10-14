"""Pruebas del dataset v3 (una fila por KOI): fuga de información y features de forma.

Usan curvas sintéticas, no requieren internet.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from koi_features import (
    CADENCE_DAYS,
    DATASET_V3_PATH,
    FEATURE_COLUMNS_V3,
    FORBIDDEN_FEATURE_PREFIX,
    koi_features,
    time_from_midtransit,
)
from preprocessing import detrend

P, T0, D = 3.0, 1.0, 0.15  # días
NOISE = 1e-4


def synthetic(depth=1e-3, shape="box", odd_even=0.0, secondary=0.0, seed=0):
    """Curva de 90 días en cadencia larga con tránsitos de forma controlada."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, 90, CADENCE_DAYS)
    flux = 1 + rng.normal(0, NOISE, t.size)
    dt = time_from_midtransit(t, P, T0)
    x = np.abs(dt) / (D / 2)  # 0 en el centro, 1 en los bordes
    inside = x < 1
    profile = np.where(inside, 1.0 if shape == "box" else 1 - x, 0.0)
    epoch = np.round((t - T0) / P).astype(int)
    d = np.where(epoch % 2 == 1, depth * (1 + odd_even), depth)
    flux -= d * profile
    dt2 = time_from_midtransit(t, P, T0 + P / 2)
    flux -= secondary * (np.abs(dt2) < D / 2)
    return t, flux


def test_no_koi_columns_are_features():
    """Ninguna columna del catálogo (koi_*) se usa como feature."""
    assert not [c for c in FEATURE_COLUMNS_V3 if c.startswith(FORBIDDEN_FEATURE_PREFIX)]
    t, f = synthetic()
    assert not [k for k in koi_features(t, f, P, T0, D) if k.startswith("koi_")]


@pytest.mark.skipif(not DATASET_V3_PATH.exists(), reason="dataset v3 no construido")
def test_dataset_v3_has_no_catalog_columns():
    cols = pd.read_csv(DATASET_V3_PATH, nrows=1).columns
    assert not [c for c in cols if c.startswith(FORBIDDEN_FEATURE_PREFIX)]
    assert set(FEATURE_COLUMNS_V3) <= set(cols)


def test_planet_like_box_transit():
    f = koi_features(*synthetic(shape="box"), P, T0, D)
    assert f["log_transit_snr"] > 1.5                       # SNR > ~30
    assert abs(10 ** f["log_depth_ppm"] - 1 - 1000) < 100   # ~1000 ppm
    assert f["n_transits_observed"] == 30
    assert f["shape_uv_ratio"] == pytest.approx(1.0, abs=0.1)
    assert f["log_odd_even_sigma"] < np.log10(1 + 4)
    assert abs(f["log_secondary_snr"]) < np.log10(1 + 4)


def test_v_shape_is_detected():
    f = koi_features(*synthetic(shape="v"), P, T0, D)
    assert f["shape_uv_ratio"] > 1.4


def test_eclipsing_binary_signatures():
    f = koi_features(*synthetic(odd_even=0.5, secondary=3e-4), P, T0, D)
    assert f["log_odd_even_sigma"] > np.log10(1 + 5)
    assert f["log_secondary_snr"] > np.log10(1 + 5)


def test_masked_detrend_preserves_long_transit():
    """Con máscara, el filtro no se come un tránsito largo (D = 0.4 d)."""
    rng = np.random.default_rng(1)
    t = np.arange(0, 60, CADENCE_DAYS)
    dt = time_from_midtransit(t, 7.0, 3.0)
    inside = np.abs(dt) < 0.2
    flux = 1 + 2e-3 * np.sin(2 * np.pi * t / 15) - 1e-3 * inside + rng.normal(0, 1e-4, t.size)
    lc = pd.DataFrame({"time": t, "flux": flux, "flux_err": np.full(t.size, 1e-4)})

    def depth(out):
        f = out["flux"].values
        return np.median(f[~inside]) - np.median(f[inside])

    masked = depth(detrend(lc, 101, mask=np.abs(dt) < 0.4))
    unmasked = depth(detrend(lc, 101))
    assert masked == pytest.approx(1e-3, rel=0.1)
    assert unmasked < masked
