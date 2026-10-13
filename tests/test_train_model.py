"""Pruebas del clasificador entrenado (data/model.pkl)."""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from train_model import DATASET_PATH, FEATURE_COLUMNS, MODEL_PATH


@pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="data/model.pkl no existe; correr `python src/train_model.py`",
)
def test_model_loads_and_predict_proba_in_unit_interval():
    """El pipeline guardado se carga y devuelve probabilidades en [0, 1]."""
    model = joblib.load(MODEL_PATH)
    row = pd.read_csv(DATASET_PATH).iloc[[0]][FEATURE_COLUMNS]

    proba = model.predict_proba(row)

    assert proba.shape == (1, 2)
    assert np.all((proba >= 0.0) & (proba <= 1.0))
    assert proba.sum() == pytest.approx(1.0)
