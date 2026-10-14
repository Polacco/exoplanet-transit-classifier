"""Pruebas del clasificador v3 entrenado (data/model.pkl)."""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from koi_features import DATASET_V3_PATH, FEATURE_COLUMNS_V3
from train_model import FEATURES_PATH, MODEL_PATH, heldout_split, load_dataset_v3

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="data/model.pkl no existe; correr `python src/train_model.py`",
)


def test_saved_feature_list_is_v3():
    """La lista guardada junto al modelo son las 9 features v3."""
    assert json.loads(FEATURES_PATH.read_text()) == FEATURE_COLUMNS_V3


def test_model_loads_and_predict_proba_in_unit_interval():
    """El pipeline guardado se carga y devuelve probabilidades en [0, 1]."""
    model = joblib.load(MODEL_PATH)
    rows = pd.read_csv(DATASET_V3_PATH)[FEATURE_COLUMNS_V3].head(20)

    proba = model.predict_proba(rows)

    assert proba.shape == (20, 2)
    assert np.all((proba >= 0.0) & (proba <= 1.0))
    assert proba.sum(axis=1) == pytest.approx(1.0)


def test_model_accepts_missing_values():
    """El imputer dentro del Pipeline permite NaN en las features."""
    model = joblib.load(MODEL_PATH)
    row = pd.DataFrame([{c: np.nan for c in FEATURE_COLUMNS_V3}])

    proba = model.predict_proba(row)

    assert np.all(np.isfinite(proba))


def test_heldout_split_has_no_star_in_both_sets():
    """Ninguna estrella aparece a la vez en train y en test."""
    df = load_dataset_v3()
    tr, te = heldout_split(df)

    assert not set(df["kepid"].iloc[tr]) & set(df["kepid"].iloc[te])
    assert len(tr) + len(te) == len(df)
