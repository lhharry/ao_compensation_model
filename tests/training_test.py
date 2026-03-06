"""Tests for training module functions (no actual training run)."""

import numpy as np
import pandas as pd

from ao_compensation_model.definitions import WINDOW_SIZE
from ao_compensation_model.training import (
    build_gru_model,
    preprocess_one_csv,
)


def test_preprocess_one_csv(tmp_path):
    """preprocess_one_csv should return features and targets arrays."""
    n = 300
    t = np.arange(n)
    df = pd.DataFrame(
        {
            "Hip_x": 10 * np.sin(2 * np.pi * t / 100),
            "Hip_x_ao": np.linspace(0, 6 * np.pi, n),
            "Hip_x_vel": np.random.default_rng(0).standard_normal(n),
            "Hip_x_omega": 3.0 * np.ones(n),
            "Hip_x_domega": 0.1 * np.ones(n),
            "target_sin": np.sin(np.linspace(0, np.pi, n)),
            "target_cos": np.cos(np.linspace(0, np.pi, n)),
        }
    )
    csv_path = tmp_path / "train.csv"
    df.to_csv(csv_path, index=False, sep=";")

    features, targets = preprocess_one_csv(csv_path)

    assert features.shape == (n, 2)
    assert targets.shape == (n, 3)


def test_build_gru_model_output_shape():
    """GRU model should have the correct input/output shapes."""
    n_features = 2
    model = build_gru_model(WINDOW_SIZE, n_features)
    assert model.input_shape == (None, WINDOW_SIZE, n_features)
    assert model.output_shape == {"phase": (None, 2), "omega": (None, 1)}


def test_build_gru_model_with_batch_size():
    """GRU model with fixed batch_size=1 should compile."""
    model = build_gru_model(WINDOW_SIZE, 2, batch_size=1)
    assert model.output_shape == {"phase": (1, 2), "omega": (1, 1)}
