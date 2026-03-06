"""Train a GRU model for adaptive-oscillator phase compensation.

Loads labelled CSV files, fits a StandardScaler, trains a GRU model with
sample weighting, and exports the best model as an optimized TFLite file.
"""

import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.layers import GRU, Dense, Input, UnitNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from ao_compensation_model.definitions import (
    BATCH_SIZE,
    DROPOUT_RATE,
    GRU_UNITS,
    LEARNING_RATE,
    MAX_EPOCHS,
    MODEL_DIR,
    TRAINING_DATA_DIR,
    WINDOW_SIZE,
)
from ao_compensation_model.utils import create_sliding_windows


def compute_sample_weights(omega_values: np.ndarray) -> np.ndarray:
    """Assign per-sample weights based on the AO frequency (omega).

    Higher weights are given to transitional phases where the AO is
    re-locking, because those are the most informative for the GRU.

    :param omega_values: Array of omega values aligned to each window.
    :return: Float32 weight array.
    """
    weights = np.where(
        omega_values > 3.0,
        0.5,  # Stable walking — AO already locked
        np.where(
            omega_values > 1.5,
            1.0,  # Re-locking phase — most critical
            np.where(
                omega_values > 0.5,
                0.8,  # Transition just starting
                0.1,  # Fully stopped — target near zero
            ),
        ),
    )
    return weights.astype(np.float32)


def preprocess_one_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a single labelled CSV and return features, targets, and omega.

    :param csv_path: Path to a training CSV with target_cos / target_sin columns.
    :return: (features, targets, omega) arrays.
    """
    df = pd.read_csv(csv_path, sep=";")

    raw_angle = np.asarray(df["Hip_x"].values)
    ao_gait_phase = np.asarray(df["Hip_x_ao"].values)
    angular_velocity = np.asarray(df["Hip_x_vel"].values)
    omega = np.asarray(df["Hip_x_omega"].values)
    domega = np.clip(np.asarray(df["Hip_x_domega"].values), -20.0, 20.0)

    ao_phase_sin = np.sin(ao_gait_phase)
    ao_phase_cos = np.cos(ao_gait_phase)

    target_sin = np.asarray(df["target_sin"].values)
    target_cos = np.asarray(df["target_cos"].values)
    targets = np.column_stack([target_sin, target_cos])

    features = np.column_stack(
        [raw_angle, angular_velocity, omega, domega, ao_phase_sin, ao_phase_cos]
    )
    return features, targets, omega


def build_gru_model(
    window_size: int,
    n_features: int,
    batch_size: int | None = None,
) -> Model:
    """Construct the GRU model architecture.

    :param window_size: Number of time steps per input window.
    :param n_features: Number of input features.
    :param batch_size: Fixed batch size (set to 1 for inference / TFLite export).
    :return: Keras Model (uncompiled).
    """
    inp = Input(shape=(window_size, n_features), batch_size=batch_size)
    x = GRU(
        units=GRU_UNITS,
        recurrent_activation="sigmoid",
        return_sequences=False,
        dropout=DROPOUT_RATE,
        recurrent_dropout=DROPOUT_RATE,
    )(inp)
    linear_out = Dense(units=2, activation="linear")(x)
    normalized_out = UnitNormalization(axis=1, name="l2_norm")(linear_out)
    return Model(inputs=inp, outputs=normalized_out)


def train():
    """Run the full training pipeline: load data, train, and export TFLite."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "gru_model_residual.h5"
    scaler_path = MODEL_DIR / "scaler.pkl"
    tflite_path = MODEL_DIR / "gru_model_optimized.tflite"

    # --- Load all training CSVs ---
    csv_files = sorted(TRAINING_DATA_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {TRAINING_DATA_DIR}")

    file_data = []
    for csv_file in csv_files:
        features, targets, omega = preprocess_one_csv(csv_file)
        file_data.append((features, targets, omega))

    # --- Fit scaler on the union of all files ---
    all_features = np.vstack([f for f, _, _ in file_data])
    scaler = StandardScaler()
    scaler.fit(all_features)
    joblib.dump(scaler, scaler_path)

    # --- Create sliding windows per file, then merge ---
    x_list, y_list, w_list = [], [], []
    for features, targets, omega in file_data:
        features_scaled = np.asarray(scaler.transform(features))
        x_file, y_file = create_sliding_windows(features_scaled, targets, WINDOW_SIZE)
        if len(x_file) == 0:
            continue
        omega_aligned = omega[WINDOW_SIZE : WINDOW_SIZE + len(x_file)]
        w_file = compute_sample_weights(omega_aligned)
        x_list.append(x_file)
        y_list.append(y_file)
        w_list.append(w_file)

    if not x_list:
        raise ValueError(
            f"No valid windows created. Check CSV lengths vs WINDOW_SIZE={WINDOW_SIZE}."
        )

    x = np.concatenate(x_list)
    y = np.concatenate(y_list)
    w = np.concatenate(w_list)

    # Shuffle so the validation split contains a mix of all files
    idx = np.random.permutation(len(x))
    x, y, w = x[idx], y[idx], w[idx]

    # --- Train ---
    model = build_gru_model(WINDOW_SIZE, x.shape[2])
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss="mse")

    callbacks = [
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
        ),
        EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True, verbose=1
        ),
        ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    model.fit(
        x,
        y,
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        sample_weight=w,
        callbacks=callbacks,
    )

    # --- Export to TFLite ---
    best_model = tf.keras.models.load_model(str(model_path), compile=False)
    inference_model = build_gru_model(WINDOW_SIZE, x.shape[2], batch_size=1)
    inference_model.set_weights(best_model.get_weights())

    converter = tf.lite.TFLiteConverter.from_keras_model(inference_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(tflite_path, "wb") as f:
        f.write(tflite_model)


if __name__ == "__main__":
    train()
