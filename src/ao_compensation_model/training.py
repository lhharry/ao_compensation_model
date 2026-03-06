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
from tensorflow.keras.layers import GRU, Dense, Input, UnitNormalization, Conv1D, BatchNormalization, MaxPooling1D
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
    VAL_SPLIT,
    WINDOW_SIZE,
)
from ao_compensation_model.utils import create_sliding_windows


def preprocess_one_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a single labelled CSV and return features and targets.

    :param csv_path: Path to a training CSV with target_cos / target_sin columns.
    :return: (features, targets) arrays. Targets have columns [sin, cos, omega].
    """
    df = pd.read_csv(csv_path, sep=";")

    raw_angle = np.asarray(df["Hip_x"].values)
    angular_velocity = np.asarray(df["Hip_x_vel"].values)

    omega = np.asarray(df["Hip_x_omega"].values)
    target_sin = np.asarray(df["target_sin"].values)
    target_cos = np.asarray(df["target_cos"].values)
    targets = np.column_stack([target_sin, target_cos, omega])

    features = np.column_stack(
        [raw_angle, angular_velocity]
    )
    return features, targets


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
    phase_out = Dense(units=2, activation="linear")(x)
    phase_normalized = UnitNormalization(axis=1, name="phase")(phase_out)
    omega_out = Dense(units=1, activation="linear", name="omega")(x)
    return Model(inputs=inp, outputs={"phase": phase_normalized, "omega": omega_out})


def train():
    """Run the full training pipeline: load data, train, and export TFLite."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "gru_model.h5"
    scaler_path = MODEL_DIR / "scaler.pkl"
    tflite_path = MODEL_DIR / "gru_model_optimized.tflite"

    # --- Load all training CSVs ---
    csv_files = sorted(TRAINING_DATA_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {TRAINING_DATA_DIR}")

    file_data = []
    for csv_file in csv_files:
        features, targets = preprocess_one_csv(csv_file)
        file_data.append((csv_file.name, features, targets))

    # --- Fit scaler on the union of all files ---
    all_features = np.vstack([f for _, f, _ in file_data])
    scaler = StandardScaler()
    scaler.fit(all_features)
    joblib.dump(scaler, scaler_path)

    # --- File-level train/val split to prevent data leakage ---
    rng = np.random.default_rng(42)
    n_files = len(file_data)
    n_val = max(1, round(n_files * VAL_SPLIT))
    indices = rng.permutation(n_files)
    val_indices = set(indices[:n_val])

    x_train_list, y_train_list = [], []
    x_val_list, y_val_list = [], []

    for i, (name, features, targets) in enumerate(file_data):
        features_scaled = np.asarray(scaler.transform(features))
        x_file, y_file = create_sliding_windows(features_scaled, targets, WINDOW_SIZE)
        if len(x_file) == 0:
            continue

        if i in val_indices:
            x_val_list.append(x_file)
            y_val_list.append(y_file)
        else:
            x_train_list.append(x_file)
            y_train_list.append(y_file)

    if not x_train_list:
        raise ValueError(
            f"No valid training windows. Check CSV lengths vs WINDOW_SIZE={WINDOW_SIZE}."
        )
    if not x_val_list:
        raise ValueError("No validation files. Need at least 2 training CSVs.")

    x_train = np.concatenate(x_train_list)
    y_train = np.concatenate(y_train_list)

    x_val = np.concatenate(x_val_list)
    y_val = np.concatenate(y_val_list)

    # Shuffle training data
    idx = rng.permutation(len(x_train))
    x_train, y_train = x_train[idx], y_train[idx]

    # --- Train ---
    y_train_phase = y_train[:, :2]
    y_train_omega = y_train[:, 2:3]
    y_val_phase = y_val[:, :2]
    y_val_omega = y_val[:, 2:3]

    model = build_gru_model(WINDOW_SIZE, x_train.shape[2])
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss={"phase": "mse", "omega": "mse"},
        loss_weights={"phase": 1.0, "omega": 5.0},
    )

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
        x_train,
        {"phase": y_train_phase, "omega": y_train_omega},
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(
            x_val,
            {"phase": y_val_phase, "omega": y_val_omega},
        ),
        callbacks=callbacks,
    )

    # --- Export to TFLite ---
    best_model = tf.keras.models.load_model(str(model_path), compile=False)
    inference_model = build_gru_model(WINDOW_SIZE, x_train.shape[2], batch_size=1)
    inference_model.set_weights(best_model.get_weights())

    converter = tf.lite.TFLiteConverter.from_keras_model(inference_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(tflite_path, "wb") as f:
        f.write(tflite_model)


if __name__ == "__main__":
    train()
