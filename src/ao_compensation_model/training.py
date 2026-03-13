"""Train a GRU model for adaptive-oscillator phase compensation.

Loads labelled CSV files, fits a StandardScaler, trains a GRU model with
sample weighting, and exports the best model as an optimized TFLite file.
"""

import io
import os
from datetime import date

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from loguru import logger
from sklearn.preprocessing import StandardScaler, RobustScaler
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
    VAL_SPLIT,
    WINDOW_SIZE,
    STRIDE
)
from ao_compensation_model.utils import create_sliding_windows, setup_logger


def preprocess_one_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a single labelled CSV and return features and targets.

    :param csv_path: Path to a training CSV with target_cos / target_sin columns.
    :return: (features, targets) arrays. Targets have columns [sin, cos, omega].
    """
    df = pd.read_csv(csv_path, sep=";")

    raw_angle = np.asarray(df["filter_hip_x"].values)
    angular_velocity = np.asarray(df["Hip_vel"].values)

    omega = np.asarray(df["target_omega"].values)
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
    x = GRU(units=GRU_UNITS, recurrent_activation="sigmoid", return_sequences=True)(inp)
    x = GRU(units=64, recurrent_activation="sigmoid", return_sequences=False)(x)
    phase_out = Dense(units=2, activation="linear")(x)          # (batch, 2)
    phase_normalized = UnitNormalization(axis=-1, name="phase")(phase_out)
    omega_out = Dense(units=1, activation="linear", name="omega")(x)  # (batch, 1)
    return Model(inputs=inp, outputs={"phase": phase_normalized, "omega": omega_out})

class EpochLogger(tf.keras.callbacks.Callback):
    """Log phase and omega validation losses after every epoch."""

    def on_train_begin(self, logs=None):
        logger.info("Training started.")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logger.info(
            "Epoch {:3d} | val_loss: {:.6f}  val_phase_loss: {:.6f}  val_omega_loss: {:.6f}",
            epoch + 1,
            logs.get("val_loss", float("nan")),
            logs.get("val_phase_loss", float("nan")),
            logs.get("val_omega_loss", float("nan")),
        )


def temporal_smoothness_loss(y_true, y_pred):
    """Penalize jerky phase predictions."""
    diff = (y_pred - y_true)[:, 1:] - (y_pred - y_true)[:, :-1]
    return tf.reduce_mean(tf.square(diff))

def cosine_phase_loss(y_true, y_pred):
    # y_true/y_pred: (batch, window_size, 2)
    return 1.0 - tf.reduce_mean(tf.reduce_sum(y_true * y_pred, axis=-1), axis=-1)

def combined_phase_loss(y_true, y_pred):
    phase_loss = cosine_phase_loss(y_true, y_pred)
    smooth_loss = temporal_smoothness_loss(y_true, y_pred)
    return phase_loss + 0.5 * smooth_loss

def train():
    """Run the full training pipeline: load data, train, and export TFLite."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    today_str = date.today().strftime("%Y_%m_%d")
    model_path = MODEL_DIR / "gru_model.keras"
    scaler_path = MODEL_DIR / "scaler.pkl"

    # --- Load all training CSVs ---
    csv_files = sorted(TRAINING_DATA_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {TRAINING_DATA_DIR}")

    file_data = []
    for csv_file in csv_files:
        features, targets = preprocess_one_csv(csv_file)
        file_data.append((csv_file.name, features, targets))

    # --- File-level train/val split to prevent data leakage ---
    rng = np.random.default_rng(42)
    n_files = len(file_data)
    n_val = max(1, round(n_files * VAL_SPLIT))
    indices = rng.permutation(n_files)
    val_indices = set(indices[:n_val].tolist())
    train_indices = set(indices[n_val:].tolist())

    # --- Fit scaler on training files  ---
    train_features_for_fit = np.vstack(
        [f for j, (_, f, _) in enumerate(file_data) if j in train_indices]
    )
    scaler = RobustScaler()
    scaler.fit(train_features_for_fit)
    joblib.dump(scaler, scaler_path)

    # --- Build windows: whole files go to train or val ---
    x_train_list, y_train_list = [], []
    x_val_list, y_val_list = [], []

    for i, (name, features, targets) in enumerate(file_data):
        features_scaled = np.asarray(scaler.transform(features))
        x_file, y_file = create_sliding_windows(features_scaled, targets, WINDOW_SIZE, STRIDE)
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

    # --- Targets: only the last timestep per window (return_sequences=False) ---
    y_train_phase = y_train[:, -1, :2]         # (N, 2)
    y_train_omega_raw = y_train[:, -1, 2:3]    # (N, 1)
    y_val_phase = y_val[:, -1, :2]
    y_val_omega_raw = y_val[:, -1, 2:3]

    # Normalize omega targets so loss scale matches phase (~[-1,1])
    omega_scaler = StandardScaler()
    y_train_omega = omega_scaler.fit_transform(y_train_omega_raw)  # (N, 1)
    y_val_omega = omega_scaler.transform(y_val_omega_raw)
    joblib.dump(omega_scaler, MODEL_DIR / "omega_scaler.pkl")

    model = build_gru_model(WINDOW_SIZE, x_train.shape[2])
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss={"phase": cosine_phase_loss, "omega": "mse"},
        loss_weights={"phase": 1.0, "omega": 1.0},
    )

    # --- Log model structure and parameter counts ---
    summary_buf = io.StringIO()
    model.summary(print_fn=lambda line: summary_buf.write(line + "\n"))
    logger.info("Model architecture:\n{}", summary_buf.getvalue())
    trainable_params = sum(int(tf.size(w)) for w in model.trainable_weights)
    non_trainable_params = sum(int(tf.size(w)) for w in model.non_trainable_weights)
    logger.info(
        "Parameters — trainable: {}  non-trainable: {}  total: {}",
        trainable_params,
        non_trainable_params,
        trainable_params + non_trainable_params,
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
        EpochLogger(),
    ]

    history = model.fit(
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

    # --- Rename saved checkpoint to include date and best val loss ---
    best_val_loss = min(history.history["val_loss"])
    named_model_path = MODEL_DIR / f"{today_str}_{best_val_loss:.4f}" / f"gru_model.keras"
    named_tflite_path = MODEL_DIR / f"{today_str}_{best_val_loss:.4f}" / f"gru_model_edge.tflite"
    if model_path.exists():
        model_path.rename(named_model_path)
        logger.info(
            "Best model saved as: {} (val_loss={:.6f})",
            named_model_path.name,
            best_val_loss,
        )

    # --- Export to TFLite ---
    best_model = tf.keras.models.load_model(str(named_model_path), compile=False)
    inference_model = build_gru_model(WINDOW_SIZE, x_train.shape[2], batch_size=1)
    inference_model.set_weights(best_model.get_weights())

    converter = tf.lite.TFLiteConverter.from_keras_model(inference_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(named_tflite_path, "wb") as f:
        f.write(tflite_model)
    logger.info("TFLite model saved as: {}", named_tflite_path.name)


if __name__ == "__main__":
    setup_logger()
    train()