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
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from tensorflow.keras.losses import Huber, CosineSimilarity
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.layers import GRU, Dense, Input, LayerNormalization, UnitNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from ao_compensation_model.definitions import (
    BATCH_SIZE,
    DROPOUT_RATE,
    GRU_UNITS,
    LEARNING_RATE,
    MAX_EPOCHS,
    MODEL_DIR,
    TARGET_LEAD,
    TRAINING_DATA_DIR,
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
    angular_velocity = np.asarray(df["filter_hip_vel"].values)

    omega = np.asarray(df["target_omega"].values)
    target_sin = np.asarray(df["target_sin"].values).copy()
    target_cos = np.asarray(df["target_cos"].values).copy()

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
    x = GRU(units=GRU_UNITS, return_sequences=False, dropout=DROPOUT_RATE)(inp)
    phase_out = Dense(units=2, activation="linear",kernel_regularizer=l2(0.001))(x)
    phase_normalized = UnitNormalization(axis=-1, name="phase")(phase_out)
    omega_out = Dense(units=1, activation="linear", name="omega",kernel_regularizer=l2(0.001))(x)
    return Model(inputs=inp, outputs={"phase": phase_normalized, "omega": omega_out})

class EpochLogger(tf.keras.callbacks.Callback):
    """Log phase and omega validation losses after every epoch."""

    def on_train_begin(self, logs=None):
        logger.info("Training started.")
        logger.info(
            "Hyperparameters:\n"
            f"  WINDOW_SIZE: {WINDOW_SIZE}\n"
            f"  STRIDE: {STRIDE}\n"
            f"  TARGET_LEAD: {TARGET_LEAD}\n"
            f"  GRU_UNITS: {GRU_UNITS}\n"
            f"  DROPOUT_RATE: {DROPOUT_RATE}\n"
            f"  BATCH_SIZE: {BATCH_SIZE}\n"
            f"  MAX_EPOCHS: {MAX_EPOCHS}\n"
            f"  LEARNING_RATE: {LEARNING_RATE}\n"
            f"  TRAINING_DATA_DIR: {TRAINING_DATA_DIR}\n"
            f"  MODEL_DIR: {MODEL_DIR}"
        )

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logger.info(
            "Epoch {:3d} | val_loss: {:.6f}  val_phase_loss: {:.6f}  val_omega_loss: {:.6f}",
            epoch + 1,
            logs.get("val_loss", float("nan")),
            logs.get("val_phase_loss", float("nan")),
            logs.get("val_omega_loss", float("nan")),
        )

def train():
    """Run the full training pipeline: load data, train, and export TFLite."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    today_str = date.today().strftime("%Y_%m_%d")
    time_str = pd.Timestamp.now().strftime("%H_%M")
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
    val_subjects = {"L"}

    # --- Fit scaler on training files  ---
    train_features_for_fit = np.vstack(
        [f for j, (_, f, _) in enumerate(file_data)
         if csv_files[j].name.split("_")[3][0] not in val_subjects]
    )
    scaler = RobustScaler()
    scaler.fit(train_features_for_fit)
    joblib.dump(scaler, scaler_path)

     # --- Build windows: whole files go to train or val ---
    x_train_list, y_train_list = [], []
    x_val_list, y_val_list = [], []

    for i, (name, features, targets) in enumerate(file_data):
        features_scaled = np.asarray(scaler.transform(features))
        x_file, y_file = create_sliding_windows(features_scaled, targets, WINDOW_SIZE, STRIDE, TARGET_LEAD)
        if len(x_file) == 0:
            continue

        subject = csv_files[i].name.split("_")[3][0]
        if subject in val_subjects:
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

    y_train_phase = y_train[:, -1, :2]         # (N, 2)
    y_train_omega_raw = y_train[:, -1, 2:3]    # (N, 1)
    y_val_phase = y_val[:, -1, :2]
    y_val_omega_raw = y_val[:, -1, 2:3]

    # Normalize omega targets so loss scale matches phase (~[-1,1])
    omega_scaler = RobustScaler()
    y_train_omega = omega_scaler.fit_transform(y_train_omega_raw)  # (N, 1)
    y_val_omega = omega_scaler.transform(y_val_omega_raw)
    joblib.dump(omega_scaler, MODEL_DIR / "omega_scaler.pkl")

    model = build_gru_model(WINDOW_SIZE, x_train.shape[2])
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE,clipnorm=1.0),
        loss={"phase": "mse", "omega": Huber(delta=0.5)},
        loss_weights={"phase": 3.0, "omega": 1.0},
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
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
        ),
        EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
        ),
        ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        EpochLogger(),
    ]
    last_omega = y_train[:, -1, 0]
    sample_weights = np.where(last_omega == 0, 0.3, 1.0)

    history = model.fit(
        x_train,
        {"phase": y_train_phase, "omega": y_train_omega},
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        sample_weight=sample_weights,
        validation_data=(
            x_val,
            {"phase": y_val_phase, "omega": y_val_omega},
        ),
        callbacks=callbacks,
    )

    # --- Rename saved checkpoint to include date and best val loss ---
    best_val_loss = min(history.history["val_loss"])
    best_model_path = MODEL_DIR / f"{today_str}_{time_str}_{best_val_loss:.4f}"
    os.makedirs(best_model_path)
    best_model = best_model_path/ f"gru_model.keras"
    tflite_path = best_model_path / f"gru_model_edge.tflite"

    if model_path.exists():
        model_path.rename(best_model)
        logger.info(
            "Best model saved as: {} (val_loss={:.6f})",
            best_model.name,
            best_val_loss,
        )

    # --- Export to TFLite ---
    best_model = tf.keras.models.load_model(str(best_model), compile=False)
    inference_model = build_gru_model(WINDOW_SIZE, x_train.shape[2], batch_size=1)
    inference_model.set_weights(best_model.get_weights())

    converter = tf.lite.TFLiteConverter.from_keras_model(inference_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    logger.info("TFLite model saved as: {}", tflite_path.name)
    joblib.dump(scaler, str(best_model_path / "scaler.pkl"))
    joblib.dump(omega_scaler, str(best_model_path / "omega_scaler.pkl"))

if __name__ == "__main__":
    setup_logger()
    train()