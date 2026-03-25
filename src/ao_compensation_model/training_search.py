"""Train a GRU model for adaptive-oscillator phase compensation.

Loads labelled CSV files, fits a StandardScaler, trains a GRU model with
sample weighting, and exports the best model as an optimized TFLite file.
"""

import io
import os
from datetime import date

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from pathlib import Path
import random
import itertools

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from loguru import logger
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.layers import GRU, Dense, Input, UnitNormalization, Conv1D, BatchNormalization, AveragePooling1D, ZeroPadding1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from ao_compensation_model.definitions import (
    DROPOUT_RATE,
    LEARNING_RATE,
    MAX_EPOCHS,
    MODEL_DIR,
    TARGET_LEAD,
    TRAINING_DATA_DIR,
    WINDOW_SIZE,
    STRIDE,
)
from ao_compensation_model.utils import create_sliding_windows, setup_logger

def set_seed(seed: int = 42):
    """Set global random seeds for exact reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['PYTHONHASHSEED'] = str(seed)

def setup_gpu():
    """Configure TensorFlow to use the GPU with dynamic memory allocation."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth so TF doesn't hoard all VRAM at startup
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            logger.info(f"Successfully configured GPU. {len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs available.")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            logger.error(f"RuntimeError during GPU configuration: {e}")
    else:
        logger.warning("No compatible GPU found. TensorFlow will fall back to CPU for training.")

def preprocess_one_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a single labelled CSV and return features and targets.

    :param csv_path: Path to a training CSV with target_cos / target_sin columns.
    :return: (features, targets) arrays. Targets have columns [sin, cos].
    """
    df = pd.read_csv(csv_path, sep=";")

    raw_angle = np.asarray(df["Hip_x"].values)
    angular_velocity = np.asarray(df["Hip_vel"].values)

    target_sin = np.asarray(df["target_sin"].values).copy()
    target_cos = np.asarray(df["target_cos"].values).copy()

    targets = np.column_stack([target_sin, target_cos])
    features = np.column_stack([raw_angle, angular_velocity])
    return features, targets

def build_gru_model(
    window_size: int,
    n_features: int,
    filters: int,
    kernel_size: int,
    pool_size: int,
    gru_units: int,
    batch_size: int | None = None,
) -> Model:
    """Construct the GRU model architecture.

    :param window_size: Number of time steps per input window.
    :param n_features: Number of input features.
    :param filters: Number of Conv1D filters.
    :param kernel_size: Conv1D kernel size.
    :param pool_size: AveragePooling1D pool size.
    :param gru_units: Number of GRU units.
    :param batch_size: Fixed batch size (set to 1 for inference / TFLite export).
    :return: Keras Model (uncompiled).
    """
    inp = Input(shape=(window_size, n_features), batch_size=batch_size)
    x_filter = Conv1D(filters=filters, kernel_size=kernel_size, padding="causal", activation="linear")(inp)
    x_norm = BatchNormalization()(x_filter)
    x_padded = ZeroPadding1D(padding=(pool_size - 1, 0))(x_norm)
    x_pool = AveragePooling1D(pool_size=pool_size, strides=1, padding="valid")(x_padded)
    x = GRU(units=gru_units, return_sequences=False, dropout=DROPOUT_RATE)(x_pool)
    phase_out = Dense(units=2, activation="linear", kernel_regularizer=l2(0.001))(x)
    phase_normalized = UnitNormalization(axis=1, name="phase")(phase_out)
    return Model(inputs=inp, outputs=phase_normalized)

class EpochLogger(tf.keras.callbacks.Callback):
    """Log validation loss after every epoch."""

    def __init__(self, gru_units, batch_size, kernel_size, filters, pool_size):
        super().__init__()
        self.gru_units = gru_units
        self.batch_size = batch_size
        self.kernel_size = kernel_size
        self.filters = filters
        self.pool_size = pool_size

    def on_train_begin(self, logs=None):
        logger.info("Training started.")
        logger.info(
            "Hyperparameters:\n"
            f"  WINDOW_SIZE: {WINDOW_SIZE}\n"
            f"  STRIDE: {STRIDE}\n"
            f"  TARGET_LEAD: {TARGET_LEAD}\n"
            f"  GRU_UNITS: {self.gru_units}\n"
            f"  DROPOUT_RATE: {DROPOUT_RATE}\n"
            f"  FILTERS: {self.filters}\n"
            f"  KERNEL_SIZE: {self.kernel_size}\n"
            f"  POOL_SIZE: {self.pool_size}\n"
            f"  BATCH_SIZE: {self.batch_size}\n"
            f"  MAX_EPOCHS: {MAX_EPOCHS}\n"
            f"  LEARNING_RATE: {LEARNING_RATE}\n"
            f"  TRAINING_DATA_DIR: {TRAINING_DATA_DIR}\n"
            f"  MODEL_DIR: {MODEL_DIR}"
        )

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logger.info(
            "Epoch {:3d} | val_loss: {:.6f}",
            epoch + 1,
            logs.get("val_loss", float("nan")),
        )

def train():
    """Run the full training pipeline: load data, train, and export TFLite."""
    setup_gpu()
    set_seed(42)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    today_str = date.today().strftime("%Y_%m_%d")
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
    val_subjects = {"LG"} 

    # --- Fit scaler on training files  ---
    train_features_for_fit = np.vstack(
        [f for j, (_, f, _) in enumerate(file_data)
         if csv_files[j].name.split("_")[-2] not in val_subjects]
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

        subject = csv_files[i].name.split("_")[-2]
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
    idx = np.random.permutation(len(x_train))
    x_train, y_train = x_train[idx], y_train[idx]

    y_train_phase = y_train[:, -1, :2]         # (N, 2)
    y_val_phase = y_val[:, -1, :2]

    # --- Hyperparameter Search Grid ---
    gru_units_grid = [256, 128, 64, 32, 16, 8]
    batch_size_grid = [64]
    kernel_size_grid = [20, 10, 5]
    filters_grid = [16, 8, 4, 2]
    pool_size_grid = [5, 3, 1]

    hyperparameters = list(itertools.product(
        gru_units_grid, batch_size_grid, kernel_size_grid, filters_grid, pool_size_grid
    ))

    logger.info("Starting hyperparameter search with {} combinations.", len(hyperparameters))

    best_overall_val_loss = float('inf')
    best_overall_config = None

    for (gru_units, batch_size, kernel_size, filters, pool_size) in hyperparameters:
        logger.info(
            "--- Testing Config: GRU={} | BATCH={} | KERNEL={} | FILTERS={} | POOL={} ---",
            gru_units, batch_size, kernel_size, filters, pool_size
        )

        time_str = pd.Timestamp.now().strftime("%H_%M_%S")
        run_name = f"gru_{gru_units}_bs_{batch_size}_k_{kernel_size}_f_{filters}_p_{pool_size}"
        model_path = MODEL_DIR / f"{run_name}.keras"

        model = build_gru_model(WINDOW_SIZE, x_train.shape[2], filters, kernel_size, pool_size, gru_units)
        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE, clipnorm=1.0),
            loss="mse",
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
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, min_lr=1e-6, verbose=1),
            EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True, verbose=1),
            ModelCheckpoint(filepath=str(model_path), monitor="val_loss", save_best_only=True, verbose=1),
            EpochLogger(gru_units, batch_size, kernel_size, filters, pool_size),
        ]

        history = model.fit(
            x_train,
            y_train_phase,
            epochs=MAX_EPOCHS,
            batch_size=batch_size,
            validation_data=(x_val, y_val_phase),
            callbacks=callbacks,
            verbose=0  # Suppressing inner epoch bars to avoid cluttered output
        )

        best_val_loss = min(history.history["val_loss"])
        best_model_path = MODEL_DIR / f"{today_str}_{time_str}_{best_val_loss:.4f}_{run_name}"
        os.makedirs(best_model_path, exist_ok=True)
        best_model_file = best_model_path / "gru_model.keras"
        tflite_path = best_model_path / "gru_model_edge.tflite"

        if model_path.exists():
            model_path.rename(best_model_file)
            logger.info(
                "Best model for this run saved as: {} (val_loss={:.6f})",
                best_model_file.name,
                best_val_loss,
            )

            # --- Export to TFLite ---
            best_model = tf.keras.models.load_model(str(best_model_file), compile=False)
            inference_model = build_gru_model(WINDOW_SIZE, x_train.shape[2], filters, kernel_size, pool_size, gru_units, batch_size=1)
            inference_model.set_weights(best_model.get_weights())

            converter = tf.lite.TFLiteConverter.from_keras_model(inference_model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter.convert()

            with open(tflite_path, "wb") as f:
                f.write(tflite_model)
            logger.info("TFLite model saved as: {}", tflite_path.name)
            joblib.dump(scaler, str(best_model_path / "scaler.pkl"))

        if best_val_loss < best_overall_val_loss:
            best_overall_val_loss = best_val_loss
            best_overall_config = (gru_units, batch_size, kernel_size, filters, pool_size)
            logger.info("*** New best overall config! Val Loss: {:.6f} ***", best_val_loss)

        # Clear session to prevent memory leaks from compounding in loops
        tf.keras.backend.clear_session()

    logger.success(
        "Hyperparameter search complete. Best Val Loss: {:.6f} with config {}",
        best_overall_val_loss,
        best_overall_config
    )

if __name__ == "__main__":
    setup_logger()
    train()