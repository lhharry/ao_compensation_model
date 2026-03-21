import os
import joblib
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque

import ai_edge_litert.interpreter as tflite
from ao_compensation_model.definitions import (
    TEST_DATA_DIR,
    WINDOW_SIZE
)

fs = 100 # Sampling frequency (Hz)
dt = 1 / fs
script_dir = Path(__file__).resolve().parent
path = 'model\\2026_03_21_12_46_0.0680'
model_path= os.path.join(script_dir, path, 'gru_model_edge.tflite')
scaler_path = os.path.join(script_dir, path,'scaler.pkl')

_stream_state = {
    "model": None,
    "scaler": joblib.load(scaler_path),
    "feature_buffer": deque(maxlen=WINDOW_SIZE),
}

# -----------------------------------------------------------------
# 1. Data Preparation
# -----------------------------------------------------------------
def _as_scalar(value):
    arr = np.asarray(value)
    if arr.size == 0:
        return 0.0
    return float(arr.reshape(-1)[-1])

def load_test_data(csv_path: Path) -> dict[str, np.ndarray]:
    """Read a test CSV and return raw column arrays as a dictionary.
    :param csv_path: Full path to the test CSV file.
    :return: Dictionary mapping column names to numpy arrays.
    """
    df = pd.read_csv(csv_path, sep=";")
    return {
        "raw_angle": np.asarray(df["raw_angle"].values),
        "filtered_angle": np.asarray(df["filtered_angle"].values),
        "angular_velocity": np.asarray(df["raw_velocity"].values),
        "filtered_angular_velocity": np.asarray(df["filtered_velocity"].values),
    }

def data_preparation(raw_hip_angle_left, input_velocity):
    raw_hip_angle_left = _as_scalar(raw_hip_angle_left)
    input_velocity = _as_scalar(input_velocity)

    feature_X = np.array([[raw_hip_angle_left, input_velocity]], dtype=np.float32)
    features_scaled = _stream_state["scaler"].transform(feature_X)

    _stream_state["feature_buffer"].append(features_scaled[0])
    if len(_stream_state["feature_buffer"]) < WINDOW_SIZE:
        return None

    features = np.array(_stream_state["feature_buffer"], dtype=np.float32)[np.newaxis, :, :]
    return features
# -----------------------------------------------------------------
# 2. Load Model
# -----------------------------------------------------------------
def build_model():
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# -----------------------------------------------------------------
# 5. Inference & Alignment
# -----------------------------------------------------------------
def model_inference(model, input_angle):
    input_details = model.get_input_details()
    output_details = model.get_output_details()
    model.set_tensor(input_details[0]['index'], input_angle)
    model.invoke()
    y_pred = model.get_tensor(output_details[0]['index'])[0]
    pred_sin = y_pred[0]
    pred_cos = y_pred[1]
    predicted_phase = np.arctan2(pred_sin, pred_cos)
    return float(predicted_phase)


def reset_stream_state():
    """Reset stream buffers so a new sequence starts cleanly."""
    _stream_state["feature_buffer"].clear()

def load_model(input_angle, input_velocity):
    """Load/use the pre-trained GRU model and return one predicted phase value."""
    if _stream_state["model"] is None:
        _stream_state["model"] = build_model()
    input_features = data_preparation(input_angle, input_velocity)
    if input_features is None:
        return 0.0
    gait_phase_prediction = model_inference(_stream_state["model"], input_features)
    return gait_phase_prediction


def predict_sequence(input_angle, input_velocity) -> np.ndarray:
    """Run streaming prediction over full sequences and return predicted phase."""
    reset_stream_state()
    preds = []
    for angle_i, velocity_i in zip(input_angle, input_velocity):
        preds.append(load_model(angle_i, velocity_i))
    return np.asarray(preds, dtype=np.float32)


def plot_prediction(data: dict[str, np.ndarray], predicted_phase: np.ndarray) -> None:
    """Plot kinematics and model-predicted phase, similar to validation view."""
    t = np.arange(len(predicted_phase)) * dt
    _, axs = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    axs[0].set_title("1. Hip Angle", fontsize=14, fontweight="bold")
    axs[0].plot(t, data["raw_angle"], label="Raw Hip Angle", color="gray", alpha=0.8)
    axs[0].plot(t, data["filtered_angle"], label="Filtered Hip Angle", color="blue", alpha=0.8)
    axs[0].set_ylabel("Angle")
    axs[0].legend(loc="upper right")
    axs[0].grid(True, alpha=0.3)

    axs[1].set_title("2. Hip Velocity", fontsize=14, fontweight="bold")
    axs[1].plot(t, data["angular_velocity"], label="Raw Hip Velocity", color="gray", alpha=0.8)
    axs[1].plot(
        t,
        data["filtered_angular_velocity"],
        label="Filtered Hip Velocity",
        color="blue",
        alpha=0.8,
    )
    axs[1].set_ylabel("Velocity")
    axs[1].legend(loc="upper right")
    axs[1].grid(True, alpha=0.3)

    axs[2].set_title("3. Model Predicted Phase", fontsize=14, fontweight="bold")
    axs[2].plot(t, predicted_phase, label="Predicted Phase (GRU)", color="green", linewidth=2)
    axs[2].set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    axs[2].set_yticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
    axs[2].set_ylabel("Phase (Rad)")
    axs[2].set_xlabel("Time (Seconds)")
    axs[2].legend(loc="upper right")
    axs[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def validate_prediction(csv_name: str) -> np.ndarray:
    """Run prediction-only validation and show plots."""
    data = load_test_data(TEST_DATA_DIR / csv_name)
    predicted_phase = predict_sequence(data["raw_angle"], data["angular_velocity"])
    plot_prediction(data, predicted_phase)
    return predicted_phase

if __name__ == "__main__":
    validate_prediction("test_set.csv")