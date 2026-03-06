"""Validate the optimized TFLite GRU model on test data.

Simulates frame-by-frame edge inference and visualises the comparison
between original AO phase, GRU-predicted phase, and ground truth.
"""

from dataclasses import dataclass
from pathlib import Path

import ai_edge_litert.interpreter as tflite  # type: ignore[import-untyped]
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ao_compensation_model.definitions import (
    MODEL_DIR,
    SAMPLING_FREQ,
    TEST_DATA_DIR,
    WINDOW_SIZE,
)
from ao_compensation_model.utils import create_sliding_windows

# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    """Holds all arrays produced by the validation pipeline."""

    time_axis: np.ndarray
    raw_angle: np.ndarray
    ao_phase: np.ndarray
    true_phase: np.ndarray
    enhanced_phase: np.ndarray
    target_sin: np.ndarray
    target_cos: np.ndarray
    target_omega: np.ndarray
    pred_sin: np.ndarray
    pred_cos: np.ndarray
    pred_omega: np.ndarray


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------


def load_test_data(csv_path: Path) -> dict[str, np.ndarray]:
    """Read a test CSV and return raw column arrays as a dictionary.

    :param csv_path: Full path to the test CSV file.
    :return: Dictionary mapping column names to numpy arrays.
    """
    df = pd.read_csv(csv_path, sep=";")
    return {
        "raw_angle": np.asarray(df["Hip_x"].values),
        "ao_gait_phase": np.asarray(df["Hip_x_ao"].values),
        "angular_velocity": np.asarray(df["Hip_x_vel"].values),
        "omega": np.asarray(df["Hip_x_omega"].values),
        "target_cos": np.asarray(df["target_cos"].values),
        "target_sin": np.asarray(df["target_sin"].values),
    }


def prepare_features_and_targets(
    data: dict[str, np.ndarray],
    scaler_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Scale features and build sliding windows for inference.

    :param data: Raw column arrays from :func:`load_test_data`.
    :param scaler_path: Path to the fitted StandardScaler pickle.
    :return: (x_windows, y_windows, target_sin, target_cos, target_omega).
    """
    target_sin = data["target_sin"]
    target_cos = data["target_cos"]
    target_omega = data["omega"]
    targets = np.column_stack([target_sin, target_cos, target_omega])

    features = np.column_stack(
        [
            data["raw_angle"],
            data["angular_velocity"],
        ]
    )

    scaler = joblib.load(scaler_path)
    features_scaled = scaler.transform(features)

    x, y = create_sliding_windows(features_scaled, targets, WINDOW_SIZE, stride=1)
    return x.astype(np.float32), y.astype(np.float32), target_sin, target_cos, target_omega


def run_tflite_inference(x: np.ndarray, model_path: Path) -> np.ndarray:
    """Run frame-by-frame TFLite inference (simulating edge deployment).

    :param x: Input windows of shape (N, WINDOW_SIZE, n_features).
    :param model_path: Path to the optimized .tflite model.
    :return: Predictions of shape (N, 3) with [sin, cos, omega] columns.
    """
    interpreter = tflite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Identify which output is phase (shape 2) and which is omega (shape 1)
    if output_details[0]["shape"][-1] == 2:
        phase_idx, omega_idx = 0, 1
    else:
        phase_idx, omega_idx = 1, 0

    predictions = []
    for i in range(len(x)):
        interpreter.set_tensor(input_details[0]["index"], x[i : i + 1])
        interpreter.invoke()
        phase = interpreter.get_tensor(output_details[phase_idx]["index"])[0]
        omega = interpreter.get_tensor(output_details[omega_idx]["index"])[0]
        predictions.append(np.concatenate([phase, omega]))

    return np.array(predictions)


def reconstruct_phases(
    data: dict[str, np.ndarray],
    target_sin: np.ndarray,
    target_cos: np.ndarray,
    target_omega: np.ndarray,
    pred_sin: np.ndarray,
    pred_cos: np.ndarray,
    pred_omega: np.ndarray,
) -> ValidationResult:
    """Reconstruct true and enhanced gait phases from sin/cos components.

    :param data: Raw column arrays from :func:`load_test_data`.
    :param target_sin: Ground-truth true-phase sine component.
    :param target_cos: Ground-truth true-phase cosine component.
    :param target_omega: Ground-truth omega values.
    :param pred_sin: Predicted true-phase sine component.
    :param pred_cos: Predicted true-phase cosine component.
    :param pred_omega: Predicted omega values.
    :return: A :class:`ValidationResult` with all aligned arrays.
    """
    offset = WINDOW_SIZE
    ao_phase = data["ao_gait_phase"]

    return ValidationResult(
        time_axis=np.arange(len(ao_phase) - offset) / SAMPLING_FREQ,
        raw_angle=data["raw_angle"][offset:],
        ao_phase=ao_phase[offset:],
        true_phase=np.arctan2(target_sin[offset:], target_cos[offset:]),
        enhanced_phase=np.arctan2(pred_sin, pred_cos),
        target_sin=target_sin[offset:],
        target_cos=target_cos[offset:],
        target_omega=target_omega[offset:],
        pred_sin=pred_sin,
        pred_cos=pred_cos,
        pred_omega=pred_omega,
    )


def plot_results(result: ValidationResult) -> None:
    """Visualise the validation results in a 3-panel figure.

    :param result: A :class:`ValidationResult` produced by :func:`reconstruct_phases`.
    """
    _, axs = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
    t = result.time_axis

    # Panel 1 — Raw kinematics
    axs[0].set_title(
        "1. Hip Kinematics (Stop-to-Walking / Walking-to-Stop Events)",
        fontsize=14,
        fontweight="bold",
    )
    axs[0].plot(
        t, result.raw_angle, label="Raw Hip Angle (deg)", color="gray", alpha=0.8
    )
    axs[0].set_ylabel("Angle")
    axs[0].legend(loc="upper right")
    axs[0].grid(True, alpha=0.3)

    # Panel 2 — Sin/Cos predictions vs targets
    axs[1].set_title("2. Predicted Sin/Cos vs Target", fontsize=14, fontweight="bold")
    axs[1].plot(t, result.target_sin, label="Target Sin", color="blue", linewidth=2)
    axs[1].plot(t, result.target_cos, label="Target Cos", color="orange", linewidth=2)
    axs[1].plot(
        t,
        result.pred_sin,
        label="Predicted Sin",
        color="purple",
        linestyle="--",
        linewidth=1.5,
    )
    axs[1].plot(
        t,
        result.pred_cos,
        label="Predicted Cos",
        color="green",
        linestyle="--",
        linewidth=1.5,
    )
    axs[1].set_ylabel("Amplitude")
    axs[1].legend(loc="upper right")
    axs[1].grid(True, alpha=0.3)

    # Panel 3 — Phase comparison
    # Panel 3 — Omega comparison
    axs[2].set_title("3. Omega: True vs Predicted", fontsize=14, fontweight="bold")
    axs[2].plot(
        t, result.target_omega, label="True Omega", color="blue", linewidth=2,
    )
    axs[2].plot(
        t, result.pred_omega, label="Predicted Omega", color="red", linestyle="--", linewidth=1.5,
    )
    axs[2].set_ylabel("Omega (rad/s)")
    axs[2].legend(loc="upper right")
    axs[2].grid(True, alpha=0.3)

    # Panel 4 — Phase comparison
    axs[3].set_title(
        r"4. Final Phase Comparison [$-\pi, \pi$]: Original AO vs Enhanced (AO + GRU)",
        fontsize=14,
        fontweight="bold",
    )
    axs[3].plot(
        t,
        result.true_phase,
        label="True Phase",
        color="green",
        linewidth=2.5,
        alpha=0.6,
    )
    axs[3].plot(
        t,
        result.ao_phase,
        label="Original AO Phase",
        color="red",
        linestyle="--",
        linewidth=1.5,
    )
    axs[3].plot(
        t,
        result.enhanced_phase,
        label="Enhanced Phase (GRU)",
        color="blue",
        linewidth=2,
    )
    axs[3].set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    axs[3].set_yticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
    axs[3].set_ylabel("Gait Phase (Rad)")
    axs[3].set_xlabel("Time (Seconds)")
    axs[3].legend(loc="upper right")
    axs[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def validate(csv_name: str) -> ValidationResult:
    """Run the full validation pipeline: load, infer, reconstruct, and plot.

    :param csv_name: Filename of the test CSV (without directory).
    :return: A :class:`ValidationResult` with all computed arrays.
    """
    data = load_test_data(TEST_DATA_DIR / csv_name)
    x, _, target_sin, target_cos, target_omega = prepare_features_and_targets(
        data, MODEL_DIR / "scaler.pkl"
    )
    y_pred = run_tflite_inference(x, MODEL_DIR / "gru_model_optimized.tflite")

    # Denormalize omega predictions
    omega_scaler_path = MODEL_DIR / "omega_scaler.pkl"
    if omega_scaler_path.exists():
        omega_scaler = joblib.load(omega_scaler_path)
        pred_omega = omega_scaler.inverse_transform(y_pred[:, 2:3]).ravel()
    else:
        pred_omega = y_pred[:, 2]

    result = reconstruct_phases(
        data,
        target_sin,
        target_cos,
        target_omega,
        pred_sin=y_pred[:, 0],
        pred_cos=y_pred[:, 1],
        pred_omega=pred_omega,
    )
    plot_results(result)
    return result


if __name__ == "__main__":
    validate("20260304_14_26_34_4km_stopgo.csv")
