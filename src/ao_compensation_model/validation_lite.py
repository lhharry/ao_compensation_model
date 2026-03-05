"""Validate the optimized TFLite GRU model on test data.

Simulates frame-by-frame edge inference and visualises the comparison
between original AO phase, enhanced (AO + GRU) phase, and ground truth.
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ai_edge_litert.interpreter as tflite

from ao_compensation_model.definitions import (
    MODEL_DIR,
    SAMPLING_FREQ,
    TEST_DATA_DIR,
    WINDOW_SIZE,
)
from ao_compensation_model.utils import create_sliding_windows


def validate(csv_name: str):
    """Run TFLite inference on a test CSV and plot the results.

    :param csv_name: Filename of the test CSV (without directory).
    """
    dataset_path = TEST_DATA_DIR / csv_name
    model_path = MODEL_DIR / "gru_model_optimized.tflite"
    scaler_path = MODEL_DIR / "scaler.pkl"

    # --- Load and prepare data ---
    df = pd.read_csv(dataset_path, sep=";")

    raw_angle = df["Hip_x"].values
    ao_gait_phase = df["Hip_x_ao"].values
    angular_velocity = df["Hip_x_vel"].values
    omega = df["Hip_x_omega"].values
    domega = df["Hip_x_domega"].values
    pred_gp = df["Hip_x_gp"].values

    ao_phase_sin = np.sin(ao_gait_phase)
    ao_phase_cos = np.cos(ao_gait_phase)
    pred_sin = np.sin(pred_gp)
    pred_cos = np.cos(pred_gp)

    target_sin = pred_cos * ao_phase_cos + pred_sin * ao_phase_sin
    target_cos = pred_sin * ao_phase_cos - pred_cos * ao_phase_sin
    gru_targets = np.column_stack([target_sin, target_cos])

    features = np.column_stack(
        [raw_angle, angular_velocity, omega, domega, ao_phase_sin, ao_phase_cos]
    )

    scaler = joblib.load(scaler_path)
    features_scaled = scaler.transform(features)

    x, y = create_sliding_windows(features_scaled, gru_targets, WINDOW_SIZE, stride=1)
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    print(f"Test input shape: {x.shape}")

    # --- TFLite inference (frame-by-frame) ---
    print("Loading TFLite model...")
    interpreter = tflite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("Running frame-by-frame inference...")
    y_pred_list = []
    for i in range(len(x)):
        interpreter.set_tensor(input_details[0]["index"], x[i : i + 1])
        interpreter.invoke()
        y_pred_list.append(interpreter.get_tensor(output_details[0]["index"])[0])

    y_pred = np.array(y_pred_list)
    pred_sin_out = y_pred[:, 0]
    pred_cos_out = y_pred[:, 1]

    # Align arrays after windowing
    offset = WINDOW_SIZE
    aligned_ao_phase = ao_gait_phase[offset:]
    aligned_raw_angle = raw_angle[offset:]
    time_axis = np.arange(len(aligned_ao_phase)) / SAMPLING_FREQ

    # Reconstruct true phase from targets
    true_phase_cos = (
        ao_phase_cos[offset:] * target_cos[offset:]
        - ao_phase_sin[offset:] * target_sin[offset:]
    )
    true_phase_sin = (
        ao_phase_sin[offset:] * target_cos[offset:]
        + ao_phase_cos[offset:] * target_sin[offset:]
    )
    aligned_true_phase = np.arctan2(true_phase_sin, true_phase_cos)

    # Reconstruct enhanced phase from GRU predictions
    enhanced_cos = (
        ao_phase_cos[offset:] * pred_cos_out - ao_phase_sin[offset:] * pred_sin_out
    )
    enhanced_sin = (
        ao_phase_sin[offset:] * pred_cos_out + ao_phase_cos[offset:] * pred_sin_out
    )
    enhanced_phase = np.arctan2(enhanced_sin, enhanced_cos)

    # --- Visualisation ---
    fig, axs = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    axs[0].set_title(
        "1. Hip Kinematics (Stop-to-Walking / Walking-to-Stop Events)",
        fontsize=14,
        fontweight="bold",
    )
    axs[0].plot(
        time_axis, aligned_raw_angle,
        label="Raw Hip Angle (deg)", color="gray", alpha=0.8,
    )
    axs[0].set_ylabel("Angle")
    axs[0].legend(loc="upper right")
    axs[0].grid(True, alpha=0.3)

    axs[1].set_title(
        "2. Predicted Sin/Cos vs Target", fontsize=14, fontweight="bold"
    )
    axs[1].plot(
        time_axis, target_sin[offset:],
        label="Target Sin", color="blue", linewidth=2,
    )
    axs[1].plot(
        time_axis, target_cos[offset:],
        label="Target Cos", color="orange", linewidth=2,
    )
    axs[1].plot(
        time_axis, pred_sin_out,
        label="Predicted Sin", color="purple", linestyle="--", linewidth=1.5,
    )
    axs[1].plot(
        time_axis, pred_cos_out,
        label="Predicted Cos", color="green", linestyle="--", linewidth=1.5,
    )
    axs[1].set_ylabel("Amplitude")
    axs[1].legend(loc="upper right")
    axs[1].grid(True, alpha=0.3)

    axs[2].set_title(
        r"3. Final Phase Comparison [$-\pi, \pi$]: Original AO vs Enhanced (AO + GRU)",
        fontsize=14,
        fontweight="bold",
    )
    axs[2].plot(
        time_axis, aligned_true_phase,
        label="True Phase", color="green", linewidth=2.5, alpha=0.6,
    )
    axs[2].plot(
        time_axis, aligned_ao_phase,
        label="Original AO Phase", color="red", linestyle="--", linewidth=1.5,
    )
    axs[2].plot(
        time_axis, enhanced_phase,
        label="Enhanced Phase (AO + GRU)", color="blue", linewidth=2,
    )
    axs[2].set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    axs[2].set_yticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
    axs[2].set_ylabel("Gait Phase (Rad)")
    axs[2].set_xlabel("Time (Seconds)")
    axs[2].legend(loc="upper right")
    axs[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    validate("20260304_14_26_34_4km_stopgo.csv")
