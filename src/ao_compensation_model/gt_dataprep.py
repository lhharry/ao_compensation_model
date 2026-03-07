"""Generate ground-truth GRU training targets from raw IMU recordings.

Reads a raw CSV, applies bandpass filtering, extracts the true gait phase,
computes true-phase sin/cos targets, and appends them to a new CSV for training.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ao_compensation_model.definitions import (
    RAW_DATA_DIR,
    SAMPLING_FREQ,
    TRAINING_DATA_DIR,
)
from ao_compensation_model.utils import (
    align_ao_phase,
    bandpass_filter,
    generate_gru_targets,
)


def prepare_targets(
    input_path, output_path, fs=SAMPLING_FREQ, threshold=None
):
    """Process a single raw CSV and write the file with appended targets.

    :param input_path: Path to the raw CSV file.
    :param output_path: Path for the output CSV with target columns.
    :param fs: Sampling frequency in Hz.
    :param threshold: Amplitude threshold for stationary detection.
    """
    df = pd.read_csv(input_path, sep=";")

    raw_hip_angle = np.asarray(df["Hip_x"].values)
    ao_phase = np.asarray(df["Hip_x_ao"].values)

    # Bandpass filter to remove noise and DC offset
    filtered_hip = bandpass_filter(raw_hip_angle, fs)

    # Align AO phase so that its peak corresponds to Hip_x peak
    aligned_phase, _, _ = align_ao_phase(
        filtered_hip, ao_phase, threshold=threshold
    )
    tp_cos = np.cos(aligned_phase)
    tp_sin = np.sin(aligned_phase)

    # Compute GRU targets (delta-phi decomposed into cos and sin)
    gru_targets = generate_gru_targets(tp_cos, tp_sin)

    df["target_cos"] = gru_targets[:, 0]
    df["target_sin"] = gru_targets[:, 1]
    df.to_csv(output_path, index=False, sep=";")


def visualize(input_path, fs=SAMPLING_FREQ, threshold=None):
    """Plot the full pipeline: raw signal, envelope, phase, and targets.

    :param input_path: Path to the raw CSV file.
    :param fs: Sampling frequency in Hz.
    :param threshold: Amplitude threshold for stationary detection.
    """
    df = pd.read_csv(input_path, sep=";")

    t = pd.to_datetime(df["Time"], format="%H:%M:%S.%f")
    raw_hip_angle = np.asarray(df["Hip_x"].values)
    ao_raw_phase = np.asarray(df["Hip_x_ao"].values)
    filtered_hip = bandpass_filter(raw_hip_angle, fs)
    aligned_phase, amplitude, used_threshold = align_ao_phase(
        filtered_hip, ao_raw_phase, threshold=threshold
    )
    tp_cos = np.cos(aligned_phase)
    tp_sin = np.sin(aligned_phase)
    gru_targets = generate_gru_targets(tp_cos, tp_sin)

    _, axs = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    axs[0].set_title("Step 1: Raw Kinematics vs Filtered Signal")
    axs[0].plot(t, raw_hip_angle, label="Raw Hip Angle", color="gray", alpha=0.6)
    axs[0].plot(t, filtered_hip, label="Filtered Signal", color="blue")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend()

    axs[1].set_title("Step 2: Amplitude Envelope & Thresholding")
    axs[1].plot(t, amplitude, label="Amplitude Envelope", color="orange")
    axs[1].axhline(
        y=used_threshold, color="red", linestyle="--",
        label=f"Threshold = {used_threshold:.4f}",
    )
    axs[1].fill_between(
        t,
        0,
        amplitude,
        where=(amplitude < used_threshold),
        color="red",
        alpha=0.2,
        label="Detected as Stopped",
    )
    axs[1].grid(True, alpha=0.3)
    axs[1].legend()

    axs[2].set_title("Step 3: Raw AO Phase vs Aligned AO Phase")
    axs[2].plot(
        t, ao_raw_phase, label=r"Raw AO Phase $\phi_{AO}$", color="red", alpha=0.5
    )
    axs[2].plot(
        t, aligned_phase, label=r"Aligned AO Phase", color="green"
    )
    axs[2].set_ylabel("Phase (Rad)")
    axs[2].grid(True, alpha=0.3)
    axs[2].legend()

    axs[3].set_title("Step 4: GRU Target (Aligned AO Phase)")
    axs[3].plot(t, gru_targets[:, 0], label="Target Cos", color="purple")
    axs[3].plot(t, gru_targets[:, 1], label="Target Sin", color="darkblue")
    axs[3].set_ylabel("Phase (Rad)")
    axs[3].set_xlabel("Time")
    axs[3].grid(True, alpha=0.3)
    axs[3].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    csv_name = "20260304_17_13_22_stopgo"
    input_file = RAW_DATA_DIR / f"{csv_name}.csv"
    output_file = TRAINING_DATA_DIR / f"{csv_name}_target.csv"

    prepare_targets(input_file, output_file)
    visualize(output_file)
