"""Override target_cos/sin to (1, 0) for all standing segments in training CSVs.

Standing is detected using the same amplitude-envelope logic as in
``align_ao_phase`` / ``align_omega``: bandpass-filter Hip_x, compute the
RMS envelope, and threshold at STATIONARY_THRESHOLD_RATIO * median peak
amplitude.
"""

import numpy as np
import pandas as pd
from loguru import logger
from scipy.signal import find_peaks

from ao_compensation_model.definitions import (
    BANDPASS_HIGHCUT,
    SAMPLING_FREQ,
    STATIONARY_THRESHOLD_RATIO,
    TRAINING_DATA_DIR,
)
from ao_compensation_model.utils import bandpass_filter


def compute_stationary_mask(
    raw_hip: np.ndarray,
    fs: int = SAMPLING_FREQ,
    dt: float = 1 / SAMPLING_FREQ,
    window_time: float = 1,
) -> np.ndarray:
    """Return a boolean mask that is True for standing (stationary) samples.

    Replicates the envelope + threshold logic from ``align_ao_phase``.
    """
    filtered = bandpass_filter(raw_hip, fs)

    min_distance = int(0.99 / (BANDPASS_HIGHCUT * dt))
    min_prominence = 0.2 * (np.max(filtered) - np.min(filtered))
    peaks, _ = find_peaks(
        filtered, height=0, distance=min_distance, prominence=min_prominence
    )

    window_size = max(1, int(window_time / dt))
    squared = filtered ** 2
    mean_sq = np.convolve(squared, np.ones(window_size) / window_size, mode="same")
    envelope = np.sqrt(mean_sq)

    if len(peaks) > 0:
        threshold = STATIONARY_THRESHOLD_RATIO * float(np.median(envelope[peaks]))
    else:
        threshold = 0.0

    return envelope < threshold


def fix_standing_targets(csv_path) -> int:
    """Set target_cos=1, target_sin=0 for standing segments in *csv_path*.

    :param csv_path: Path to a training CSV (modified in-place).
    :return: Number of samples changed.
    """
    df = pd.read_csv(csv_path, sep=";")
    raw_hip = np.asarray(df["Hip_x"].values)
    mask = compute_stationary_mask(raw_hip)

    n_changed = int(mask.sum())
    df.loc[mask, "target_cos"] = 1.0
    df.loc[mask, "target_sin"] = 0.0
    df.to_csv(csv_path, index=False, sep=";")
    return n_changed


if __name__ == "__main__":
    csv_files = sorted(TRAINING_DATA_DIR.glob("*.csv"))
    logger.info(f"Found {len(csv_files)} training CSV(s) in {TRAINING_DATA_DIR}")

    for path in csv_files:
        n = fix_standing_targets(path)
        logger.info(f"{path.name}: {n} standing samples → (cos=1, sin=0)")

    logger.success("Done.")
