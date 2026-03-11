"""Utility functions for signal processing, data preparation, and logging."""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from loguru import logger
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, find_peaks, lfilter, lfilter_zi

from ao_compensation_model.definitions import (
    BANDPASS_HIGHCUT,
    BANDPASS_LOWCUT,
    BANDPASS_ORDER,
    DATE_FORMAT,
    DEFAULT_LOG_FILENAME,
    DEFAULT_LOG_LEVEL,
    ENCODING,
    LOG_DIR,
    SAMPLING_FREQ,
    STATIONARY_THRESHOLD_RATIO,
)

# ---------------------------------------------------------------------------
# Signal Processing
# ---------------------------------------------------------------------------


class RealTimeBandpassFilter:
    """Causal bandpass filter for real-time, sample-by-sample processing."""

    def __init__(self, lowcut: float, highcut: float, fs: int, order: int = 5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        coeffs = butter(order, [low, high], btype="band")
        assert coeffs is not None
        self.b: np.ndarray = np.asarray(coeffs[0])
        self.a: np.ndarray = np.asarray(coeffs[1])
        self.zi = lfilter_zi(self.b, self.a)
        self.is_initialized = False

    def process_point(self, new_value: float) -> float:
        """Filter a single new data point and return the filtered value."""
        if not self.is_initialized:
            self.zi = self.zi * new_value
            self.is_initialized = True
        filtered_array, self.zi = lfilter(self.b, self.a, [new_value], zi=self.zi)
        return filtered_array[0]


def bandpass_filter(
    data: np.ndarray,
    fs: int = SAMPLING_FREQ,
    lowcut: float = BANDPASS_LOWCUT,
    highcut: float = BANDPASS_HIGHCUT,
    order: int = BANDPASS_ORDER,
) -> np.ndarray:
    """Apply a zero-phase Butterworth bandpass filter.

    Removes high-frequency noise and eliminates DC offset so the signal
    oscillates around zero (prerequisite for Hilbert transform, etc.).

    :param data: Input signal array.
    :param fs: Sampling frequency in Hz.
    :param lowcut: Lower cutoff frequency in Hz.
    :param highcut: Upper cutoff frequency in Hz.
    :param order: Filter order.
    :return: Filtered signal array.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    coeffs = butter(order, [low, high], btype="band")
    assert coeffs is not None
    return np.asarray(filtfilt(coeffs[0], coeffs[1], data))


# ---------------------------------------------------------------------------
# Phase & Target Extraction
# ---------------------------------------------------------------------------


def align_ao_phase(
    filtered_signal: np.ndarray,
    ao_phase: np.ndarray,
    dt: float = 1 / SAMPLING_FREQ,
    threshold: float | None = None,
    window_time: float = 1,
    ao_error_threshold: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Align AO phase to Hip_x peaks by a single global time-shift.

    1. Find the 5th peak of the AO phase signal.
    2. Find the nearest Hip_x (sawtooth) peak to that AO peak.
    3. Compute the time-shift and ``np.roll`` the entire AO signal once.
    4. Build per-cycle sawtooth references; if the per-cycle RMS error
       exceeds *ao_error_threshold*, fall back to sawtooth for that cycle.

    :param filtered_signal: Bandpass-filtered Hip_x signal.
    :param ao_phase: Raw Hip_x_ao phase signal (expected in [-pi, pi]).
    :param dt: Sampling period in seconds.
    :param threshold: Amplitude threshold for stationary detection.
        *None* (default) = auto-compute as
        ``STATIONARY_THRESHOLD_RATIO * median(envelope[peaks])``.
    :param window_time: Window length (seconds) for the RMS envelope.
    :param ao_error_threshold: Max RMS error (rad) before falling back to sawtooth.
    :return: (aligned_phase, amplitude_envelope, used_threshold, aligned_ao_full).
    """
    min_distance = int(0.99 / (BANDPASS_HIGHCUT * dt))
    peaks, _ = find_peaks(filtered_signal, height=0, width=min_distance)
    ao_peaks, _ = find_peaks(ao_phase, height=0, width=min_distance)

    # --- Global time-shift using the 5th AO peak ---
    ref_idx = min(4, len(ao_peaks) - 1)  # 0-based index 9 = 10th peak
    ao_ref_pos = ao_peaks[ref_idx]
    # Nearest Hip_x (sawtooth) peak
    nearest_hip_peak = peaks[np.argmin(np.abs(peaks - ao_ref_pos))]
    time_shift = int(nearest_hip_peak - ao_ref_pos)
    shifted_ao = np.roll(ao_phase, time_shift)

    aligned_phase = np.zeros_like(ao_phase)
    for i in range(len(peaks) - 1):
        start, end = peaks[i], peaks[i + 1]
        n_samples = end - start

        # Sawtooth reference: linear ramp from -pi to pi
        sawtooth = np.linspace(-np.pi, np.pi, n_samples)

        # Per-cycle quality check
        rms_error = np.sqrt(np.mean((shifted_ao[start:end] - sawtooth) ** 2))
        if rms_error > ao_error_threshold:
            aligned_phase[start:end] = sawtooth
        else:
            aligned_phase[start:end] = shifted_ao[start:end]

    # RMS amplitude envelope via moving average of squared signal
    window_size = max(1, int(window_time / dt))
    squared_signal = filtered_signal**2
    mean_squared = np.convolve(
        squared_signal, np.ones(window_size) / window_size, mode="same"
    )
    amplitude_envelope = np.sqrt(mean_squared)

    # Auto-compute threshold from median peak amplitude when not specified
    if threshold is None:
        if len(peaks) > 0:
            threshold = STATIONARY_THRESHOLD_RATIO * float(
                np.median(amplitude_envelope[peaks])
            )
        else:
            threshold = 0.0

    # Clamp to zero where stationary
    aligned_phase[amplitude_envelope < threshold] = 0

    return aligned_phase, amplitude_envelope, threshold


def calculate_offline_omega(
    theta_il: np.ndarray,
    time_array: np.ndarray,
    fs: int = SAMPLING_FREQ,
) -> np.ndarray:
    """Calculate highly accurate offline fundamental angular frequency (omega).

    Uses zero-phase filtering and peak-based stride detection to produce a
    smooth, continuous omega curve suitable as a ground-truth target.

    :param theta_il: 1-D array of inter-limb hip flexion angles.
    :param time_array: 1-D array of corresponding timestamps in seconds.
    :param fs: Sampling frequency in Hz.
    :return: 1-D array of continuous offline angular frequency (rad/s).
    """
    # Step 1: Zero-phase low-pass filtering (4th order Butterworth, 10 Hz)
    nyquist = 0.5 * fs
    cutoff = 10 / nyquist
    b, a = butter(4, cutoff, btype="low")
    theta_filtered = filtfilt(b, a, theta_il)

    # Step 2: Detect gait events (peaks = maximum hip flexion)
    min_stride_samples = int(0.5 * fs)
    threshold = float(np.percentile(theta_filtered, 75))
    peaks, _ = find_peaks(
        theta_filtered, distance=min_stride_samples, height=threshold
    )

    if len(peaks) < 2:
        return np.zeros_like(time_array)

    # Step 3: Stride period and discrete omega
    peak_times = time_array[peaks]
    stride_periods = np.diff(peak_times)
    omega_discrete = 2 * np.pi / stride_periods

    # Step 4: Cubic interpolation for a continuous omega curve
    midpoints = peak_times[:-1] + stride_periods / 2
    interp_func = interp1d(
        midpoints, omega_discrete, kind="cubic",
        bounds_error=False, fill_value="extrapolate",
    )
    omega_continuous = interp_func(time_array)

    # Step 5: Zero out non-walking segments
    max_stride_duration = 2.5  # seconds
    for i, period in enumerate(stride_periods):
        if period > max_stride_duration:
            omega_continuous[peaks[i] : peaks[i + 1]] = 0.0

    omega_continuous[: peaks[0]] = 0.0
    omega_continuous[peaks[-1] :] = 0.0

    # Prevent negative frequencies from spline overshoot
    omega_continuous = np.maximum(omega_continuous, 0.0)

    return omega_continuous


def align_omega(
    filtered_signal: np.ndarray,
    ao_omega: np.ndarray,
    time_array: np.ndarray,
    dt: float = 1 / SAMPLING_FREQ,
    fs: int = SAMPLING_FREQ,
    omega_error_threshold: float = 1.0,
    threshold: float | None = None,
    window_time: float = 1,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Align AO omega to the offline ground-truth with per-cycle quality check.

    For each stride cycle the RMS error between the AO-provided omega and the
    offline reference is evaluated.  If the error exceeds
    *omega_error_threshold* the cycle is replaced by the offline omega;
    otherwise the (potentially more precise) AO omega is kept.

    :param filtered_signal: Bandpass-filtered Hip_x signal (used for peak
        detection and amplitude envelope).
    :param ao_omega: Raw AO angular-frequency estimate from the device.
    :param time_array: 1-D array of timestamps in seconds.
    :param dt: Sampling period in seconds.
    :param fs: Sampling frequency in Hz.
    :param omega_error_threshold: Max per-cycle RMS error (rad/s) before
        falling back to the offline omega.
    :param threshold: Amplitude threshold for stationary detection.
        *None* = auto-compute.
    :param window_time: Window length (seconds) for the RMS envelope.
    :return: (aligned_omega, amplitude_envelope, used_threshold).
    """
    # Compute the offline reference omega
    theta_il = filtered_signal  # inter-limb angle approximated by filtered Hip_x
    offline_omega = calculate_offline_omega(theta_il, time_array, fs)

    # Hip_x peaks define stride boundaries
    min_distance = int(0.99 / (BANDPASS_HIGHCUT * dt))
    peaks, _ = find_peaks(filtered_signal, height=0, width=min_distance)

    aligned_omega = np.copy(offline_omega)

    # Per-cycle quality check
    for i in range(len(peaks) - 1):
        start, end = peaks[i], peaks[i + 1]
        rms_error = np.sqrt(np.mean((ao_omega[start:end] - offline_omega[start:end]) ** 2))
        if rms_error <= omega_error_threshold:
            # AO omega is close enough — keep it
            aligned_omega[start:end] = ao_omega[start:end]
        # else: offline_omega already in place

    # RMS amplitude envelope (same as align_ao_phase)
    window_size = max(1, int(window_time / dt))
    squared_signal = filtered_signal ** 2
    mean_squared = np.convolve(
        squared_signal, np.ones(window_size) / window_size, mode="same"
    )
    amplitude_envelope = np.sqrt(mean_squared)

    if threshold is None:
        if len(peaks) > 0:
            threshold = STATIONARY_THRESHOLD_RATIO * float(
                np.median(amplitude_envelope[peaks])
            )
        else:
            threshold = 0.0

    # Zero out stationary segments
    aligned_omega[amplitude_envelope < threshold] = 0.0

    return aligned_omega, amplitude_envelope, threshold


def generate_gru_targets(
    tp_cos: np.ndarray,
    tp_sin: np.ndarray,
) -> np.ndarray:
    """Compute GRU training targets to cos and sin.

    The true phase is decomposed into cosine and sine components.

    :param tp_cos: Cosine of the true phase.
    :param tp_sin: Sine of the true phase.
    :return: Array of shape (N, 2) with columns [target_cos, target_sin].
    """

    target_cos = tp_cos
    target_sin = tp_sin
    return np.column_stack([target_cos, target_sin])


# ---------------------------------------------------------------------------
# Data Windowing
# ---------------------------------------------------------------------------


def create_sliding_windows(
    data: np.ndarray,
    target: np.ndarray,
    window_size: int,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Create overlapping sliding windows from time-series data.

    :param data: Input feature array of shape (T, F).
    :param target: Target array of shape (T, ...).
    :param window_size: Number of time steps per window.
    :param stride: Step size between windows.
    :return: (X_windows, y_windows) arrays.
    """
    x_windows, y_windows = [], []
    for i in range(0, len(data) - window_size, stride):
        x_windows.append(data[i : i + window_size])
        y_windows.append(target[i + window_size])
    return np.array(x_windows), np.array(y_windows)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def create_timestamped_filepath(suffix: str, output_dir: Path, prefix: str) -> Path:
    """Generate a timestamped file path.

    :param suffix: File extension (without dot).
    :param output_dir: Output directory.
    :param prefix: Filename prefix.
    :return: Path to the timestamped file.
    """
    timestamp = datetime.now().strftime(DATE_FORMAT)
    filepath = output_dir / f"{prefix}_{timestamp}.{suffix}"
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.touch(exist_ok=True)
    return filepath


def setup_logger(
    filename: str = DEFAULT_LOG_FILENAME,
    stderr_level: str = DEFAULT_LOG_LEVEL,
    log_level: str = DEFAULT_LOG_LEVEL,
    log_dir: Path | None = None,
) -> Path:
    """Configure the logger with file and stderr outputs.

    :param filename: Base name for the log file.
    :param stderr_level: Logging level for stderr.
    :param log_level: Logging level for the log file.
    :param log_dir: Directory for the log file (defaults to LOG_DIR).
    :return: Path to the created log file.
    """
    logger.remove()
    log_filepath = log_dir if log_dir is not None else LOG_DIR
    filepath_with_time = create_timestamped_filepath(
        output_dir=log_filepath, prefix=filename, suffix="log"
    )
    logger.add(sys.stderr, level=stderr_level)
    logger.add(filepath_with_time, level=log_level, encoding=ENCODING, enqueue=True)
    logger.info(f"Logging to '{filepath_with_time}'.")
    return filepath_with_time
