"""Test the utils module."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from ao_compensation_model.definitions import LogLevel
from ao_compensation_model.utils import (
    RealTimeBandpassFilter,
    bandpass_filter,
    create_sliding_windows,
    create_timestamped_filepath,
    extract_true_phase,
    generate_gru_targets,
    setup_logger,
)


def test_logger_init() -> None:
    """Test logger initialization."""
    with TemporaryDirectory() as log_dir:
        log_dir_path = Path(log_dir)
        log_filepath = setup_logger(filename="log_file", log_dir=log_dir_path)
        assert Path(log_filepath).exists()
    assert not Path(log_filepath).exists()


def test_log_level() -> None:
    """Test the log level."""
    # Act
    log_levels = list(LogLevel())

    # Assert
    assert type(log_levels) is list


# ---------------------------------------------------------------------------
# Signal processing tests
# ---------------------------------------------------------------------------


def test_bandpass_filter_removes_dc():
    """Bandpass filter should remove DC offset."""
    fs = 100
    t = np.arange(0, 5, 1 / fs)
    # DC offset of 10 + 1 Hz sine
    signal = 10.0 + np.sin(2 * np.pi * 1.0 * t)
    filtered = bandpass_filter(signal, fs=fs)
    # DC should be near zero
    assert abs(np.mean(filtered)) < 0.5


def test_bandpass_filter_preserves_passband():
    """A 1 Hz signal should largely pass through the default bandpass."""
    fs = 100
    t = np.arange(0, 10, 1 / fs)
    signal = np.sin(2 * np.pi * 1.0 * t)
    filtered = bandpass_filter(signal, fs=fs)
    # Most energy should be preserved
    assert np.std(filtered) > 0.3 * np.std(signal)


def test_realtime_bandpass_filter():
    """RealTimeBandpassFilter should produce output for each sample."""
    filt = RealTimeBandpassFilter(lowcut=0.3, highcut=3.0, fs=100, order=4)
    results = [filt.process_point(np.sin(2 * np.pi * 1.0 * i / 100)) for i in range(200)]
    assert len(results) == 200
    assert all(isinstance(v, (float, np.floating)) for v in results)


# ---------------------------------------------------------------------------
# Phase extraction & target generation tests
# ---------------------------------------------------------------------------


def test_extract_true_phase_output_shape():
    """extract_true_phase should return arrays of same length as input."""
    fs = 100
    t = np.arange(0, 5, 1 / fs)
    signal = np.sin(2 * np.pi * 1.0 * t)
    filtered = bandpass_filter(signal, fs=fs)
    phase, amplitude = extract_true_phase(filtered, dt=1 / fs)
    assert phase.shape == filtered.shape
    assert amplitude.shape == filtered.shape


def test_extract_true_phase_stationary_clamped():
    """Stationary regions (near-zero signal) should produce phase ~ 0."""
    n = 500
    signal = np.zeros(n)
    # A tiny signal — should be detected as stationary
    signal += 0.001 * np.sin(2 * np.pi * np.arange(n) / 100)
    phase, _ = extract_true_phase(signal, dt=0.01, threshold=0.083)
    assert np.all(np.abs(phase) < 0.01)


def test_generate_gru_targets_shape():
    """generate_gru_targets should return (N, 2) array."""
    n = 100
    tp_cos = np.cos(np.linspace(0, 4 * np.pi, n))
    tp_sin = np.sin(np.linspace(0, 4 * np.pi, n))
    ao_cos = np.cos(np.linspace(0, 4 * np.pi, n) + 0.5)
    ao_sin = np.sin(np.linspace(0, 4 * np.pi, n) + 0.5)
    targets = generate_gru_targets(tp_cos, tp_sin, ao_cos, ao_sin)
    assert targets.shape == (n, 2)


def test_generate_gru_targets_identity():
    """When AO phase == true phase, delta-phi should be (1, 0)."""
    n = 50
    phase = np.linspace(0, 2 * np.pi, n)
    cos_p = np.cos(phase)
    sin_p = np.sin(phase)
    targets = generate_gru_targets(cos_p, sin_p, cos_p, sin_p)
    np.testing.assert_allclose(targets[:, 0], 1.0, atol=1e-10)
    np.testing.assert_allclose(targets[:, 1], 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Sliding windows tests
# ---------------------------------------------------------------------------


def test_create_sliding_windows_shape():
    """create_sliding_windows output shapes should be consistent."""
    t, f = 200, 4
    data = np.random.default_rng(0).standard_normal((t, f))
    target = np.random.default_rng(1).standard_normal((t, 2))
    window_size = 50
    x, y = create_sliding_windows(data, target, window_size, stride=1)
    assert x.shape[1] == window_size
    assert x.shape[2] == f
    assert y.shape[1] == 2
    assert x.shape[0] == y.shape[0]


def test_create_sliding_windows_stride():
    """Larger stride should produce fewer windows."""
    t, f = 200, 3
    data = np.random.default_rng(0).standard_normal((t, f))
    target = np.random.default_rng(1).standard_normal((t, 2))
    x1, _ = create_sliding_windows(data, target, 50, stride=1)
    x2, _ = create_sliding_windows(data, target, 50, stride=5)
    assert x2.shape[0] < x1.shape[0]


# ---------------------------------------------------------------------------
# Timestamped file path test
# ---------------------------------------------------------------------------


def test_create_timestamped_filepath():
    """Should create a file with timestamp in the name."""
    with TemporaryDirectory() as tmpdir:
        path = create_timestamped_filepath("log", Path(tmpdir), "test")
        assert path.exists()
        assert path.suffix == ".log"
        assert "test" in path.name
