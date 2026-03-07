"""Test the utils module."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from ao_compensation_model.definitions import LogLevel
from ao_compensation_model.utils import (
    RealTimeBandpassFilter,
    align_ao_phase,
    bandpass_filter,
    create_sliding_windows,
    create_timestamped_filepath,
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
    results = [
        filt.process_point(np.sin(2 * np.pi * 1.0 * i / 100)) for i in range(200)
    ]
    assert len(results) == 200
    assert all(isinstance(v, (float, np.floating)) for v in results)


# ---------------------------------------------------------------------------
# Phase extraction & target generation tests
# ---------------------------------------------------------------------------


def test_generate_gru_targets_shape():
    """generate_gru_targets should return (N, 2) array."""
    n = 100
    tp_cos = np.cos(np.linspace(0, 4 * np.pi, n))
    tp_sin = np.sin(np.linspace(0, 4 * np.pi, n))
    targets = generate_gru_targets(tp_cos, tp_sin)
    assert targets.shape == (n, 2)


def test_generate_gru_targets_values():
    """generate_gru_targets should return cos/sin values."""
    n = 50
    phase = np.linspace(0, 2 * np.pi, n)
    cos_p = np.cos(phase)
    sin_p = np.sin(phase)
    targets = generate_gru_targets(cos_p, sin_p)
    np.testing.assert_allclose(targets[:, 0], cos_p, atol=1e-10)
    np.testing.assert_allclose(targets[:, 1], sin_p, atol=1e-10)


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


def test_align_ao_phase_output_shape():
    """align_ao_phase should return arrays of same length as input."""
    fs = 100
    t = np.arange(0, 5, 1 / fs)
    signal = np.sin(2 * np.pi * 1.0 * t)
    filtered = bandpass_filter(signal, fs=fs)
    # Simulate an AO phase that wraps around [-pi, pi]
    ao_phase = np.mod(2 * np.pi * 1.0 * t + 0.5, 2 * np.pi) - np.pi
    aligned, amplitude, used_thr = align_ao_phase(filtered, ao_phase, dt=1 / fs)
    assert aligned.shape == filtered.shape
    assert amplitude.shape == filtered.shape


def test_align_ao_phase_bounded():
    """Aligned AO phase should be in [-pi, pi]."""
    fs = 100
    t = np.arange(0, 5, 1 / fs)
    signal = 10 * np.sin(2 * np.pi * 1.0 * t)
    filtered = bandpass_filter(signal, fs=fs)
    ao_phase = np.mod(2 * np.pi * 1.0 * t + 1.0, 2 * np.pi) - np.pi
    aligned, _, _ = align_ao_phase(filtered, ao_phase, dt=1 / fs)
    assert np.all(aligned >= -np.pi - 0.1)
    assert np.all(aligned <= np.pi + 0.1)


def test_align_ao_phase_sawtooth_fallback():
    """Bad AO should fall back to sawtooth (linear ramp)."""
    fs = 100
    t = np.arange(0, 5, 1 / fs)
    signal = 10 * np.sin(2 * np.pi * 1.0 * t)
    filtered = bandpass_filter(signal, fs=fs)
    # Deliberately bad AO: random noise
    rng = np.random.default_rng(42)
    bad_ao = rng.uniform(-np.pi, np.pi, len(t))
    aligned, _, _ = align_ao_phase(filtered, bad_ao, dt=1 / fs, ao_error_threshold=0.5)
    # All non-zero segments should be within [-pi, pi] (sawtooth bounds)
    nonzero = aligned[aligned != 0]
    if len(nonzero) > 0:
        assert np.all(nonzero >= -np.pi - 1e-10)
        assert np.all(nonzero <= np.pi + 1e-10)


def test_align_ao_phase_auto_threshold():
    """When threshold=None, auto-compute from amplitude envelope at peaks."""
    fs = 100
    t = np.arange(0, 5, 1 / fs)
    signal = 10 * np.sin(2 * np.pi * 1.0 * t)
    filtered = bandpass_filter(signal, fs=fs)
    ao_phase = np.mod(2 * np.pi * 1.0 * t + 0.5, 2 * np.pi) - np.pi
    _, _, used_thr = align_ao_phase(filtered, ao_phase, dt=1 / fs, threshold=None)
    assert used_thr > 0, "Auto threshold should be positive for a walking signal"


def test_align_ao_phase_manual_threshold():
    """When a manual threshold is given, it should be returned unchanged."""
    fs = 100
    t = np.arange(0, 5, 1 / fs)
    signal = 10 * np.sin(2 * np.pi * 1.0 * t)
    filtered = bandpass_filter(signal, fs=fs)
    ao_phase = np.mod(2 * np.pi * 1.0 * t + 0.5, 2 * np.pi) - np.pi
    _, _, used_thr = align_ao_phase(filtered, ao_phase, dt=1 / fs, threshold=0.42)
    assert used_thr == 0.42


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
