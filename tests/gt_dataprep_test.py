"""Tests for gt_dataprep module."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

from ao_compensation_model.gt_dataprep import prepare_targets


def _make_raw_csv(path: Path, n: int = 500) -> None:
    """Write a minimal raw CSV that prepare_targets can consume."""
    t = np.arange(n)
    hip_x = 10 * np.sin(2 * np.pi * t / 100)  # ~1 Hz gait cycle at fs=100
    ao_phase = np.linspace(0, 2 * np.pi * 5, n)
    times = pd.date_range("17:00:00", periods=n, freq="10ms").strftime("%H:%M:%S.%f")
    df = pd.DataFrame({"Time": times, "Hip_x": hip_x, "Hip_x_ao": ao_phase})
    df.to_csv(path, index=False, sep=";")


def test_prepare_targets_creates_output():
    """prepare_targets should create an output CSV with target columns."""
    with TemporaryDirectory() as tmpdir:
        raw = Path(tmpdir) / "raw.csv"
        out = Path(tmpdir) / "out.csv"
        _make_raw_csv(raw)

        prepare_targets(raw, out, fs=100)

        assert out.exists()
        df = pd.read_csv(out, sep=";")
        assert "target_cos" in df.columns
        assert "target_sin" in df.columns
        assert len(df) > 0


def test_prepare_targets_values_bounded():
    """Target cos/sin values should be in [-1, 1]."""
    with TemporaryDirectory() as tmpdir:
        raw = Path(tmpdir) / "raw.csv"
        out = Path(tmpdir) / "out.csv"
        _make_raw_csv(raw, n=1000)

        prepare_targets(raw, out, fs=100)

        df = pd.read_csv(out, sep=";")
        assert df["target_cos"].between(-1.01, 1.01).all()
        assert df["target_sin"].between(-1.01, 1.01).all()


def test_visualize_does_not_crash():
    """visualize should run without error (with plt.show mocked)."""
    from ao_compensation_model.gt_dataprep import visualize

    with TemporaryDirectory() as tmpdir:
        raw = Path(tmpdir) / "raw.csv"
        _make_raw_csv(raw, n=500)

        with patch("matplotlib.pyplot.show"):
            visualize(raw, fs=100)
