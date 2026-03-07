"""Common definitions for the ao_compensation_model package."""

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

np.set_printoptions(precision=3, floatmode="fixed", suppress=True)

# --- Package Directories ---
PACKAGE_DIR: Path = Path(__file__).resolve().parent
DATASET_DIR: Path = PACKAGE_DIR / "dataset"
RAW_DATA_DIR: Path = DATASET_DIR / "raw"
TRAINING_DATA_DIR: Path = DATASET_DIR / "training"
TEST_DATA_DIR: Path = DATASET_DIR / "test"
MODEL_DIR: Path = PACKAGE_DIR / "model"

# --- Project Directories ---
ROOT_DIR: Path = Path("src").parent
DATA_DIR: Path = ROOT_DIR / "data"
LOG_DIR: Path = DATA_DIR / "logs"

# --- Encoding & Formatting ---
ENCODING: str = "utf-8"
DATE_FORMAT = "%Y-%m-%d_%H-%M-%S"

# --- Signal Processing ---
SAMPLING_FREQ: int = 100
BANDPASS_LOWCUT: float = 0.3
BANDPASS_HIGHCUT: float = 3.0
BANDPASS_ORDER: int = 4
STATIONARY_THRESHOLD: float = 0.05
STATIONARY_THRESHOLD_RATIO: float = 0.3

# --- GRU Model ---
WINDOW_SIZE: int = 100
GRU_UNITS: int = 64
DROPOUT_RATE: float = 0.2
BATCH_SIZE: int = 64
MAX_EPOCHS: int = 100
LEARNING_RATE: float = 0.001
VAL_SPLIT: float = 0.2


@dataclass
class LogLevel:
    """Log level."""

    trace: str = "TRACE"
    debug: str = "DEBUG"
    info: str = "INFO"
    success: str = "SUCCESS"
    warning: str = "WARNING"
    error: str = "ERROR"
    critical: str = "CRITICAL"

    def __iter__(self):
        """Iterate over log levels."""
        return iter(asdict(self).values())


DEFAULT_LOG_LEVEL = LogLevel.info
DEFAULT_LOG_FILENAME = "log_file"
