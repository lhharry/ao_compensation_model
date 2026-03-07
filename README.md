# ao_compensation_model

[![CI](https://github.com/lhharry/ao_compensation_model/actions/workflows/ci.yml/badge.svg)](https://github.com/lhharry/ao_compensation_model/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/ao-compensation-model)](https://pypi.org/project/ao-compensation-model/)
[![Docker](https://github.com/lhharry/ao_compensation_model/actions/workflows/docker.yml/badge.svg)](https://github.com/lhharry/ao_compensation_model/actions/workflows/docker.yml)

A GRU-based compensation model that improves the performance of adaptive oscillators (AOs) during stop-go and go-stop gait transitions. The model learns the phase error between the AO output and the ground-truth gait phase, and applies a real-time correction on edge devices via TFLite.

## Pipeline

| Step | Command | Description |
|------|---------|-------------|
| 1 | `prep` | Bandpass-filters raw IMU hip angles, extracts ground-truth gait phase, and computes delta-phi training targets. |
| 2 | `train` | Trains a GRU network on sliding windows of AO features and exports an optimized TFLite model. |
| 3 | `validate` | Runs frame-by-frame TFLite inference on test data and visualises AO phase vs. enhanced phase vs. ground truth. |
| 4 | `txt2csv` | Converts raw sensor text files (tab / comma / semicolon delimited) in a folder to semicolon-delimited CSVs. |

## Install

From PyPI:

```bash
pip install ao-compensation-model
```

From source:

```bash
git clone https://github.com/lhharry/ao_compensation_model.git
cd ao_compensation_model
uv sync
```

## Usage

### CLI

All commands follow the pattern:

```bash
uv run python -m ao_compensation_model <command> [options]
```

#### Data Preparation

```bash
# Prepare ground-truth targets from all raw CSVs
uv run python -m ao_compensation_model prep

# Prepare a single file
uv run python -m ao_compensation_model prep --file 20260304_17_13_22_stopgo.csv

# Prepare with a manual stationary threshold (default: auto)
uv run python -m ao_compensation_model prep --file recording.csv --threshold 0.1
```

#### Training

```bash
uv run python -m ao_compensation_model train
```

#### Validation

```bash
# Validate all test files
uv run python -m ao_compensation_model validate

# Validate a specific test file
uv run python -m ao_compensation_model validate --file 20260304_14_26_34_4km_stopgo.csv
```

#### File Conversion

```bash
# Opens a folder picker GUI
uv run python -m ao_compensation_model txt2csv

# Convert a specific folder
uv run python -m ao_compensation_model txt2csv --file /path/to/folder
```

### CLI Flags

| Flag | Applies to | Description |
|------|------------|-------------|
| `--file` | `prep`, `validate`, `txt2csv` | `prep`: single CSV to process. `validate`: single test CSV. `txt2csv`: folder path. |
| `--threshold` | `prep` | Amplitude threshold for stationary detection. Omit or pass `auto` for automatic (default: `auto`). |
| `--log-level` | all | Log level (`TRACE`, `DEBUG`, `INFO`, `SUCCESS`, `WARNING`, `ERROR`, `CRITICAL`). |
| `--stderr-level` | all | Stderr log level. |

### As a Library

```python
from ao_compensation_model.training import build_gru_model
from ao_compensation_model.utils import bandpass_filter, align_ao_phase
from ao_compensation_model.validation import validate
```

## Development

0. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) from Astral.
1. `git clone git@github.com:lhharry/ao_compensation_model.git`
2. `make init` — create virtual environment and install dependencies
3. `make format` — format code and run type checks
4. `make test` — run the test suite with coverage
5. `make clean` — delete temporary files and directories

## Publishing

Pushing a version tag triggers automatic publishing to PyPI via GitHub Actions (Trusted Publishing):

```bash
# Update version in pyproject.toml, then:
git tag v0.1.1
git push origin --tags
```

## Structure

<!-- TREE-START -->

```
├── src
│   └── ao_compensation_model
│       ├── __init__.py
│       ├── __main__.py
│       ├── app.py
│       ├── definitions.py
│       ├── gt_dataprep.py
│       ├── training.py
│       ├── txt2csv.py
│       ├── utils.py
│       ├── validation.py
│       ├── dataset/
│       └── model/
├── tests
│   ├── __init__.py
│   ├── conftest.py
│   ├── app_test.py
│   ├── gt_dataprep_test.py
│   ├── training_test.py
│   └── utils_test.py
├── .github/workflows/
├── CONTRIBUTING.md
├── Dockerfile
├── LICENSE
├── Makefile
├── README.md
└── pyproject.toml
```

<!-- TREE-END -->
