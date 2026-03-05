# ao_compensation_model

[![CI](https://github.com/lhharry/ao_compensation_model/actions/workflows/ci.yml/badge.svg)](https://github.com/lhharry/ao_compensation_model/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/ao-compensation-model)](https://pypi.org/project/ao-compensation-model/)
[![Docker](https://github.com/lhharry/ao_compensation_model/actions/workflows/docker.yml/badge.svg)](https://github.com/lhharry/ao_compensation_model/actions/workflows/docker.yml)

A GRU-based compensation model that improves the performance of adaptive oscillators (AOs) during stop-go and go-stop gait transitions. The model learns the phase error between the AO output and the ground-truth gait phase, and applies a real-time correction on edge devices via TFLite.

## Pipeline

1. **Data Preparation** (`prep`) — Bandpass-filters raw IMU hip angles, extracts ground-truth gait phase via Hilbert-like analysis, and computes delta-phi training targets.
2. **Training** (`train`) — Trains a GRU network on sliding windows of AO features, exports to an optimized TFLite model.
3. **Validation** (`validate`) — Runs frame-by-frame TFLite inference on test data and visualises original AO phase vs. enhanced (AO + GRU) phase vs. ground truth.

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

## Development

0. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) from Astral.
1. `git clone git@github.com:lhharry/ao_compensation_model.git`
2. `make init` — create virtual environment and install dependencies
3. `make format` — format code and run type checks
4. `make test` — run the test suite with coverage
5. `make clean` — delete temporary files and directories

## Usage

### As a CLI

```bash
# Prepare ground-truth targets from raw CSVs
uv run python -m ao_compensation_model prep

# Train the GRU model
uv run python -m ao_compensation_model train

# Validate on test data
uv run python -m ao_compensation_model validate
```

### As a library

```python
from ao_compensation_model.training import build_gru_model, compute_sample_weights
from ao_compensation_model.utils import bandpass_filter, extract_true_phase
from ao_compensation_model.validation import validate
```

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
