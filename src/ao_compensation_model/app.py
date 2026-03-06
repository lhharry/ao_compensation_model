"""Application entry points for the ao_compensation_model pipeline."""

from pathlib import Path

from loguru import logger

from ao_compensation_model.definitions import DEFAULT_LOG_LEVEL, STATIONARY_THRESHOLD
from ao_compensation_model.utils import setup_logger


def _run_prep(file: str | None, threshold: float) -> None:
    """Prepare ground-truth targets from raw sensor data."""
    from ao_compensation_model.definitions import RAW_DATA_DIR, TRAINING_DATA_DIR
    from ao_compensation_model.gt_dataprep import prepare_targets, visualize

    logger.info(f"Stationary threshold: {threshold}")

    if file is not None:
        csv_path = Path(file)
        if not csv_path.is_absolute():
            csv_path = RAW_DATA_DIR / csv_path
        if not csv_path.suffix:
            csv_path = csv_path.with_suffix(".csv")
        csv_files = [csv_path]
    else:
        csv_files = sorted(RAW_DATA_DIR.glob("*.csv"))

    logger.info("Preparing ground-truth targets from raw data...")
    for csv_file in csv_files:
        out = TRAINING_DATA_DIR / f"{csv_file.stem}_target.csv"
        prepare_targets(csv_file, out, threshold=threshold)
        visualize(out, threshold=threshold)
        logger.info(f"  {csv_file.name} -> {out.name}")
    logger.success("Data preparation complete.")


def _run_train(file: str | None, threshold: float) -> None:
    """Train the GRU model."""
    from ao_compensation_model.training import train

    logger.info("Starting GRU training pipeline...")
    train()
    logger.success("Training complete.")


def _run_validate(file: str | None, threshold: float) -> None:
    """Validate model on test data."""
    from ao_compensation_model.definitions import TEST_DATA_DIR
    from ao_compensation_model.validation import validate

    if file is not None:
        csv_path = Path(file)
        if not csv_path.suffix:
            csv_path = csv_path.with_suffix(".csv")
        csv_files = [csv_path.name if csv_path.is_absolute() else str(csv_path)]
    else:
        csv_files = [f.name for f in sorted(TEST_DATA_DIR.glob("*.csv"))]

    logger.info("Running validation on test data...")
    for name in csv_files:
        logger.info(f"  Validating: {name}")
        validate(name)
    logger.success("Validation complete.")


def _run_txt2csv(file: str | None, threshold: float) -> None:
    """Convert text sensor files to semicolon-delimited CSVs."""
    from ao_compensation_model.txt2csv import convert_folder_to_csv

    if file is not None:
        folder = str(Path(file).resolve())
    else:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        folder = filedialog.askdirectory(title="Select Folder Containing Sensor Files")
        root.destroy()

    if folder:
        logger.info(f"Converting files in: {folder}")
        convert_folder_to_csv(folder)
        logger.success("Conversion complete.")
    else:
        logger.warning("No folder selected. Cancelled.")


_COMMANDS = {
    "prep": _run_prep,
    "train": _run_train,
    "validate": _run_validate,
    "txt2csv": _run_txt2csv,
}


def main(
    command: str = "train",
    file: str | None = None,
    threshold: float = STATIONARY_THRESHOLD,
    log_level: str = DEFAULT_LOG_LEVEL,
    stderr_level: str = DEFAULT_LOG_LEVEL,
) -> None:
    """Run the selected pipeline command.

    :param command: One of 'prep', 'train', 'validate', or 'txt2csv'.
    :param file: Optional single CSV file (name or path) for 'prep'.
    :param threshold: Stationary amplitude threshold for phase extraction.
    :param log_level: The log level to use.
    :param stderr_level: The std err level to use.
    :return: None
    """
    setup_logger(log_level=log_level, stderr_level=stderr_level)

    handler = _COMMANDS.get(command)
    if handler is None:
        logger.error(
            f"Unknown command: '{command}'. Use 'prep', 'train', 'validate', or 'txt2csv'."
        )
        return
    handler(file, threshold)
