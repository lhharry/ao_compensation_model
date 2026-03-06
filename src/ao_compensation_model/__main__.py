"""CLI entry point: python -m ao_compensation_model <command>."""

import argparse

from ao_compensation_model.app import main
from ao_compensation_model.definitions import DEFAULT_LOG_LEVEL, LogLevel, STATIONARY_THRESHOLD

if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser("ao_compensation_model pipeline")
    parser.add_argument(
        "command",
        choices=["prep", "train", "validate", "txt2csv"],
        help="Pipeline step to run: prep | train | validate | txt2csv",
    )
    parser.add_argument(
        "--file",
        default=None,
        help="Process a single CSV file (name or path) instead of all files.",
        required=False,
        type=str,
    )
    parser.add_argument(
        "--threshold",
        default=STATIONARY_THRESHOLD,
        help=f"Stationary amplitude threshold (default: {STATIONARY_THRESHOLD}).",
        required=False,
        type=float,
    )
    parser.add_argument(
        "--log-level",
        default=DEFAULT_LOG_LEVEL,
        choices=list(LogLevel()),
        help="Set the log level.",
        required=False,
        type=str,
    )
    parser.add_argument(
        "--stderr-level",
        default=DEFAULT_LOG_LEVEL,
        choices=list(LogLevel()),
        help="Set the stderr level.",
        required=False,
        type=str,
    )
    args = parser.parse_args()

    main(
        command=args.command,
        file=args.file,
        threshold=args.threshold,
        log_level=args.log_level,
        stderr_level=args.stderr_level,
    )
