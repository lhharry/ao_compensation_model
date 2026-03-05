"""CLI entry point: python -m ao_compensation_model <command>."""

import argparse

from ao_compensation_model.app import main
from ao_compensation_model.definitions import DEFAULT_LOG_LEVEL, LogLevel

if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser("ao_compensation_model pipeline")
    parser.add_argument(
        "command",
        choices=["prep", "train", "validate"],
        help="Pipeline step to run: prep | train | validate",
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
        log_level=args.log_level,
        stderr_level=args.stderr_level,
    )
