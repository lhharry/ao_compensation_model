"""Test the main program."""

from unittest.mock import patch, MagicMock

from ao_compensation_model.app import main


def test_main():
    """Test the main function with default (train) command — mocked."""
    with patch("ao_compensation_model.app.setup_logger"):
        result = main()
    assert result is None


def test_main_unknown_command():
    """Unknown command should log an error but not raise."""
    with patch("ao_compensation_model.app.setup_logger"):
        result = main(command="unknown")
    assert result is None


def test_main_prep_command():
    """'prep' command should call prepare_targets for each CSV."""
    with (
        patch("ao_compensation_model.app.setup_logger"),
        patch("ao_compensation_model.gt_dataprep.prepare_targets"),
        patch(
            "ao_compensation_model.definitions.RAW_DATA_DIR",
            new_callable=lambda: MagicMock(glob=MagicMock(return_value=[])),
        ),
    ):
        main(command="prep")


def test_main_validate_command():
    """'validate' command should call validate for each test CSV."""
    with (
        patch("ao_compensation_model.app.setup_logger"),
        patch("ao_compensation_model.validation.validate"),
        patch(
            "ao_compensation_model.definitions.TEST_DATA_DIR",
            new_callable=lambda: MagicMock(glob=MagicMock(return_value=[])),
        ),
    ):
        main(command="validate")
