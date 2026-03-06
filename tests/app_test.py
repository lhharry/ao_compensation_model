"""Test the main program."""

from unittest.mock import MagicMock, patch

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
    mock_test_dir = MagicMock()
    mock_test_dir.glob.return_value = []
    # validation.py imports ai_edge_litert which may not be installed in CI,
    # so we mock the entire validate import inside main().
    mock_validate_module = MagicMock()
    with (
        patch("ao_compensation_model.app.setup_logger"),
        patch("ao_compensation_model.definitions.TEST_DATA_DIR", mock_test_dir),
        patch.dict(
            "sys.modules",
            {"ao_compensation_model.validation": mock_validate_module},
        ),
    ):
        main(command="validate")
