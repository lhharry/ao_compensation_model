"""Test the main program."""

from ao_compensation_model.app import main


def test_main():
    """Test the main function."""
    assert main() is None
