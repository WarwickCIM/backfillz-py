"""Test module for backfillz."""

from backfillz import __author__, __email__, __version__


def test_project_info():
    """Test __author__ value."""
    assert __author__ == "Roly Perera"
    assert __email__ == "rperera@turing.ac.uk"
    assert __version__ == "0.0.0"
