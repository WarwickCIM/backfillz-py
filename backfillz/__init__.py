"""backfillz-py namespace."""

from importlib_metadata import PackageNotFoundError, version
from backfillz.trace_slice_histogram import plot as plot_slice_histogram

__author__ = "Roly Perera"
__email__ = "rperera@turing.ac.uk"

# Used to automatically set version number from github actions
# as well as not break when being tested locally
try:
    __version__ = version(__package__)  # type: ignore
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"
