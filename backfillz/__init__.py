"""backfillz namespace."""

from importlib_metadata import PackageNotFoundError, version

from backfillz.core import Backfillz
from backfillz.trace_dial import plot as plot_trace_dial
from backfillz.trace_slice_histogram import plot as plot_slice_histogram

# Used to automatically set version number from github actions
# as well as not break when being tested locally
try:
    __version__ = version(__package__)  # type: ignore
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"
