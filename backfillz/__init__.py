"""backfillz namespace."""

from importlib_metadata import PackageNotFoundError, version

from backfillz.backfillz import Backfillz

# Used to automatically set version number from github actions
# as well as not break when being tested locally
try:
    __version__ = version(__package__)  # type: ignore
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"
