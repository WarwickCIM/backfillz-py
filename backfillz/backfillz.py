from datetime import datetime
from enum import Enum
import re
import sys
from typing import List

from stan.fit import Fit  # type: ignore

from backfillz.data import MCMCRun
from backfillz.plot import default_config
from backfillz.spiral_stream import SpiralStream
from backfillz.theme import BackfillzTheme, default
from backfillz.trace_dial import TraceDial
from backfillz.trace_slice_histogram import TraceSliceHistogram


class HistoryEvent(Enum):
    """Category of event."""

    OBJECT_CREATION = 1
    SLICE_HISTOGRAM = 2
    TRACE_DIAL = 3
    SPIRAL_STREAM = 4


class HistoryEntry:
    """An entry in the Backfillz history log."""

    count = 0

    ident: int
    date: datetime
    event: HistoryEvent
    python_version: str
    saved: bool

    def __init__(self, event: HistoryEvent, saved: bool) -> None:
        """Construct a history entry."""
        self.ident = HistoryEntry.count
        HistoryEntry.count += 1
        self.date = datetime.now()
        self.event = event
        self.python_version = sys.version
        self.saved = saved


class Backfillz:
    """Represents a Backfillz user session."""

    theme: BackfillzTheme
    mcmc_run: MCMCRun
    verbose: bool
    plot_history: List[HistoryEntry]

    def __init__(self, fit: Fit, verbose: bool = False) -> None:
        """Initialise a Backfillz session."""
        self.mcmc_run = MCMCRun(fit)
        self.verbose = verbose
        self.set_theme(default)
        self.plot_history = [
            HistoryEntry(HistoryEvent.OBJECT_CREATION, False)
        ]

    def set_theme(self, theme: BackfillzTheme) -> None:
        """Set Backfillz theme."""
        if self.verbose:
            print("Setting backfillz object theme to " + theme.name)
        self.theme = theme

    def plot_slice_histogram(self, param: str, save_plot: bool = False) -> None:
        """Create and plot a slice histogram."""
        fig = TraceSliceHistogram.fig(self.mcmc_run, self.theme, self.verbose, param)
        self.plot_history.append(HistoryEntry(HistoryEvent.SLICE_HISTOGRAM, save_plot))
        fig.show(config=default_config())

    def plot_trace_dial(self, param: str, save_plot: bool = False) -> None:
        """Create and plot a trace dial."""
        fig = TraceDial.fig(self.mcmc_run, self.theme, self.verbose, param)
        self.plot_history.append(HistoryEntry(HistoryEvent.TRACE_DIAL, save_plot))
        fig.show(config=default_config())

    def plot_spiral_stream(self, param: str, save_plot: bool = False) -> None:
        """Create and plot a spiral stream."""
        fig = SpiralStream.fig(self.mcmc_run, self.theme, self.verbose, param)
        self.plot_history.append(HistoryEntry(HistoryEvent.SPIRAL_STREAM, save_plot))
        fig.show(config=default_config())
        found = fig.to_image(format="png")
        filename: str = "tests/expected_spiral_stream"
        try:
            file = open(filename + ".png", "rb")
            expected = file.read()
            if expected != found:
                file_new = open(filename + ".new.png", "wb")
                file_new.write(found)
                assert False
        except FileNotFoundError:
            file_new = open(filename + ".png", "wb")
            file_new.write(found)
