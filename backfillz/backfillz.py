from typing import List

import plotly.graph_objects as go  # type: ignore
from stan.fit import Fit  # type: ignore

from backfillz.data import MCMCRun
from backfillz.spiral_stream import SpiralStream
from backfillz.theme import BackfillzTheme, default
from backfillz.trace_dial import TraceDial
from backfillz.trace_slice_histogram import TraceSliceHistogram


class Backfillz:
    """A Backfillz user session."""

    theme: BackfillzTheme
    mcmc_run: MCMCRun
    verbose: bool

    def __init__(self, fit: Fit, verbose: bool = False) -> None:
        """Initialise a Backfillz session."""
        self.mcmc_run = MCMCRun(fit)
        self.verbose = verbose
        self.set_theme(default)

    def set_theme(self, theme: BackfillzTheme) -> None:
        """Set Backfillz theme."""
        if self.verbose:
            print("Setting backfillz object theme to " + theme.name)
        self.theme = theme

    def plot_slice_histogram(self, param: str, save_plot: bool = False) -> go.Figure:
        """Create and plot a slice histogram."""
        return TraceSliceHistogram.fig(self.mcmc_run, self.theme, self.verbose, param)

    def plot_trace_dial(self, param: str, burn_in_iter: int, save_plot: bool = False) -> go.Figure:
        """Create and plot a trace dial."""
        return TraceDial.fig(self.mcmc_run, self.theme, self.verbose, param, burn_in_iter)

    def plot_spiral_stream(self, param: str, steps: List[int], save_plot: bool = False) -> go.Figure:
        """Create and plot a spiral stream."""
        return SpiralStream.fig(self.mcmc_run, self.theme, self.verbose, param, steps)
