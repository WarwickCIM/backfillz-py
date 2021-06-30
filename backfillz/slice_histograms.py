from dataclasses import dataclass
from math import ceil, floor
from typing import List

import numpy as np
from plotly.basedatatypes import BaseTraceType  # type: ignore
import plotly.graph_objects as go  # type: ignore
import scipy.stats as stats  # type: ignore

from backfillz.data import Props, Slice
from backfillz.plot import LeafPlot


@dataclass
class SliceHistogram(LeafPlot):
    """Plot histograms for arbitrary subsets of chains, plus optional KDE plots for individual chains."""

    slc: Slice
    n_slc: int

    @property
    def plot_elements(self) -> List[BaseTraceType]:
        """Histogram for a slice (aggregating all chains) plus density plot for each chain."""
        ns: List[int] = [n for n, _ in enumerate(self.data.chains)]
        return [self.histo(ns, self.theme.fg_colour, 1)] + [self.chain_plot(n) for n in ns]

    # Histogram for any subset of the chains.
    def histo(self, ns: List[int], color: str, bin_size: float) -> go.Histogram:
        chain_slices: List[np.ndarray] = self.data.chain_slices(self.slc)
        return go.Histogram(
            x=[x for n in ns for x in chain_slices[n]],
            xbins=dict(start=floor(self.data.min_sample), end=ceil(self.data.max_sample), size=bin_size),
            marker=dict(
                color=self.theme.bg_colour,
                line=dict(color=color, width=1)
            ),
            histnorm='probability',
            xaxis='x' + self.axis_id,
            yaxis='y' + self.axis_id,
        )

    # Non-parametric KDE, smoothed with a Gaussian kernel, for a given chain.
    def chain_plot(self, n: int) -> go.Scatter:
        x = np.linspace(self.data.min_sample, self.data.max_sample, 200)
        chain_slices = self.data.chain_slices(self.slc)
        return go.Scatter(
            x=x,
            y=stats.kde.gaussian_kde(chain_slices[n])(x),
            mode='lines',
            line=dict(width=2, color=self.theme.palette[n]),
            xaxis='x' + self.axis_id,
            yaxis='y' + self.axis_id,
        )

    @property
    def xaxis_props(self) -> Props:
        bottom: bool = self.n_slc == 0
        top: bool = self.n_slc == len(self.data.slcs) - 1
        # single slice requires special treatment; haven't figured out how to mirror tick labels
        if len(self.data.slcs) == 1:
            return dict(mirror='ticks')
        elif bottom:
            return dict()
        elif top:
            return dict(side='top')
        else:
            return dict(visible=False)

    @property
    def yaxis_props(self) -> Props:
        return dict(side='right', rangemode='nonnegative')
