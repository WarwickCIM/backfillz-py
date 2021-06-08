from dataclasses import dataclass
from math import ceil, floor
from typing import List

import numpy as np
from plotly.basedatatypes import BaseTraceType  # type: ignore
import plotly.graph_objects as go  # type: ignore
import scipy.stats as stats  # type: ignore

from backfillz.core import Props, Slice
from backfillz.plot import LeafPlot


@dataclass
class SliceHistogram(LeafPlot):
    """Histogram for a slice (aggregating all chains) plus density plot for each chain."""

    slc: Slice
    n_slc: int

    @property
    def plot_elements(self) -> List[BaseTraceType]:
        return [self.histo] + self.chain_plots

    @property
    def histo(self) -> go.Histogram:
        return go.Histogram(
            x=[x for xs in self.data.chain_slices(self.slc) for x in xs],
            xbins=dict(start=floor(self.data.min_sample), end=ceil(self.data.max_sample), size=1),
            marker=dict(
                color=self.theme.bg_colour,
                line=dict(color=self.theme.fg_colour, width=1)
            ),
            histnorm='probability',
        )

    @property
    def chain_plots(self) -> List[go.Scatter]:
        return [self.chain_plot(n) for n, _ in enumerate(self.data.chains)]

    # Non-parametric KDE, smoothed with a Gaussian kernel, for a given chain.
    def chain_plot(self, n: int) -> go.Scatter:
        x = np.linspace(self.data.min_sample, self.data.max_sample, 200)
        chain_slices = self.data.chain_slices(self.slc)
        return go.Scatter(
            x=x,
            y=stats.kde.gaussian_kde(chain_slices[n])(x),
            mode='lines',
            line=dict(width=2, color=self.theme.palette[n]),
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
