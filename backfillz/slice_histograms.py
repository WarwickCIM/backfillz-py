from dataclasses import dataclass
from math import ceil, floor
from typing import cast, List, Tuple

import numpy as np
from plotly.basedatatypes import BaseTraceType  # type: ignore
import plotly.graph_objects as go  # type: ignore
import scipy.stats as stats  # type: ignore

from backfillz.data import ParameterSlices, Props, Slice
from backfillz.plot import LeafPlot


Bins = Tuple[List[float], List[float]]


@dataclass
class SliceHistogram(LeafPlot[ParameterSlices]):
    """Histogram for arbitrary subsets of chains, plus optional KDE plots for individual chains."""

    slc: Slice
    n_slc: int

    @property
    def plot_elements(self) -> List[BaseTraceType]:
        """Histogram for a slice (aggregating subset of chains) plus density plot for each chain."""
        ns: List[int] = [*range(0, len(self.data.chains))]
        return [self.histo(ns, self.theme.fg_colour, 1), *[self.chain_plot(n) for n in ns]]

    # Histogram bins for specified subset of chains.
    def bins(self, ns: List[int], bin_size: float) -> Bins:
        return cast(Bins, np.histogram(
            [x for n in ns for x in self.data.chain_slices(self.slc)[n]],
            [*np.arange(floor(self.data.min_sample), ceil(self.data.max_sample), bin_size)],
            density=True,
        ))

    # Histogram for specified subset of chains. Compute our own bins so we're in full control.
    def histo(self, ns: List[int], color: str, bin_size: float) -> go.Bar:
        ys, xs = self.bins(ns, bin_size)
        return go.Bar(
            x=xs,
            y=ys,
            marker=dict(color=self.theme.bg_colour, line=dict(color=color, width=1)),
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
            yaxis='y' + self.axis_id,
            xaxis='x' + self.axis_id,
        )

    @property
    def xaxis_props(self) -> Props:
        bottom: bool = self.n_slc == 0
        top: bool = self.n_slc == len(self.data.slcs) - 1
        # single slice requires special treatment; haven't figured out how to mirror tick labels
        props: Props
        if len(self.data.slcs) == 1:
            props = dict(mirror='ticks')
        elif bottom:
            props = dict()
        elif top:
            props = dict(side='top')
        else:
            props = dict(visible=False)
        return {**props, **dict(range=(self.data.min_sample, self.data.max_sample))}

    @property
    def yaxis_props(self) -> Props:
        return dict(side='right', rangemode='nonnegative')
