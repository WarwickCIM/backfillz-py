from dataclasses import dataclass
from math import ceil, floor
from typing import List

import numpy as np
from plotly.basedatatypes import BaseTraceType  # type: ignore
import plotly.graph_objects as go  # type: ignore
import scipy.stats as stats  # type: ignore

from backfillz.core import Props, Slice
from backfillz.plot import annotate, LeafPlot, Plot, segment, VerticalSubplots


@dataclass
class DensityPlot(LeafPlot):
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
            histnorm='probability'
        )

    # non-parametric KDE, smoothed with a Gaussian kernel; one per chain
    @property
    def chain_plots(self) -> List[go.Scatter]:
        x = np.linspace(self.data.min_sample, self.data.max_sample, 200)
        chain_slices = self.data.chain_slices(self.slc)
        return [
            go.Scatter(
                x=x,
                y=stats.kde.gaussian_kde(chain_slices[n])(x),
                mode='lines',
                line=dict(width=2, color=self.theme.palette[n]),
            )
            for n, _ in enumerate(self.data.chains)
        ]

    @property
    def xaxis_props(self) -> Props:
        bottom, top = self.n_slc == 0, self.n_slc == len(self.data.slcs) - 1
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


class DensityPlots(VerticalSubplots):
    """One density plot per slice."""

    def make_plots(self) -> List[Plot]:
        return [
            DensityPlot(
                axis_ids=[self.axis_ids[n]],
                x_domain=self.x_domain,
                y_domain=segment(self.y_domain, len(self.data.slcs), n),
                data=self.data,
                theme=self.theme,
                slc=slc,
                n_slc=n,
                row=self.row + len(self.data.slcs) - 1 - n,
                col=self.col,
            )
            for n, slc in enumerate(self.data.slcs)
        ]

    def add_title(self, fig: go.Figure) -> None:
        # oof -- adjust for x-axis
        annotate(fig, 16, self.top_left, 'left', 'bottom', 0.03, "Density Plots for Slices")
