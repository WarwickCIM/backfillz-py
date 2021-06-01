from dataclasses import dataclass
from math import ceil, floor
from typing import List

import numpy as np
import pandas as pd  # type: ignore
import plotly.graph_objects as go  # type: ignore
from plotly.subplots import make_subplots  # type: ignore
import scipy.stats as stats  # type: ignore

from backfillz.core import Backfillz, HistoryEntry, HistoryEvent, ParameterSlices, Props, Slice
from backfillz.plot import annotate, LeafPlot, Plot, RootPlot, scale, segment, VerticalSubplots
from backfillz.theme import BackfillzTheme


@dataclass
class TracePlot(LeafPlot):
    """Left-hand component."""

    def render(self, fig: go.Figure) -> None:
        for trace in self.traces():
            fig.add_trace(trace, self.row, self.col)
        for box in self.boxes():
            fig.add_trace(box, self.row, self.col)

    # one per chain
    def traces(self) -> List[go.Scatter]:
        return [
            go.Scatter(
                x=chain,
                y=list(range(0, self.data.n_iter)),
                line=dict(color=self.theme.palette[n])
            )
            for n, chain in enumerate(self.data.chains)
        ]

    # one per slice
    def boxes(self) -> List[go.Scatter]:
        return [
            go.Scatter(
                x=[self.data.min_sample] * 2 + [self.data.max_sample] * 2 + [self.data.min_sample],
                y=scale(self.data.n_iter, [slc.lower, slc.upper, slc.upper, slc.lower, slc.lower]),
                mode='lines',
                line=dict(width=2, color=self.theme.fg_colour),
            )
            for slc in self.data.slcs
        ]

    @property
    def xaxis_props(self) -> Props:
        return dict(range=[self.data.min_sample, self.data.max_sample])

    @property
    def yaxis_props(self) -> Props:
        return dict(range=[0, self.data.n_iter])

    def add_title(self, fig: go.Figure) -> None:
        annotate(fig, 16, self.top_left, 'left', 'bottom', None, "Trace Plot With Slices")


@dataclass
class JoiningSegments(LeafPlot):
    """Middle component."""

    def render(self, fig: go.Figure) -> None:
        for seg in self.segments():
            fig.add_trace(seg, self.row, self.col)
        fig.add_trace(self.y_labels(), self.row, self.col)

    # one per slice
    def segments(self) -> List[go.Scatter]:
        return [
            go.Scatter(
                x=[0, 1, 1, 0],
                y=scale(self.data.n_iter, [slc.lower, lower, upper, slc.upper]),
                mode='lines',
                line=dict(color=self.theme.fg_colour, width=1),
                fill='toself',
                fillcolor='rgba(240,240,240,255)'
            )
            for n, slc in enumerate(self.data.slcs, start=1)
            for lower, upper in [((n - 1) / self.data.n_slcs, n / self.data.n_slcs)]
        ]

    # one numerical marker per slice delimiter
    def y_labels(self) -> go.Scatter:
        y = self.slice_delimiters
        return go.Scatter(
            x=[0] * len(y),
            y=y,
            mode='text',
            text=[int(y) for y in y],
            textposition='middle right'
        )

    @property
    def slice_delimiters(self) -> List[float]:
        """Unique slice start/end points, expressed in iterations."""
        delims: List[float] = [*{*[y for slc in self.data.slcs for y in [slc.lower, slc.upper]]}]
        return scale(self.data.n_iter, delims)

    @property
    def xaxis_props(self) -> Props:
        return dict(rangemode='nonnegative', visible=False)

    @property
    def yaxis_props(self) -> Props:
        return dict(
            range=[0, self.data.n_iter],
            tickmode='array',
            tickvals=self.slice_delimiters,
            showticklabels=False
        )


@dataclass
class DensityPlot(LeafPlot):
    """Histogram for a slice (aggregating all chains) plus density plot for each chain."""

    slc: Slice
    n_slc: int

    def render(self, fig: go.Figure) -> None:
        chain_slices: List[np.ndarray] = [
            self.data.chains[
                n,
                floor(self.slc.lower * self.data.n_iter):floor(self.slc.upper * self.data.n_iter)
            ]
            for n in range(0, self.data.n_chains)
        ]

        fig.add_trace(self.histo(chain_slices), self.row, self.col)
        for chain_plot in self.chain_plots(chain_slices):
            fig.add_trace(chain_plot, self.row, self.col)

    def histo(self, chain_slices: List[np.ndarray]) -> go.Histogram:
        return go.Histogram(
            x=[x for xs in chain_slices for x in xs],
            xbins=dict(start=floor(self.data.min_sample), end=ceil(self.data.max_sample), size=1),
            marker=dict(
                color=self.theme.bg_colour,
                line=dict(color=self.theme.fg_colour, width=1)
            ),
            histnorm='probability'
        )

    # non-parametric KDE, smoothed with a Gaussian kernel; one per chain
    def chain_plots(self, chain_slices: List[np.ndarray]) -> List[go.Scatter]:
        x = np.linspace(self.data.min_sample, self.data.max_sample, 200)
        return [
            go.Scatter(
                x=x,
                y=stats.kde.gaussian_kde(chain_slices[n])(x),
                mode='lines',
                line=dict(width=2, color=self.theme.palette[n]),
            )
            for n in range(0, self.data.n_chains)
        ]

    @property
    def xaxis_props(self) -> Props:
        bottom, top = self.n_slc == 0, self.n_slc == self.data.n_slcs - 1
        # single slice requires special treatment; haven't figured out how to mirror tick labels
        if self.data.n_slcs == 1:
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
    """Right-hand component: one density plot per slice."""

    def make_plots(self) -> List[Plot]:
        return [
            DensityPlot(
                axis_ids=[self.axis_ids[n]],
                x_domain=self.x_domain,
                y_domain=segment(self.y_domain, self.data.n_slcs, n),
                data=self.data,
                theme=self.theme,
                slc=slc,
                n_slc=n,
                row=self.row + self.data.n_slcs - 1 - n,
                col=self.col,
            )
            for n, slc in enumerate(self.data.slcs)
        ]

    def add_title(self, fig: go.Figure) -> None:
        # oof -- adjust for x-axis
        annotate(fig, 16, self.top_left, 'left', 'bottom', 0.03, "Density Plots for Slices")


class SliceHistogram(RootPlot):
    """Top-level plot, for a given parameter."""

    data: ParameterSlices
    theme: BackfillzTheme
    left_w = 0.4  # width of trace plot
    middle_w = 0.2  # width of joining segments

    @property
    def plots(self) -> List[Plot]:
        return [self.trace_plot, self.joining_segments, self.density_plots]

    @property
    def trace_plot(self) -> TracePlot:
        return TracePlot(
            axis_ids=[None],
            x_domain=(0, self.left_w),
            y_domain=(0, 1.0),
            row=1,
            col=1,
            data=self.data,
            theme=self.theme,
        )

    @property
    def joining_segments(self) -> JoiningSegments:
        return JoiningSegments(
            axis_ids=[2],
            x_domain=(self.left_w, self.left_w + self.middle_w),
            y_domain=(0, 1.0),
            row=1,
            col=2,
            data=self.data,
            theme=self.theme,
        )

    @property
    def density_plots(self) -> DensityPlots:
        return DensityPlots(
            axis_ids=[n + 3 for n in reversed(range(self.data.n_slcs))],
            x_domain=(self.left_w + self.middle_w, 1),
            y_domain=(0, 1.0),
            row=1,
            col=3,
            data=self.data,
            theme=self.theme,
        )

    def __init__(self, backfillz: Backfillz, slcs: List[Slice], param: str):
        self.theme = backfillz.theme
        self.data = ParameterSlices(
            slcs=slcs,
            param=param,
            chains=backfillz.iter_chains(param),
            max_sample=np.amax(backfillz.mcmc_samples[param]),
            min_sample=np.amin(backfillz.mcmc_samples[param]),
        )

    def layout(self) -> go.Figure:
        n_slcs: int = self.data.n_slcs
        layout: go.Layout = go.Layout(
            title=f"Trace slice histogram of {self.data.param}",
            titlefont=dict(size=30),
            plot_bgcolor=self.theme.bg_colour,
            showlegend=False,
        )
        fig: go.Figure = go.Figure(layout=layout)
        specs: List[List[object]] = \
            [[dict(rowspan=n_slcs), dict(rowspan=n_slcs), dict()]] + \
            [[None, None, dict()] for _ in self.data.slcs[1:]] + \
            [[dict(), None, None] for _ in self.data.chains]

        make_subplots(
            rows=n_slcs + self.data.n_chains,  # density plots + Raftery-Lewis plots
            cols=3,
            figure=fig,
            specs=specs,
            horizontal_spacing=0,
            vertical_spacing=0,
            print_grid=True,
        )

        return fig

    def add_title(self, fig: go.Figure) -> None:
        annotate(
            fig, 14, (1, -0.03), 'right', 'top', None,  # adjust for x-axis
            "Backfillz-py by CIM, University of Warwick and The Alan Turing Institute"
        )

    @staticmethod
    def plot(backfillz: Backfillz, save_plot: bool = False) -> None:
        """Plot a slice histogram."""
        params = pd.Series(backfillz.mcmc_samples.param_names[0:1])  # just first param for now
        slcs: List[Slice] = [Slice(0.028, 0.04), Slice(0.1, 0.2), Slice(0.4, 0.9)]

        for param in params:
            # Assume scalar parameter for now; what about vectors?
            SliceHistogram(backfillz, slcs, param).render()

        backfillz.plot_history.append(HistoryEntry(HistoryEvent.SLICE_HISTOGRAM, save_plot))
