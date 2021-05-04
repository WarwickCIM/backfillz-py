from dataclasses import dataclass
from math import ceil, floor
from typing import Any, Dict, List

import numpy as np
import pandas as pd  # type: ignore
import plotly.graph_objects as go  # type: ignore
from plotly.subplots import make_subplots  # type: ignore
import scipy.stats as stats  # type: ignore

from backfillz.core import Backfillz, HistoryEntry, HistoryEvent
from backfillz.theme import BackfillzTheme


@dataclass
class Slice:
    """A slice of an MCMC trace."""

    lower: float
    upper: float


Param = str
Slices = Dict[Param, List[Slice]]


@dataclass
class ChartData:
    """The MCMC data being presented."""

    theme: BackfillzTheme
    slcs: List[Slice]
    param: str
    chains: np.ndarray
    max_sample: float
    min_sample: float

    @property
    def n_chains(self) -> int:
        """Return number of chains."""
        return int(self.chains.shape[0])

    @property
    def n_iter(self) -> int:
        """Return number of MCMC iterations per chain."""
        return int(self.chains.shape[1])


@dataclass
class TracePlot:
    """Left-hand component."""

    traces: List[go.Scatter]    # one per chain
    boxes: List[go.Scatter]     # one per slice

    def __init__(self, chart: ChartData):
        """Make a trace plot."""
        self.traces = [
            go.Scatter(
                x=chart.chains[n],
                y=list(range(0, chart.chains[n].size)),
                line=dict(color=chart.theme.palette[n])
            )
            for n in range(0, chart.n_chains)
        ]
        self.boxes = [
            go.Scatter(
                x=[chart.min_sample] * 2 + [chart.max_sample] * 2 + [chart.min_sample],
                y=_scale(chart.n_iter, [slc.lower, slc.upper, slc.upper, slc.lower, slc.lower]),
                mode='lines',
                line=dict(width=2, color=chart.theme.fg_colour),
            )
            for slc in chart.slcs
        ]

    def render(self, fig: go.Figure) -> None:
        """Render a trace plot into fig."""
        for trace in self.traces:
            fig.add_trace(trace, row=1, col=1)
        for box in self.boxes:
            fig.add_trace(box, row=1, col=1)


@dataclass
class JoiningSegments:
    """Middle component."""

    segments: List[go.Scatter]  # one per slice
    y_labels: go.Scatter  # one point per unique slice start/end point

    def __init__(self, chart: ChartData):
        """Make a joining segment."""
        width: int = 30  # check against R version
        y_scale: int = chart.n_iter
        self.segments = [
            go.Scatter(
                x=_scale(width, [0, 1, 1, 0]),
                y=_scale(y_scale, [slc.lower, lower, upper, slc.upper]),
                mode='lines',
                line=dict(color=chart.theme.fg_colour, width=1),
                fill='toself',
                fillcolor='rgba(240,240,240,255)'
            )
            for n_slc, slc in enumerate(chart.slcs, start=1)
            for lower, upper in [((n_slc - 1) / len(chart.slcs), n_slc / len(chart.slcs))]
        ]
        y = _scale(chart.n_iter, [*{*[y for slc in chart.slcs for y in [slc.lower, slc.upper]]}])
        self.y_labels = go.Scatter(
            x=[0] * len(y),
            y=y,
            mode='text',
            text=[int(y) for y in y],
            textposition='middle right'
        )

    def render(self, fig: go.Figure, row: int, col: int) -> None:
        """Render the joining segments into fig at row and column."""
        for segment in self.segments:
            fig.add_trace(segment, row, col)
        fig.add_trace(self.y_labels, row, col)


@dataclass
class DensityPlot:
    """Histogram for a slice (aggregating all chains) plus density plot for each chain."""

    histo: go.Histogram
    chain_plots: List[go.Scatter]  # one per chain

    def __init__(self, chart: ChartData, slc: Slice):
        """Make an instance."""
        chain_slices: List[np.ndarray] = [
            chart.chains[n, floor(slc.lower * chart.n_iter):floor(slc.upper * chart.n_iter)]
            for n in range(0, chart.n_chains)
        ]
        self.histo = go.Histogram(
            x=[x for xs in chain_slices for x in xs],
            xbins=dict(start=floor(chart.min_sample), end=ceil(chart.max_sample), size=1),
            marker=dict(
                color=chart.theme.bg_colour,
                line=dict(color=chart.theme.fg_colour, width=1)
            ),
            histnorm='probability'
        )
        # non-parametric KDE, smoothed with a Gaussian kernel
        x = np.linspace(chart.min_sample, chart.max_sample, 200)
        self.chain_plots = [
            go.Scatter(
                x=x,
                y=stats.kde.gaussian_kde(chain_slices[n])(x),
                mode='lines',
                line=dict(width=2, color=chart.theme.palette[n]),
            )
            for n in range(0, chart.n_chains)
        ]

    def render(self, fig: go.Figure, row: int, col: int) -> None:
        """Render density plot into fig at row and column."""
        fig.add_trace(self.histo, row, col)
        for chain_plot in self.chain_plots:
            fig.add_trace(chain_plot, row, col)


@dataclass
class DensityPlots:
    """Right-hand component: one density plot per slice."""

    density_plots: List[DensityPlot]

    def __init__(self, chart: ChartData):
        """Make an instance."""
        self.density_plots = [DensityPlot(chart, slc) for slc in chart.slcs[::-1]]

    def render(self, fig: go.Figure) -> None:
        """Render density plots into fig."""
        for n_slice, density_plot in enumerate(self.density_plots):
            density_plot.render(fig, row=n_slice + 1, col=3)


@dataclass
class RafteryLewisPlots:
    """Bottom component: one Raftery-Lewis plot per chain."""
    plots: List[go.Scatter]

    def __init__(self, chart: ChartData):
        """Make an instance."""
        # raftery_lewis(x, q=0.025, r=0.005)  # same as defaults used in R version


class SliceHistogram:
    """Top-level plot, for a given parameter."""

    backfillz: Backfillz
    chart: ChartData

    def __init__(self, backfillz: Backfillz, slcs: List[Slice], param: str):
        """Construct a Slice Histogram for a given parameter from a list of slices."""
        self.backfillz = backfillz
        chains: np.ndarray = backfillz.iter_chains(param)
        self.chart = ChartData(
            theme=backfillz.theme,
            slcs=slcs,
            param=param,
            chains=chains,
            max_sample=np.amax(backfillz.mcmc_samples[param]),
            min_sample=np.amin(backfillz.mcmc_samples[param]),
        )

    @property
    def figure(self) -> go.Figure:
        """Derive Plotly figure from 3 parts."""
        fig: go.Figure = self._layout()
        self.render(fig)
        return fig

    def _layout(self) -> go.Figure:
        n_slcs: int = len(self.chart.slcs)
        layout: go.Layout = go.Layout(
            title=f"Trace slice histogram of {self.chart.param}",
            titlefont=dict(size=32),
            plot_bgcolor=self.chart.theme.bg_colour,
            showlegend=False,
            xaxis=dict(range=[self.chart.min_sample, self.chart.max_sample]),
            yaxis=dict(range=[0, self.chart.n_iter]),
            xaxis2=dict(visible=False),
            yaxis2=dict(range=[0, self.chart.n_iter]),
        )
        fig: go.Figure = go.Figure(layout=layout)
        specs: List[List[object]] = \
            [[dict(rowspan=n_slcs), dict(rowspan=n_slcs), dict()]] + \
            [[None, None, dict()] for _ in self.chart.slcs[1:]]

        # Need a structured way to configure subplot titles
        make_subplots(
            rows=n_slcs,
            cols=3,
            figure=fig,
            specs=specs,
            horizontal_spacing=0,
            vertical_spacing=0,
            shared_xaxes=True,
            print_grid=True,
            subplot_titles=["Trace Plot with Slices", None, "Density Plots for Slices"]
        )

        # TODO: magic number 3 occurs twice here
        for n_slc, _ in enumerate(self.chart.slcs):
            yaxis = 'yaxis' + str(3 + n_slc)
            fig.layout[yaxis].update(side='right', rangemode='nonnegative')
        fig.layout['xaxis3'].update(mirror='allticks', side='top', showticklabels=True)

        axis_settings: Dict[str, Any] = dict(
            showgrid=False,
            zeroline=False,
            linecolor=self.chart.theme.fg_colour,
            ticks='outside',
            tickwidth=1,
            ticklen=5,
            tickcolor=self.chart.theme.fg_colour,
            fixedrange=True,  # disable selection zoom
        )

        fig.update_xaxes(**axis_settings)
        fig.update_yaxes(**axis_settings)

        fig.layout['xaxis2'].update(rangemode='nonnegative')
        fig.layout['yaxis2'].update(
            tickmode='array',
            tickvals=_scale(
                self.chart.n_iter,
                [*{*[y for slc in self.chart.slcs for y in [slc.lower, slc.upper]]}]
            ),
            showticklabels=False  # JoiningSegments will take care of these
        )

        fig.layout.annotations[1].update(y=1.03)  # oof

        # TODO: eliminate magic indices 0, 1 and magic use of xaxis3
        fig.layout.annotations[0].update(xanchor='left', x=fig.layout.xaxis.domain[0])
        fig.layout.annotations[1].update(xanchor='left', x=fig.layout.xaxis3.domain[0])
        return fig

    def render(self, fig: go.Figure) -> None:
        """Render the plot into fig."""
        TracePlot(self.chart).render(fig)
        JoiningSegments(self.chart).render(fig, 1, 2)
        DensityPlots(self.chart).render(fig)
        RafteryLewisPlots(self.chart).render(fig)


def plot_slice_histogram(backfillz: Backfillz, save_plot: bool = False) -> None:
    """Plot a slice histogram."""
    params = pd.Series(backfillz.mcmc_samples.param_names[0:1])  # just first param for now
    slice_list: List[Slice] = [
        Slice(0.028, 0.04), Slice(0.1, 0.2), Slice(0.4, 0.9)
    ]
    slices: Slices = {param: slice_list for param in params}

    config = dict(displayModeBar=False, showAxisDragHandles=False)
    for param in params:
        # Assume scalar parameter for now; what about vectors?
        SliceHistogram(backfillz, slices[param], param).figure.show(config=config)

    # Update log
    backfillz.plot_history.append(HistoryEntry(HistoryEvent.SLICE_HISTOGRAM, save_plot))


def _scale(factor: float, xs: List[float]) -> List[float]:
    return [x * factor for x in xs]
