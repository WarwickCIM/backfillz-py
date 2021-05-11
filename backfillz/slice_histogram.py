from dataclasses import dataclass
from math import ceil, floor
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd  # type: ignore
import plotly.graph_objects as go  # type: ignore
from plotly.subplots import make_subplots  # type: ignore
from rpy2.robjects import numpy2ri  # type: ignore
from rpy2.robjects.packages import importr  # type: ignore
import scipy.stats as stats  # type: ignore

from backfillz.core import Backfillz, HistoryEntry, HistoryEvent
from backfillz.theme import BackfillzTheme

coda = importr("coda")  # use R for raftery.diag; might be a better diagnostic in PyMC3
numpy2ri.activate()


@dataclass
class Slice:
    """A slice of an MCMC trace."""

    lower: float
    upper: float


Param = str
Slices = Dict[Param, List[Slice]]

# ids assigned as axis suffices by Plotly; omitted for first subplot
AxisIds = Tuple[Optional[int], Optional[int]]
Props = Dict[str, Any]


def increment_axes(axis_ids: AxisIds, n: int) -> AxisIds:
    assert isinstance(axis_ids[0], int)
    assert isinstance(axis_ids[1], int)
    return axis_ids[0] + n, int(axis_ids[1]) + n


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
        return int(self.chains.shape[0])

    @property
    def n_iter(self) -> int:
        """Return number of MCMC iterations per chain."""
        return int(self.chains.shape[1])


@dataclass
class Subplot:
    """A Plotly subplot and its assigned axis ids."""

    chart: ChartData
    axis_ids: AxisIds

    def layout_axes(self, fig: go.Figure) -> None:
        """Configure my x and y axis settings in fig."""
        fig.layout[self.xaxis_id].update(**self.xaxis_props)
        fig.layout[self.yaxis_id].update(**self.yaxis_props)

    @property
    def xaxis_id(self) -> str:
        """My Plotly-assigned x-axis id."""
        return 'xaxis' + ('' if self.axis_ids[0] is None else str(self.axis_ids[0]))

    @property
    def yaxis_id(self) -> str:
        """My Plotly-assigned y-axis id."""
        return 'yaxis' + ('' if self.axis_ids[1] is None else str(self.axis_ids[1]))

    @property
    def xaxis_props(self) -> Props:
        """My custom x-axis settings; subclasses can override."""
        return dict()

    @property
    def yaxis_props(self) -> Props:
        """My custom y-axis settings; subclasses can override."""
        return dict()

    def render(self, fig: go.Figure, row: int, col: int) -> None:
        """Render me into fig at supplied row and column."""
        pass


@dataclass
class Subplots:
    """A collection of vertically arranged subplots."""

    axis_ids: AxisIds  # x_axis + first y_axis
    plots: List[Subplot]

    def layout_axes(self, fig: go.Figure) -> None:
        """Ask each subplot to configure its axes."""
        for plot in self.plots:
            plot.layout_axes(fig)

    @property
    def xaxis_id(self) -> str:
        """Plotly-assigned x-axis id for my first (uppermost) subplot."""
        return 'xaxis' + ('' if self.axis_ids[0] is None else str(self.axis_ids[0]))

    def render(self, fig: go.Figure, row: int, col: int) -> None:
        """Render my subplots into fig, placing subplots into descending rows."""
        for n, plot in enumerate(self.plots):
            plot.render(fig, row=row + n, col=col)


@dataclass
class TracePlot(Subplot):
    """Left-hand component."""

    traces: List[go.Scatter]    # one per chain
    boxes: List[go.Scatter]     # one per slice

    def __init__(self, chart: ChartData, axis_ids: AxisIds):
        super().__init__(chart, axis_ids)
        self.traces = [
            go.Scatter(
                x=chain,
                y=list(range(0, chart.n_iter)),
                line=dict(color=chart.theme.palette[n])
            )
            for n, chain in enumerate(chart.chains)
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

    @property
    def xaxis_props(self) -> Props:
        return dict(range=[self.chart.min_sample, self.chart.max_sample])

    @property
    def yaxis_props(self) -> Props:
        return dict(range=[0, self.chart.n_iter])

    def render(self, fig: go.Figure, row: int, col: int) -> None:
        """Render a trace plot into fig."""
        for trace in self.traces:
            fig.add_trace(trace, row, col)
        for box in self.boxes:
            fig.add_trace(box, row, col)


@dataclass
class JoiningSegments(Subplot):
    """Middle component."""

    width: int = 30  # check against R version

    # one per slice
    def segments(self) -> List[go.Scatter]:
        return [
            go.Scatter(
                x=_scale(JoiningSegments.width, [0, 1, 1, 0]),
                y=_scale(self.chart.n_iter, [slc.lower, lower, upper, slc.upper]),
                mode='lines',
                line=dict(color=self.chart.theme.fg_colour, width=1),
                fill='toself',
                fillcolor='rgba(240,240,240,255)'
            )
            for n_slc, slc in enumerate(self.chart.slcs, start=1)
            for lower, upper in [((n_slc - 1) / len(self.chart.slcs), n_slc / len(self.chart.slcs))]
        ]

    # one point per unique slice start/end point
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
        delims: List[float] = [*{*[y for slc in self.chart.slcs for y in [slc.lower, slc.upper]]}]
        return _scale(self.chart.n_iter, delims)

    @property
    def xaxis_props(self) -> Props:
        return dict(rangemode='nonnegative', visible=False)

    @property
    def yaxis_props(self) -> Props:
        return dict(
            range=[0, self.chart.n_iter],
            tickmode='array',
            tickvals=_scale(
                self.chart.n_iter,
                [*{*[y for slc in self.chart.slcs for y in [slc.lower, slc.upper]]}]
            ),
            showticklabels=False
        )

    def render(self, fig: go.Figure, row: int, col: int) -> None:
        for segment in self.segments():
            fig.add_trace(segment, row, col)
        fig.add_trace(self.y_labels(), row, col)


@dataclass
class DensityPlot(Subplot):
    """Histogram for a slice (aggregating all chains) plus density plot for each chain."""

    slc: Slice

    @property
    def xaxis_props(self) -> Props:
        return dict(mirror='allticks', side='top', showticklabels=True)

    @property
    def yaxis_props(self) -> Props:
        return dict(side='right', rangemode='nonnegative')

    def histo(self, chain_slices: List[np.ndarray]) -> go.Histogram:
        return go.Histogram(
            x=[x for xs in chain_slices for x in xs],
            xbins=dict(start=floor(self.chart.min_sample), end=ceil(self.chart.max_sample), size=1),
            marker=dict(
                color=self.chart.theme.bg_colour,
                line=dict(color=self.chart.theme.fg_colour, width=1)
            ),
            histnorm='probability'
        )

    # non-parametric KDE, smoothed with a Gaussian kernel; one per chain
    def chain_plots(self, chain_slices: List[np.ndarray]) -> List[go.Scatter]:
        x = np.linspace(self.chart.min_sample, self.chart.max_sample, 200)
        return [
            go.Scatter(
                x=x,
                y=stats.kde.gaussian_kde(chain_slices[n])(x),
                mode='lines',
                line=dict(width=2, color=self.chart.theme.palette[n]),
            )
            for n in range(0, self.chart.n_chains)
        ]

    def render(self, fig: go.Figure, row: int, col: int) -> None:
        chain_slices: List[np.ndarray] = [
            self.chart.chains[
                n,
                floor(self.slc.lower * self.chart.n_iter):floor(self.slc.upper * self.chart.n_iter)
            ]
            for n in range(0, self.chart.n_chains)
        ]

        fig.add_trace(self.histo(chain_slices), row, col)
        for chain_plot in self.chain_plots(chain_slices):
            fig.add_trace(chain_plot, row, col)


@dataclass
class DensityPlots(Subplots):
    """Right-hand component: one density plot per slice."""

    def __init__(self, chart: ChartData, axis_ids: AxisIds):
        """Make an instance."""
        super().__init__(axis_ids, [
            DensityPlot(chart, increment_axes(axis_ids, n), slc)
            for n, slc in enumerate(chart.slcs[::-1])
        ])


@dataclass
class RafteryLewisPlot(Subplot):
    """Beneath left-hand component: one Raftery-Lewis plot per chain."""

    n_chain: int

    @property
    def xaxis_props(self) -> Props:
        return dict(visible=False)

    @property
    def yaxis_props(self) -> Props:
        return dict(visible=False)

    def plot(self) -> go.Scatter:
        return go.Scatter(
            x=list(range(0, self.chart.n_iter)),
            y=self.chart.chains[self.n_chain],
            line=dict(color=self.chart.theme.palette[self.n_chain])
        )

    def render(self, fig: go.Figure, row: int, col: int) -> None:
        fig.add_trace(self.plot(), row, col)


@dataclass
class RafteryLewisPlots(Subplots):
    """Bottom component: one Raftery-Lewis plot per chain."""

    def __init__(self, chart: ChartData, axis_ids: AxisIds):
        super().__init__(axis_ids, [
            RafteryLewisPlot(chart, increment_axes(axis_ids, n), n)
            for n, _ in enumerate(chart.chains)
        ])

    def _required_sample_size(self, chain: np.ndarray) -> float:
        """Return N component of resmatrix component of result of raftery.diag R function."""
        result = coda.raftery_diag(chain, q=0.025, r=0.005)  # same as defaults used in R version
        resmatrix = result[1][0]
        return float(resmatrix[1])


class SliceHistogram:
    """Top-level plot, for a given parameter."""

    backfillz: Backfillz
    chart: ChartData
    tracePlot: TracePlot
    rafteryLewisPlots: RafteryLewisPlots
    joiningSegments: JoiningSegments
    densityPlots: DensityPlots

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
        self.tracePlot = TracePlot(self.chart, (None, None))
        self.rafteryLewisPlots = RafteryLewisPlots(self.chart, (6, 6))
        self.joiningSegments = JoiningSegments(self.chart, (2, 2))
        self.densityPlots = DensityPlots(self.chart, (3, 3))

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
        )
        fig: go.Figure = go.Figure(layout=layout)
        specs: List[List[object]] = \
            [[dict(rowspan=n_slcs), dict(rowspan=n_slcs), dict()]] + \
            [[None, None, dict()] for _ in self.chart.slcs[1:]] + \
            [[dict(), None, None] for _ in self.chart.chains]

        make_subplots(
            rows=n_slcs + self.chart.n_chains,  # density plots + Raftery-Lewis plots
            cols=3,
            figure=fig,
            specs=specs,
            horizontal_spacing=0,
            vertical_spacing=0,
            print_grid=True,
            subplot_titles=["Trace Plot with Slices", None, "Density Plots for Slices"]
        )

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

        for plot in [
            self.tracePlot,
            self.densityPlots,
            self.joiningSegments,
            self.rafteryLewisPlots
        ]:
            plot.layout_axes(fig)

        # TODO: eliminate magic indices 0, 1
        annotations = fig.layout.annotations
        annotations[0].update(xanchor='left', x=fig.layout[self.tracePlot.xaxis_id].domain[0])
        annotations[1].update(y=1.03)  # oof -- adjust title subgraph
        annotations[1].update(xanchor='left', x=fig.layout[self.densityPlots.xaxis_id].domain[0])

        return fig

    def render(self, fig: go.Figure) -> None:
        """Render plot into fig."""
        self.tracePlot.render(fig, 1, 1)
        self.rafteryLewisPlots.render(fig, len(self.chart.slcs) + 1, 1)
        self.joiningSegments.render(fig, 1, 2)
        self.densityPlots.render(fig, 1, 3)


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
