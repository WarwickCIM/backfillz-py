from dataclasses import dataclass
from math import ceil, floor
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd  # type: ignore
import plotly.graph_objects as go  # type: ignore
from plotly.subplots import make_subplots  # type: ignore
from rpy2.robjects import numpy2ri  # type: ignore
from rpy2.robjects.packages import importr  # type: ignore
import scipy.stats as stats  # type: ignore

from backfillz.core import Backfillz, HistoryEntry, HistoryEvent
from backfillz.plot \
    import _scale, AxisIds, ChartData, increment_axes, Props, segment, Slice, Slices, Subplot, Subplots

coda = importr("coda")  # use R for raftery.diag; might be a better diagnostic in PyMC3
numpy2ri.activate()


@dataclass
class TracePlot(Subplot):
    """Left-hand component."""

    # one per chain
    def traces(self) -> List[go.Scatter]:
        return [
            go.Scatter(
                x=chain,
                y=list(range(0, self.data.n_iter)),
                line=dict(color=self.data.theme.palette[n])
            )
            for n, chain in enumerate(self.data.chains)
        ]

    # one per slice
    def boxes(self) -> List[go.Scatter]:
        return [
            go.Scatter(
                x=[self.data.min_sample] * 2 + [self.data.max_sample] * 2 + [self.data.min_sample],
                y=_scale(self.data.n_iter, [slc.lower, slc.upper, slc.upper, slc.lower, slc.lower]),
                mode='lines',
                line=dict(width=2, color=self.data.theme.fg_colour),
            )
            for slc in self.data.slcs
        ]

    @property
    def xaxis_props(self) -> Props:
        return dict(range=[self.data.min_sample, self.data.max_sample])

    @property
    def yaxis_props(self) -> Props:
        return dict(domain=self.y_domain, range=[0, self.data.n_iter])

    def render(self, fig: go.Figure, row: int, col: int) -> None:
        for trace in self.traces():
            fig.add_trace(trace, row, col)
        for box in self.boxes():
            fig.add_trace(box, row, col)


@dataclass
class JoiningSegments(Subplot):
    """Middle component."""

    # one per slice
    def segments(self) -> List[go.Scatter]:
        return [
            go.Scatter(
                x=[0, 1, 1, 0],
                y=_scale(self.data.n_iter, [slc.lower, lower, upper, slc.upper]),
                mode='lines',
                line=dict(color=self.data.theme.fg_colour, width=1),
                fill='toself',
                fillcolor='rgba(240,240,240,255)'
            )
            for n_slc, slc in enumerate(self.data.slcs, start=1)
            for lower, upper in [((n_slc - 1) / len(self.data.slcs), n_slc / len(self.data.slcs))]
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
        delims: List[float] = [*{*[y for slc in self.data.slcs for y in [slc.lower, slc.upper]]}]
        return _scale(self.data.n_iter, delims)

    @property
    def xaxis_props(self) -> Props:
        return dict(rangemode='nonnegative', visible=False)

    @property
    def yaxis_props(self) -> Props:
        return dict(
            domain=self.y_domain,
            range=[0, self.data.n_iter],
            tickmode='array',
            tickvals=_scale(
                self.data.n_iter,
                [*{*[y for slc in self.data.slcs for y in [slc.lower, slc.upper]]}]
            ),
            showticklabels=False
        )

    def render(self, fig: go.Figure, row: int, col: int) -> None:
        for seg in self.segments():
            fig.add_trace(seg, row, col)
        fig.add_trace(self.y_labels(), row, col)


@dataclass
class DensityPlot(Subplot):
    """Histogram for a slice (aggregating all chains) plus density plot for each chain."""

    slc: Slice
    n_slc: int

    @property
    def xaxis_props(self) -> Props:
        return dict(mirror='allticks', side='top', showticklabels=True)

    @property
    def yaxis_props(self) -> Props:
        return dict(
            side='right',
            rangemode='nonnegative',
            domain=self.y_domain
        )

    def histo(self, chain_slices: List[np.ndarray]) -> go.Histogram:
        return go.Histogram(
            x=[x for xs in chain_slices for x in xs],
            xbins=dict(start=floor(self.data.min_sample), end=ceil(self.data.max_sample), size=1),
            marker=dict(
                color=self.data.theme.bg_colour,
                line=dict(color=self.data.theme.fg_colour, width=1)
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
                line=dict(width=2, color=self.data.theme.palette[n]),
            )
            for n in range(0, self.data.n_chains)
        ]

    def render(self, fig: go.Figure, row: int, col: int) -> None:
        chain_slices: List[np.ndarray] = [
            self.data.chains[
                n,
                floor(self.slc.lower * self.data.n_iter):floor(self.slc.upper * self.data.n_iter)
            ]
            for n in range(0, self.data.n_chains)
        ]

        fig.add_trace(self.histo(chain_slices), row, col)
        for chain_plot in self.chain_plots(chain_slices):
            fig.add_trace(chain_plot, row, col)


@dataclass
class DensityPlots(Subplots):
    """Right-hand component: one density plot per slice."""

    def __init__(self, axis_ids: AxisIds, y_domain: Tuple[float, float], data: ChartData):
        super().__init__(axis_ids=axis_ids, y_domain=y_domain, data=data, plots=[
            DensityPlot(
                increment_axes(axis_ids, n_slc),
                segment(y_domain, len(data.slcs), n_slc), data, slc, n_slc
            )
            for n_slc, slc in enumerate(data.slcs)
        ])


@dataclass
class RafteryLewisPlot(Subplot):
    """Beneath left-hand component: one Raftery-Lewis plot per chain."""

    n_chain: int

    @property
    def xaxis_props(self) -> Props:
        return dict(
            visible=False,
            range=[0, max(self.data.n_iter, self.required_sample_size())]
        )

    @property
    def yaxis_props(self) -> Props:
        return dict(visible=False, domain=_scale(0.04, [self.n_chain, self.n_chain + 1]))

    def required_sample_size(self) -> int:
        """Return N component of resmatrix component of result of raftery.diag R function."""
        result = coda.raftery_diag(self.data.chains[self.n_chain])
        resmatrix = result[1][0]
        return int(resmatrix[1])  # N is a float, but represents an iteration count

    def plot(self) -> go.Scatter:
        return go.Scatter(
            x=list(range(0, self.data.n_iter)),
            y=self.data.chains[self.n_chain],
            line=dict(color=self.data.theme.palette[self.n_chain])
        )

    def render(self, fig: go.Figure, row: int, col: int) -> None:
        fig.add_trace(self.plot(), row, col)


@dataclass
class RafteryLewisPlots(Subplots):
    """Bottom component: one Raftery-Lewis plot per chain."""

    def __init__(self, axis_ids: AxisIds, y_domain: Tuple[float, float], data: ChartData):
        super().__init__(axis_ids=axis_ids, y_domain=y_domain, data=data, plots=[
            RafteryLewisPlot(increment_axes(axis_ids, n), y_domain, data, n)
            for n, _ in enumerate(data.chains)
        ])


class SliceHistogram:
    """Top-level plot, for a given parameter."""

    backfillz: Backfillz
    data: ChartData
    tracePlot: TracePlot
    rafteryLewisPlots: RafteryLewisPlots
    joiningSegments: JoiningSegments
    densityPlots: DensityPlots

    def __init__(self, backfillz: Backfillz, slcs: List[Slice], param: str):
        """Construct a Slice Histogram for a given parameter from a list of slices."""
        self.backfillz = backfillz
        chains: np.ndarray = backfillz.iter_chains(param)
        self.data = ChartData(
            theme=backfillz.theme,
            slcs=slcs,
            param=param,
            chains=chains,
            max_sample=np.amax(backfillz.mcmc_samples[param]),
            min_sample=np.amin(backfillz.mcmc_samples[param]),
        )
        self.tracePlot = TracePlot((None, None), (0.25, 1.0), self.data)
        self.rafteryLewisPlots = RafteryLewisPlots((6, 6), (0, 0.20), self.data)
        self.joiningSegments = JoiningSegments((2, 2), (0.25, 1.0), self.data)
        self.densityPlots = DensityPlots((3, 3), (0.25, 1.0), self.data)

    @property
    def figure(self) -> go.Figure:
        """Derive Plotly figure from 3 parts."""
        fig: go.Figure = self._layout()
        self.render(fig)
        return fig

    def _layout(self) -> go.Figure:
        n_slcs: int = len(self.data.slcs)
        layout: go.Layout = go.Layout(
            title=f"Trace slice histogram of {self.data.param}",
            titlefont=dict(size=32),
            plot_bgcolor=self.data.theme.bg_colour,
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
            subplot_titles=["Trace Plot with Slices", None, "Density Plots for Slices"]
        )

        axis_settings: Dict[str, Any] = dict(
            showgrid=False,
            zeroline=False,
            linecolor=self.data.theme.fg_colour,
            ticks='outside',
            tickwidth=1,
            ticklen=5,
            tickcolor=self.data.theme.fg_colour,
            fixedrange=True,  # disable selection zoom
        )

        fig.update_xaxes(**axis_settings)
        fig.update_yaxes(**axis_settings)

        self.tracePlot.layout_axes(fig)
        self.densityPlots.layout_axes(fig)
        self.joiningSegments.layout_axes(fig)
        self.rafteryLewisPlots.layout_axes(fig)

        # TODO: eliminate magic indices 0, 1
        annotations = fig.layout.annotations
        annotations[0].update(xanchor='left', x=fig.layout[self.tracePlot.xaxis_id].domain[0])
        annotations[1].update(y=1.03)  # oof -- adjust title subgraph
        annotations[1].update(xanchor='left', x=fig.layout[self.densityPlots.xaxis_id].domain[0])

        return fig

    def render(self, fig: go.Figure) -> None:
        """Render subplots into fig at appropriate rows/columns."""
        self.tracePlot.render(fig, 1, 1)
        self.rafteryLewisPlots.render(fig, len(self.data.slcs) + 1, 1)
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
