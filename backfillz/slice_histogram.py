from dataclasses import dataclass
from math import ceil, floor
from typing import Any, cast, List

import numpy as np
import pandas as pd  # type: ignore
import plotly.graph_objects as go  # type: ignore
from plotly.subplots import make_subplots  # type: ignore
from rpy2.robjects import numpy2ri  # type: ignore
from rpy2.robjects.packages import importr  # type: ignore
import scipy.stats as stats  # type: ignore

from backfillz.core import Backfillz, HistoryEntry, HistoryEvent
from backfillz.plot \
    import _scale, ChartData, Plot, Props, segment, Slice, Slices, Subplot, VerticalSubplots

coda = importr("coda")  # use R for raftery.diag; might be a better diagnostic in PyMC3
numpy2ri.activate()


@dataclass
class TracePlot(Subplot):
    """Left-hand component."""

    def render(self, fig: go.Figure, row: int, col: int) -> None:
        for trace in self.traces():
            fig.add_trace(trace, row, col)
        for box in self.boxes():
            fig.add_trace(box, row, col)

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
        return dict(range=[0, self.data.n_iter])


@dataclass
class JoiningSegments(Subplot):
    """Middle component."""

    def render(self, fig: go.Figure, row: int, col: int) -> None:
        for seg in self.segments():
            fig.add_trace(seg, row, col)
        fig.add_trace(self.y_labels(), row, col)

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
        return _scale(self.data.n_iter, delims)

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
class DensityPlot(Subplot):
    """Histogram for a slice (aggregating all chains) plus density plot for each chain."""

    slc: Slice
    n_slc: int

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
                slc=slc,
                n_slc=n
            )
            for n, slc in enumerate(self.data.slcs)
        ]

    @property
    def uppermost(self) -> DensityPlot:
        return cast(DensityPlot, self.plots[-1])


@dataclass
class RafteryLewisPlot(Subplot):
    """Raftery-Lewis plot for a chain."""

    n_chain: int

    def render(self, fig: go.Figure, row: int, col: int) -> None:
        fig.add_trace(self.plot(), row, col)
        fig.add_trace(self.warning_cross(), row, col)

    def plot(self) -> go.Scatter:
        return go.Scatter(
            x=list(range(0, self.data.n_iter)),
            y=self.data.chains[self.n_chain],
            line=dict(color=self.data.theme.palette[self.n_chain])
        )

    def warning_cross(self) -> go.Scatter:
        """Singleton scatterplot to render X if iterations fall short of required sample size."""
        return go.Scatter(
            x=[self.required_sample_size],
            y=[0],
            mode='text',
            text=['X' if self.required_sample_size > self.data.n_iter else ''],
            textposition='middle center',
            cliponaxis=False  # ensure visible
        )

    @property
    def required_sample_size(self) -> int:
        """N component of resmatrix component of result of raftery.diag R function."""
        result = coda.raftery_diag(self.data.chains[self.n_chain])
        resmatrix = result[1][0]
        return int(resmatrix[1])  # N is a float, but represents an iteration count

    @property
    def xaxis_props(self) -> Props:
        return dict(
            visible=False,
            range=[0, max(self.data.n_iter, self.required_sample_size)]
        )

    @property
    def yaxis_props(self) -> Props:
        return dict(visible=False)


class RafteryLewisPlots(VerticalSubplots):
    """Bottom component: one Raftery-Lewis plot per chain."""

    def make_plots(self) -> List[Plot]:
        return [
            RafteryLewisPlot(
                axis_ids=[self.axis_ids[n]],
                x_domain=self.x_domain,
                y_domain=segment(self.y_domain, self.data.n_chains, n),
                data=self.data,
                n_chain=n
            )
            for n, _ in enumerate(self.data.chains)
        ]


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
        lower_h = 0.2       # height of Raftery-Lewis section
        lower_margin = 0.4
        left_w = 0.4        # width of trace plot
        middle_w = 0.2      # width of joining segments

        # Axis ids are one of Plotly's design failures. No easy way to extract them from the layout.
        self.tracePlot = TracePlot(
            axis_ids=[None],
            x_domain=(0, left_w),
            y_domain=(lower_h, 1.0),
            data=self.data
        )
        self.joiningSegments = JoiningSegments(
            axis_ids=[2],
            x_domain=(left_w, left_w + middle_w),
            y_domain=(lower_h, 1.0),
            data=self.data
        )
        self.densityPlots = DensityPlots(
            axis_ids=[n + 3 for n in reversed(range(self.data.n_slcs))],
            x_domain=(left_w + middle_w, 1),
            y_domain=(lower_h, 1.0),
            data=self.data
        )
        self.rafteryLewisPlots = RafteryLewisPlots(
            axis_ids=[n + 3 + len(slcs) for n in reversed(range(self.data.n_chains))],
            x_domain=(0, left_w),
            y_domain=(0, lower_h * (1 - lower_margin)),
            data=self.data
        )

    def layout(self) -> go.Figure:
        n_slcs: int = self.data.n_slcs
        layout: go.Layout = go.Layout(
            title=f"Trace slice histogram of {self.data.param}",
            titlefont=dict(size=30),
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
            # TODO: redo using annotations
            subplot_titles=["Trace Plot with Slices", "", "Density Plots for Slices"]
        )

        self.tracePlot.layout_axes(fig)
        self.densityPlots.layout_axes(fig)
        self.joiningSegments.layout_axes(fig)
        self.rafteryLewisPlots.layout_axes(fig)

        # TODO: push magic indices 0, 1 into constructors of subplots
        annotations = fig.layout.annotations
        annotations[0].update(xanchor='left', x=fig.layout[self.tracePlot.xaxis_id].domain[0])
        annotations[1].update(y=1.03)  # oof -- adjust title subgraph
        annotations[1].update(xanchor='left', x=fig.layout[self.densityPlots.uppermost.xaxis_id].domain[0])

        annotate(fig, x=0, y=0, xanchor='left', text="Raftery-Lewis Diagnostic")
        annotate(
            fig, x=1, y=0, xanchor='right',
            text="Backfillz-py by CIM, University of Warwick and The Alan Turing Institute"
        )

        return fig

    def render(self) -> None:
        """Create fig and render subplots at appropriate rows/columns."""
        fig: go.Figure = self.layout()
        self.tracePlot.render(fig, 1, 1)
        self.rafteryLewisPlots.render(fig, self.data.n_slcs + 1, 1)
        self.joiningSegments.render(fig, 1, 2)
        self.densityPlots.render(fig, 1, 3)
        fig.show(config=dict(displayModeBar=False, showAxisDragHandles=False))


def annotate(fig: go.Figure, **kwargs: Any) -> None:
    """Add an annotation to supplied figure, with supplied arguments in addition to some default settings."""
    fig.add_annotation(
        xref='paper',
        yref='paper',
        yanchor='top',
        showarrow=False,
        font=dict(size=14),
        **kwargs,
    )


def plot_slice_histogram(backfillz: Backfillz, save_plot: bool = False) -> None:
    """Plot a slice histogram."""
    params = pd.Series(backfillz.mcmc_samples.param_names[0:1])  # just first param for now
    slice_list: List[Slice] = [Slice(0.028, 0.04), Slice(0.1, 0.2), Slice(0.4, 0.9)]
    slices: Slices = {param: slice_list for param in params}

    for param in params:
        # Assume scalar parameter for now; what about vectors?
        SliceHistogram(backfillz, slices[param], param).render()

    backfillz.plot_history.append(HistoryEntry(HistoryEvent.SLICE_HISTOGRAM, save_plot))
