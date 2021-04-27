from dataclasses import dataclass
from math import ceil, floor
from typing import Dict, List

import numpy as np
import pandas as pd  # type: ignore
import plotly.graph_objects as go  # type: ignore
from plotly.subplots import make_subplots  # type: ignore

from backfillz.Backfillz import Backfillz, HistoryEntry, HistoryEvent
from backfillz.BackfillzTheme import BackfillzTheme


@dataclass
class Slice:
    """A slice of an MCMC trace."""

    lower: float
    upper: float


Slices = Dict[str, List[Slice]]


@dataclass
class _TracePlot:
    traces: List[go.Scatter]    # one per chain
    boxes: List[go.Scatter]     # one per slice

    def __init__(self, theme: BackfillzTheme, chains: np.ndarray, n_chains: int):
        self.traces = [
            go.Scatter(
                x=chains[n],
                y=list(range(0, chains[n].size)),
                line=dict(color=theme.palette[n])
            )
            for n in range(0, n_chains)
        ]
        self.boxes = []


@dataclass
class _JoiningSegment:
    quadrangle: go.Scatter
    upper_line: go.Scatter
    lower_line: go.Scatter

    def __init__(
        self,
        theme: BackfillzTheme,
        width: int,
        y_scale: float,
        n_slc: int,
        n_slcs: int,
        slc: Slice
    ):
        lower, upper = (n_slc - 1) / n_slcs, n_slc / n_slcs
        self.quadrangle = go.Scatter(
            x=_scale(width, [0, 1, 1, 0]),
            y=_scale(y_scale, [slc.lower, lower, upper, slc.upper]),
            mode='lines',
            line=dict(width=0),
            fill='toself',
            fillcolor='rgba(240,240,240,255)'
        )
        self.lower_line = go.Scatter(
            x=_scale(width, [0, 1]),
            y=_scale(y_scale, [slc.lower, lower]),
            mode='lines',
            line=dict(color=theme.fg_colour, width=1)
        )
        self.upper_line = go.Scatter(
            x=_scale(width, [0, 1]),
            y=_scale(y_scale, [slc.upper, upper]),
            mode='lines',
            line=dict(color=theme.fg_colour, width=1)
        )


@dataclass
class _DensityPlot:
    histo: go.Histogram
    # box: go.Scatter

    def __init__(
        self,
        theme: BackfillzTheme,
        slc: Slice,
        chains: np.ndarray,
        n_iter: int,
        min_sample: float,
        max_sample: float
    ):
        # chain 0 only for now; need to consider all?
        self.histo = go.Histogram(
            x=chains[0, floor(slc.lower * n_iter):floor(slc.upper * n_iter)],
            xbins=dict(start=floor(min_sample), end=ceil(max_sample), size=1),
            marker=dict(color=theme.bg_colour, line=dict(color=theme.fg_colour, width=1)),
            histnorm='probability'
        )


class SliceHistogram:
    """Top-level slice histogram plot for a given parameter."""

    backfillz: Backfillz
    slcs: List[Slice]
    param: str
    chains: np.ndarray
    n_chains: int
    n_iter: int
    min_sample: float
    max_sample: float
    _joining_segments: List[go.Scatter]

    def __init__(self, backfillz: Backfillz, slcs: List[Slice], param: str):
        """Construct a Slice Histogram for a given parameter from a list of slices."""
        self.backfillz = backfillz
        self.slcs = slcs
        self.param = param
        self.chains = backfillz.iter_chains(param)
        [self.n_chains, self.n_iter] = self.chains.shape
        self.max_sample = np.amax(backfillz.mcmc_samples[param])
        self.min_sample = np.amin(backfillz.mcmc_samples[param])
        # p.title=f"Trace slice histogram of {param}",
        # p.title.text_color = backfillz.theme.text_col_title

    @property
    def trace_plots(self) -> _TracePlot:
        """For each chain, get trace plot (leftmost part)."""
        return _TracePlot(self.backfillz.theme, self.chains, self.n_chains)

    @property
    def joining_segments(self) -> List[_JoiningSegment]:
        """For each slice, get joining segments (middle part)."""
        width: int = 30  # check against R version
        y_scale: int = self.n_iter
        return [
            _JoiningSegment(self.backfillz.theme, width, y_scale, n_slc, len(self.slcs), slc)
            for n_slc, slc in enumerate(self.slcs, start=1)
        ]

    @property
    def histos(self) -> List[_DensityPlot]:
        """For each slice, get histogram and sample density plot per chain."""
        return [
            _DensityPlot(
                self.backfillz.theme,
                slc,
                self.chains,
                self.n_iter,
                self.min_sample,
                self.max_sample
            )
            for slc in self.slcs[::-1]
        ]

    @property
    def figure(self) -> go.Figure:
        """Derive Plotly figure from 3 parts."""
        layout: go.Layout = go.Layout(
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            xaxis=dict(range=[self.min_sample, self.max_sample]),
            xaxis2=dict(visible=False),
            yaxis=dict(range=[0, self.n_iter]),
            yaxis2=dict(range=[0, self.n_iter]),
        )
        fig: go.Figure = go.Figure(layout=layout)
        specs: List[List[object]] = \
            [[dict(rowspan=len(self.slcs)), dict(rowspan=len(self.slcs)), dict()]] + \
            [[None, None, dict()] for _ in self.slcs[1:]]
        make_subplots(
            rows=len(self.slcs),
            cols=3,
            figure=fig,
            specs=specs,
            horizontal_spacing=0,
            vertical_spacing=0,
            shared_xaxes=True,
            print_grid=True,
        )

        for trace in self.trace_plots.traces:
            fig.add_trace(trace, row=1, col=1)
        for joining_segment in self.joining_segments:
            fig.add_trace(joining_segment.quadrangle, row=1, col=2)
            fig.add_trace(joining_segment.lower_line, row=1, col=2)
            fig.add_trace(joining_segment.upper_line, row=1, col=2)
        for n_slice, densityPlot in enumerate(self.histos):
            yaxis = 'yaxis' + str(3 + n_slice)  # ouch: 3
            fig.layout[yaxis]['side'] = 'right'
            fig.add_trace(densityPlot.histo, row=n_slice + 1, col=3)

        return fig


def plot_slice_histogram(backfillz: Backfillz, save_plot: bool = False) -> None:
    """Plot a slice histogram."""
    params = pd.Series(backfillz.mcmc_samples.param_names[0:1])  # just first param for now
    slice_list: List[Slice] = [
        Slice(0, 0.004), Slice(0.004, 0.01), Slice(0.01, 0.02), Slice(0.02, 0.04), Slice(0.04, 1)
    ]
    slices: Slices = {param: slice_list for param in params}

    for param in params:
        # Assume scalar parameter for now; what about vectors?
        SliceHistogram(backfillz, slices[param], param).figure.show()

    # Update log
    backfillz.plot_history.append(HistoryEntry(HistoryEvent.SLICE_HISTOGRAM, save_plot))


def _scale(factor: float, xs: List[float]) -> List[float]:
    return [x * factor for x in xs]
