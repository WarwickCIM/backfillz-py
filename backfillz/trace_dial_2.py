from dataclasses import dataclass
import math
from typing import List

import numpy as np
from plotly.basedatatypes import BaseTraceType
import plotly.graph_objects as go
from plotly.subplots import make_subplots  # type: ignore

from backfillz.core import Backfillz, HistoryEntry, HistoryEvent, ParameterSlices, Props, Slice
from backfillz.plot import cols, LeafPlot, Plot, RootPlot, segment, Specs, VerticalSubplots
from backfillz.slice_histograms import SliceHistogram
from backfillz.theme import BackfillzTheme


@dataclass
class DialPlot(LeafPlot):
    """Trace dial plot on the left."""

    hole_size: float = 1 / 3
    donut_start: float = 0.5 * math.pi
    donut_end: float = 2 * math.pi

    @property
    def plot_elements(self) -> List[BaseTraceType]:
        return [trace for trace in self.polar_traces]  # type coonversion

    @property
    def xaxis_props(self) -> Props:
        return dict(scaleanchor='y', scaleratio=1)

    @property
    def yaxis_props(self) -> Props:
        return dict(range=[-1, 1])

    @staticmethod
    def to_angular(x: float) -> float:
        """Normalised x coordinate as angular coordinate in 3/4 circle."""
        return DialPlot.donut_start + x * (DialPlot.donut_end - DialPlot.donut_start)

    @staticmethod
    def to_radial(y: float) -> float:
        """Normalised y coordinate as radial coordinate along upper 2/3 of radius."""
        return DialPlot.hole_size + y * (1 - DialPlot.hole_size)

    def normalise_iter(self, n: int) -> float:
        return n / self.data.n_iter

    def normalise_sample(self, y: float) -> float:
        return (y - self.data.min_sample) / (self.data.max_sample - self.data.min_sample)

    def polar_trace(self, n: int) -> go.Scatter:
        chain = [(x, y) for x, y in enumerate(self.data.chains[n])]
        xs = [x / (len(chain) - 1) for x, _ in chain]                           # normalise x
        ys = [DialPlot.to_radial(self.normalise_sample(y)) for _, y in chain]   # normalise y
        xs_ang = [DialPlot.to_angular(x) for x in xs]
        xs_circ = [math.cos(x) * ys[n] for n, x in enumerate(xs_ang)]
        ys_circ = [math.sin(x) * ys[n] for n, x in enumerate(xs_ang)]
        return go.Scatter(
            x=xs_circ,
            y=ys_circ,
            line=dict(color=self.theme.palette[n])
        )

    @property
    def polar_traces(self) -> List[go.Scatter]:
        return [self.polar_trace(n) for n, _ in enumerate(self.data.chains)]


@dataclass
class SliceHistograms(VerticalSubplots):
    """One slice histogram per slice."""

    def make_plots(self) -> List[Plot]:
        return [
            SliceHistogram(
                axis_id=self.axis_ids[n],
                x_domain=self.x_domain,
                y_domain=segment(self.y_domain, len(self.data.slcs), n),
                data=self.data,
                theme=self.theme,
                slc=slc,
                n_slc=n,
                row=1,
                col=1,
            )
            for n, slc in enumerate(self.data.slcs)
        ]


@dataclass
class TraceDial:
    """Top-level plot, for a given parameter and chain."""

    data: ParameterSlices
    theme: BackfillzTheme

    @property
    def plots(self) -> List[Plot]:
        return [self.dial_plot]

    @property
    def dial_plot(self) -> DialPlot:
        return DialPlot(
            axis_id='',
            x_domain=(0.0, 1),
            y_domain=(0.0, 1),
            row=1,
            col=1,
            data=self.data,
            theme=self.theme,
        )

    @property
    def histograms(self) -> SliceHistograms:
        return SliceHistograms(
            axis_ids=['', '2'],
            x_domain=(0.5, 1.0),
            y_domain=(0.5, 1.0),
            row=1,
            col=1,
            data=self.data,
            theme=self.theme,
        )

    def grid_specs(self, fig: go.Figure) -> Specs:
        return [[dict()]]  # maybe don't need subfigs for this one

    @property
    def title(self) -> str:
        return f"Pretzel plot for {self.data.param}"

    def render(self) -> None:
        """Create fig and render subplots."""
        layout: go.Layout = go.Layout(
            title=self.title,
            titlefont=dict(size=30),
            plot_bgcolor=self.theme.bg_colour,
            showlegend=False,
            barmode='overlay',
            xaxis2=dict(domain=[0.5, 1], anchor='y2'),
            yaxis2=dict(domain=[0.5, 1], anchor='x2')
        )

        fig: go.Figure = go.Figure(layout=layout)

        specs: Specs = self.grid_specs(fig)

        make_subplots(
            rows=len(specs),
            cols=cols(specs),
            figure=fig,
            specs=specs,
            horizontal_spacing=0,
            vertical_spacing=0,
            print_grid=True,
        )

        for plot in self.plots:
            plot.layout_axes(fig)

        for plot in self.plots:
            plot.render(fig)

#        fig.show(config=dict(displayModeBar=False, showAxisDragHandles=False))

        trace1 = go.Scatter(
            x=[1, 2, 3],
            y=[4, 3, 2]
        )
        trace2 = go.Scatter(
            x=[20, 30, 40],
            y=[30, 40, 50],
            xaxis='x2',
            yaxis='y2'
        )
        data = [trace1, trace2]
        layout = go.Layout(
            xaxis2=dict(
                domain=[0.6, 0.95],
                anchor='y2'
            ),
            yaxis2=dict(
                domain=[0.6, 0.95],
                anchor='x2'
            )
        )
        fig = go.Figure(data=data, layout=layout)
        fig.show()
#        plotly.iplot(fig, filename='simple-inset')

    @staticmethod
    def plot(backfillz: Backfillz, save_plot: bool = False) -> None:
        slcs: List[Slice] = [Slice(0.0, 0.04), Slice(0.4, 1)]  # how to decide
        param: str = backfillz.params[0]  # pick first parameter for now (mu)
        data: ParameterSlices = ParameterSlices(
            slcs=slcs,
            param=param,
            chains=backfillz.iter_chains(param),
            max_sample=np.amax(backfillz.mcmc_samples[param]),
            min_sample=np.amin(backfillz.mcmc_samples[param]),
        )

        TraceDial(data, backfillz.theme).render()
        backfillz.plot_history.append(HistoryEntry(HistoryEvent.TRACE_DIAL, save_plot))
