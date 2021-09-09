from dataclasses import dataclass
from math import floor, nan, pi
from typing import List, Sequence, Tuple

import numpy as np
from plotly.basedatatypes import BaseTraceType  # type: ignore
import plotly.graph_objects as go  # type: ignore

from backfillz.data import Domain, MCMCRun, ParameterSlices, Props, segment, to_domain
from backfillz.plot import (
    AggregatePlot, Axis, background_rect, fresh_axis_id, LeafPlot, left_vertical_title,
    Plot, polar_plot, RootPlot, tick_every
)
from backfillz.slice_histograms import SliceHistogram
from backfillz.theme import BackfillzTheme


@dataclass
class TraceDialData(ParameterSlices):
    """Data being visualised by a trace dial plot."""

    burn_in_iter: int


def arc(x_domain: Domain, y: float, n_steps: int) -> Tuple[List[float], List[float]]:
    """An arc at distance y from (0,0) with angular extent x_domain."""
    xs = [*range(0, n_steps)]
    ys = [y] * n_steps
    return polar_plot(xs, ys, Axis((0, n_steps - 1), x_domain), Axis((0.0, 1.0), DialPlot.radial_domain))


@dataclass
class DialPlot(LeafPlot[TraceDialData]):
    """Trace dial plot (3/4 segment)."""

    radial_domain: Domain = 1 / 3, 1.0
    angular_domain: Domain = 0.5 * pi, 2 * pi

    @property
    def plot_elements(self) -> List[BaseTraceType]:
        return [*self.donut_segments, *self.polar_traces, *self.inner_ticks]

    @property
    def xaxis_props(self) -> Props:
        # scaleanchor='y', scaleratio=1 would force an exact square/circle, but then can't
        # set histogram widths correctly
        return dict(visible=False)

    @property
    def yaxis_props(self) -> Props:
        return dict(range=[-1, 1], visible=False)

    @staticmethod
    def donut_segment(x_domain: Domain, fillcolor: str) -> go.Scatter:
        xs1, ys1 = arc(x_domain, 1.0, 100)
        xs2, ys2 = arc(x_domain, 0.0, 50)
        xs = xs1 + xs2[::-1]
        ys = ys1 + ys2[::-1]
        return go.Scatter(x=xs, y=ys, line=dict(width=0), fill='toself', fillcolor=fillcolor)

    @staticmethod
    def slice_domain(slc: Domain) -> Domain:
        start, end = slc
        return to_domain(start, DialPlot.angular_domain), to_domain(end, DialPlot.angular_domain)

    @property
    def donut_segments(self) -> List[go.Scatter]:
        [burn_in, remaining] = self.data.slcs
        return [
            DialPlot.donut_segment(DialPlot.slice_domain(burn_in), self.theme.burn_in_segment),
            DialPlot.donut_segment(DialPlot.slice_domain(remaining), self.theme.remaining_segment)
        ]

    @property
    def angular_axis(self) -> Axis:
        return Axis((0, self.data.n_iter), DialPlot.angular_domain)

    @property
    def radial_axis(self) -> Axis:
        return Axis((self.data.min_sample, self.data.max_sample), DialPlot.radial_domain)

    @property
    def inner_ticks(self) -> List[go.Scatter]:
        ticks_per_circle = 80  # somewhat arbitrary
        tick_gap: int = tick_every(ticks_per_circle, self.angular_axis)
        start, end = self.angular_axis.range
        xs1 = [x * tick_gap for x in range(floor(start), floor(end / tick_gap))]
        xs2 = [start, self.data.burn_in_iter, end]
        top, bottom1, bottom2 = -0.04, -0.07, -0.14
        return [
            self.radial_ticks(xs1, (top, bottom1), self.theme.mg_colour),
            self.radial_ticks(xs2, (top, bottom2), self.theme.fg_colour),
            self.radial_tick_marks(xs2, bottom2)
        ]

    def radial_tick_marks(self, xs: Sequence[float], tick_bottom: float) -> go.Scatter:
        y_axis: Axis = Axis((0.0, 1.0), DialPlot.radial_domain)
        x, y = polar_plot(xs, [tick_bottom - 0.05] * len(xs), self.angular_axis, y_axis)
        return go.Scatter(x=x, y=y, text=[str(x) for x in xs], mode='text', textposition='middle left')

    def radial_ticks(self, xs: Sequence[float], tick_size: Tuple[float, float], colour: str) -> go.Scatter:
        """Ticks at supplied angular positions, sized relative to radial_domain."""
        top, bottom = tick_size
        y_axis: Axis = Axis((0.0, 1.0), DialPlot.radial_domain)
        x1, y1 = polar_plot(xs, [top] * len(xs), self.angular_axis, y_axis)
        x2, y2 = polar_plot(xs, [bottom] * len(xs), self.angular_axis, y_axis)
        x = [x for p in zip(x1, x2, x2) for x in p]
        y = [y for p in zip(y1, y2, [nan] * len(x2)) for y in p]  # nan to avoid joining all points
        return go.Scatter(x=x, y=y, mode='lines', line=dict(width=1, color=colour))

    def polar_trace(self, n: int) -> go.Scatter:
        chain: np.ndarray = self.data.chains[n]
        xs, ys = polar_plot([*range(0, len(chain))], [*chain], self.angular_axis, self.radial_axis)
        return go.Scatter(x=xs, y=ys, line=dict(color=self.theme.palette[n]))

    @property
    def polar_traces(self) -> List[go.Scatter]:
        return [self.polar_trace(n) for n, _ in enumerate(self.data.chains)]


@dataclass
class TraceDialHistogram(SliceHistogram[TraceDialData]):
    """Slice histogram for trace dial plot."""

    bin_size: float = 1.0

    @property
    def plot_elements(self) -> List[BaseTraceType]:
        return [
            *[self.step_plot(n) for n, _ in enumerate(self.data.chains)],
            self.histo([*range(0, len(self.data.chains))], self.theme.mg_colour, TraceDialHistogram.bin_size),
        ]

    def step_plot(self, n: int) -> go.Scatter:
        ys, xs = self.bins([n], TraceDialHistogram.bin_size)
        return go.Scatter(
            x=xs, y=ys,
            mode='lines',
            line=dict(width=1, color=self.theme.palette[n], shape='hvh'),
            xaxis='x' + self.axis_id, yaxis='y' + self.axis_id,
        )

    @property
    def xaxis_props(self) -> Props:
        props: Props
        if self.n_slc == len(self.data.slcs) - 1:
            props = dict(side='top')
        else:
            props = dict(visible=False)
        return {**props, **dict(range=(self.data.min_sample, self.data.max_sample))}


@dataclass
class SliceHistograms(AggregatePlot[TraceDialData]):
    """One slice histogram per slice."""

    def make_plots(self) -> List[Plot[TraceDialData]]:
        return [
            TraceDialHistogram(
                axis_id=fresh_axis_id(),
                x_domain=self.x_domain,
                y_domain=segment(self.y_domain, len(self.data.slcs), n),
                data=self.data,
                theme=self.theme,
                slc=slc,
                n_slc=n,
            )
            for n, slc in enumerate(self.data.slcs)
        ]


@dataclass
class TraceDial(RootPlot[TraceDialData]):
    """Top-level plot, for a given parameter and chain."""

    def make_plots(self) -> List[Plot[TraceDialData]]:
        return [self.dial_plot, self.histograms]

    @property
    def dial_plot(self) -> DialPlot:
        # Use Plotly default axes
        return DialPlot(axis_id='', x_domain=(0.0, 1), y_domain=(0.0, 1), data=self.data, theme=self.theme)

    @property
    def histograms(self) -> SliceHistograms:
        start, end = DialPlot.radial_domain
        return SliceHistograms(
            x_domain=(0.5 + start * 0.5, 0.5 + end * 0.5),
            y_domain=(0.5, 1.0),
            data=self.data,
            theme=self.theme,
        )

    @property
    def title(self) -> str:
        return f"Pretzel plot for {self.data.param}"

    @property
    def burn_in_histo(self) -> Plot[TraceDialData]:
        return self.histograms.plots[0]

    @property
    def sample_histo(self) -> Plot[TraceDialData]:
        return self.histograms.plots[1]

    @property
    def layout_props(self) -> Props:
        # plotting region won't be exactly square but best we can do to align histogram width with donut
        length: int = 800
        return dict(
            width=length,
            height=length,
            shapes=[
                background_rect(self.burn_in_histo, self.theme.burn_in_segment),
                background_rect(self.sample_histo, self.theme.remaining_segment)
            ]
        )

    def add_additional_titles(self, fig: go.Figure) -> None:
        left_vertical_title(fig, self.burn_in_histo, "Burn-in histogram")
        left_vertical_title(fig, self.sample_histo, "Sample histogram")

    @staticmethod
    def fig(
        mcmc_run: MCMCRun,
        theme: BackfillzTheme,
        verbose: bool,
        param: str,
        burn_in_iter: int
    ) -> go.Figure:
        """Create a trace slice histogram."""
        burn_in_end: float = burn_in_iter / mcmc_run.samples.num_samples
        slcs: List[Domain] = [(0.0, burn_in_end), (burn_in_end, 1)]
        return TraceDial(
            x_domain=(0.0, 1.0),
            y_domain=(0.0, 1.0),
            data=TraceDialData(mcmc_run, param, slcs, burn_in_iter),
            theme=theme,
            verbose=verbose,
        ).make_fig()
