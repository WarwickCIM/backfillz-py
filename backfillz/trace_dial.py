from dataclasses import dataclass
import math
from typing import List, Tuple

import numpy as np
from plotly.basedatatypes import BaseTraceType  # type: ignore
import plotly.graph_objects as go  # type: ignore

from backfillz.data import Axis, Domain, MCMCRun, normalise, ParameterSlices, Props, segment, Slice, to_domain
from backfillz.plot import (
    AggregatePlot, alpha, background_rect, fresh_axis_id, LeafPlot, left_vertical_title, Plot, RootPlot
)
from backfillz.slice_histograms import SliceHistogram
from backfillz.theme import BackfillzTheme


@dataclass
class DialPlot(LeafPlot):
    """Trace dial plot on the left."""

    radial_domain: Domain = 1 / 3, 1.0
    angular_domain: Domain = 0.5 * math.pi, 2 * math.pi

    @property
    def plot_elements(self) -> List[BaseTraceType]:
        return [*self.donut_segments, *self.polar_traces, self.radial_axis]

    @property
    def xaxis_props(self) -> Props:
        # scaleanchor='y', scaleratio=1 would force an exact square/circle, but then can't
        # set histogram widths correctly
        return dict(visible=False)

    @property
    def yaxis_props(self) -> Props:
        return dict(range=[-1, 1], visible=False)

    @staticmethod
    def polar_plot(
        xs: List[float],
        ys: List[float],
        x_axis: Axis,
        y_axis: Axis
    ) -> Tuple[List[float], List[float]]:
        assert len(xs) == len(ys)
        xs_angular = [x_axis.translate(x) for x in xs]
        ys_radial = [y_axis.translate(y) for y in ys]
        return ([math.cos(x) * ys_radial[n] for n, x in enumerate(xs_angular)],
                [math.sin(x) * ys_radial[n] for n, x in enumerate(xs_angular)])

    @staticmethod
    def arc(x_domain: Domain, y: float, n_segments: int) -> Tuple[List[float], List[float]]:
        xs = [*range(0, n_segments)]
        ys = [y] * n_segments
        return DialPlot.polar_plot(xs, ys, normalise(xs, x_domain), Axis((0.0, 1.0), DialPlot.radial_domain))

    @staticmethod
    def donut_segment(x_domain: Domain, fillcolor: str) -> go.Scatter:
        xs1, ys1 = DialPlot.arc(x_domain, 1.0, 100)
        xs2, ys2 = DialPlot.arc(x_domain, 0.0, 50)
        x = xs1 + xs2[::-1]
        y = ys1 + ys2[::-1]
        return go.Scatter(x=x, y=y, line=dict(width=0), fill='toself', fillcolor=fillcolor)

    @staticmethod
    def slice_domain(slc: Slice) -> Domain:
        return to_domain(slc.lower, DialPlot.angular_domain), to_domain(slc.upper, DialPlot.angular_domain)

    @property
    def donut_segments(self) -> List[go.Scatter]:
        [burn, remaining] = self.data.slcs
        colours = DerivativeColours(self.theme)
        return [
            DialPlot.donut_segment(DialPlot.slice_domain(burn), colours.inner_burn_segment),
            DialPlot.donut_segment(DialPlot.slice_domain(remaining), colours.remaining_segment)
        ]

    @property
    def radial_axis(self) -> go.Scatter:
        x1, y1 = DialPlot.arc(DialPlot.angular_domain, -0.1, 100)
        x2, y2 = DialPlot.arc(DialPlot.angular_domain, -0.13, 100)
        y2a = [math.nan for x in x2]
        xs = [x for p in zip(x1, x2, x2) for x in p]
        ys = [y for p in zip(y1, y2, y2a) for y in p]
        return go.Scatter(x=xs, y=ys, line=dict(width=1, color=self.theme.fg_colour))

    def polar_trace(self, n: int, x_axis: Axis, y_axis: Axis) -> go.Scatter:
        chain: np.ndarray = self.data.chains[n]
        xs, ys = DialPlot.polar_plot([*range(0, len(chain))], [*chain], x_axis, y_axis)
        return go.Scatter(x=xs, y=ys, line=dict(color=self.theme.palette[n]))

    @property
    def polar_traces(self) -> List[go.Scatter]:
        x_axis = Axis((0, self.data.n_iter), DialPlot.angular_domain)
        y_axis = Axis((self.data.min_sample, self.data.max_sample), DialPlot.radial_domain)
        return [self.polar_trace(n, x_axis, y_axis) for n, _ in enumerate(self.data.chains)]


@dataclass
class TraceDialHistogram(SliceHistogram):
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
class SliceHistograms(AggregatePlot):
    """One slice histogram per slice."""

    def make_plots(self) -> List[Plot]:
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
class TraceDial(RootPlot):
    """Top-level plot, for a given parameter and chain."""

    def make_plots(self) -> List[Plot]:
        return [self.dial_plot, self.histograms]

    @property
    def dial_plot(self) -> DialPlot:
        return DialPlot(
            axis_id='',  # Plotly default axes
            x_domain=(0.0, 1),
            y_domain=(0.0, 1),
            data=self.data,
            theme=self.theme,
        )

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
    def burn_in_histo(self) -> Plot:
        return self.histograms.plots[0]

    @property
    def sample_histo(self) -> Plot:
        return self.histograms.plots[1]

    @property
    def layout_props(self) -> Props:
        # plotting region won't be exactly square but best we can do to align histogram width with donut
        colours = DerivativeColours(self.theme)
        return dict(
            width=800, height=800,
            shapes=[
                background_rect(self.burn_in_histo, colours.inner_burn_segment),
                background_rect(self.sample_histo, colours.remaining_segment)
            ]
        )

    def add_additional_titles(self, fig: go.Figure) -> None:
        left_vertical_title(fig, self.burn_in_histo, "Burn-in histogram")
        left_vertical_title(fig, self.sample_histo, "Sample histogram")

    @staticmethod
    def fig(mcmc_run: MCMCRun, theme: BackfillzTheme, verbose: bool, param: str) -> go.Figure:
        """Create a trace slice histogram."""
        burn_in_iter: int = 500  # how to decide?
        burn_in_end: float = burn_in_iter / mcmc_run.samples.num_samples
        return TraceDial(
            x_domain=(0.0, 1.0),
            y_domain=(0.0, 1.0),
            data=ParameterSlices(
                slcs=[Slice(0.0, burn_in_end), Slice(burn_in_end, 1)],
                param=param,
                chains=mcmc_run.iter_chains(param),
                max_sample=np.amax(mcmc_run.samples[param]),
                min_sample=np.amin(mcmc_run.samples[param]),
            ),
            theme=theme,
            verbose=verbose
        ).make_fig()


# Not using these properties yet.
@dataclass
class DerivativeColours:
    """Colours uniquely determined by a theme."""

    theme: BackfillzTheme

    @property
    def trace_line(self) -> str:
        return self.theme.text_font_colour

    @property
    def guide_lines(self) -> str:
        return self.theme.fg_colour

    @property
    def inner_burn_segment(self) -> str:
        return alpha(self.theme.mg_colour, self.theme.alpha + 0.2)

    @property
    def outer_burn_segment(self) -> str:
        return alpha(self.theme.mg_colour, self.theme.alpha + 0.1)

    @property
    def remaining_segment(self) -> str:
        return alpha(self.theme.mg_colour, self.theme.alpha - 0.3)
