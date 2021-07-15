from dataclasses import dataclass
import math
from typing import List, Tuple

import numpy as np
from plotly.basedatatypes import BaseTraceType  # type: ignore
import plotly.graph_objects as go  # type: ignore

from backfillz.data import Domain, MCMCRun, normalise, ParameterSlices, Props, Slice
from backfillz.plot import (
    AggregatePlot, alpha, annotate, background_rect, fresh_axis_id, LeafPlot, Plot, RootPlot, segment
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
        return [*self.donut_segments, *[self.polar_trace(n) for n, _ in enumerate(self.data.chains)]]

    @property
    def xaxis_props(self) -> Props:
        # scaleanchor='y', scaleratio=1 would force an exact square/circle, but then can't
        # set histogram widths correctly
        return dict(visible=False)

    @property
    def yaxis_props(self) -> Props:
        return dict(range=[-1, 1], visible=False)

    @staticmethod
    def to_angular(x: float, domain: Domain) -> float:
        """Convert normalised x coordinate to angular coordinate within supplied domain."""
        start, end = domain
        return start + x * (end - start)

    @staticmethod
    def to_radial(y: float) -> float:
        """Map normalised y coordinate into radial domain."""
        start, end = DialPlot.radial_domain
        return start + y * (end - start)

    @staticmethod
    def polar_plot(xs: List[float], ys: List[float], x_domain: Domain) -> Tuple[List[float], List[float]]:
        assert len(xs) == len(ys)
        xs_ang = [DialPlot.to_angular(x, x_domain) for x in normalise(xs)]
        ys_radial = [DialPlot.to_radial(y) for y in normalise(ys)]
        return ([math.cos(x) * ys_radial[n] for n, x in enumerate(xs_ang)],
                [math.sin(x) * ys_radial[n] for n, x in enumerate(xs_ang)])

    @staticmethod
    def donut_segment(x_domain: Domain, fillcolor: str) -> go.Scatter:
        n_segments: int = 100
        xs = [0.0] + [*range(0, n_segments)] + [n_segments - 1] + [*range(n_segments - 1, -1, -1)]
        ys = [0.0] + [1.0] * n_segments + [1.0] + [0.0] * n_segments
        x, y = DialPlot.polar_plot(xs, ys, x_domain)
        return go.Scatter(x=x, y=y, line=dict(width=0), fill='toself', fillcolor=fillcolor)

    @staticmethod
    def slice_domain(slc: Slice) -> Domain:
        return (
            DialPlot.to_angular(slc.lower, DialPlot.angular_domain),
            DialPlot.to_angular(slc.upper, DialPlot.angular_domain)
        )

    @property
    def donut_segments(self) -> List[go.Scatter]:
        [burn, remaining] = self.data.slcs
        colours = DerivativeColours(self.theme)
        return [
            DialPlot.donut_segment(DialPlot.slice_domain(burn), colours.inner_burn_segment),
            DialPlot.donut_segment(DialPlot.slice_domain(remaining), colours.remaining_segment)
        ]

    def polar_trace(self, n: int) -> go.Scatter:
        chain: np.ndarray = self.data.chains[n]
        xs, ys = DialPlot.polar_plot([*range(0, len(chain))], [*chain], DialPlot.angular_domain)
        return go.Scatter(x=xs, y=ys, line=dict(color=self.theme.palette[n]))


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
        annotate(fig, 14, self.burn_in_histo.top_left, 'right', 'top', None, "Burn-in histogram", textangle=-90)
        annotate(fig, 14, self.sample_histo.top_left, 'right', 'top', None, "Sample histogram", textangle=-90)

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
