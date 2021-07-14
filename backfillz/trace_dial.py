from dataclasses import dataclass
import math
from typing import List, Sequence, Tuple

import numpy as np
from plotly.basedatatypes import BaseTraceType  # type: ignore
import plotly.graph_objects as go  # type: ignore

from backfillz.data import Domain, MCMCRun, normalise, ParameterSlices, Props, Slice
from backfillz.plot import AggregatePlot, alpha, annotate, fresh_axis_id, LeafPlot, Plot, RootPlot, segment
from backfillz.slice_histograms import SliceHistogram
from backfillz.theme import BackfillzTheme


@dataclass
class DialPlot(LeafPlot):
    """Trace dial plot on the left."""

    hole_size: float = 1 / 3
    donut_domain: Domain = 0.5 * math.pi, 2 * math.pi  # radians

    @property
    def plot_elements(self) -> List[BaseTraceType]:
        return [self.donut_segment, *self.polar_traces]

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
        """Map a normalised y coordinate into upper 2/3 of radius."""
        return DialPlot.hole_size + y * (1 - DialPlot.hole_size)

    @staticmethod
    def polar_plot(xs: List[float], ys: List[float]) -> Tuple[List[float], List[float]]:
        assert len(xs) == len(ys)
        xs_ang = [DialPlot.to_angular(x, DialPlot.donut_domain) for x in normalise(xs)]
        ys_radial = [DialPlot.to_radial(y) for y in normalise(ys)]
        return ([math.cos(x) * ys_radial[n] for n, x in enumerate(xs_ang)],
                [math.sin(x) * ys_radial[n] for n, x in enumerate(xs_ang)])

    def polar_trace(self, n: int) -> go.Scatter:
        chain: np.ndarray = self.data.chains[n]
        xs, ys = DialPlot.polar_plot([*range(0, len(chain))], [*chain])
        return go.Scatter(x=xs, y=ys, line=dict(color=self.theme.palette[n]))

    @property
    def donut_segment(self) -> go.Scatter:
        n_segments: int = 100
        xs = [0] + [*range(0, n_segments)] + [n_segments - 1] + [*range(n_segments - 1, -1, -1)]
        ys = [0.0] + [1.0] * n_segments + [1.0] + [0.0] * n_segments
        x, y = DialPlot.polar_plot(xs, ys)
        return go.Scatter(x=x, y=y, line=dict(width=0), fill='toself', fillcolor=self.theme.mg_colour)

    @property
    def polar_traces(self) -> List[go.Scatter]:
        return [self.polar_trace(n) for n, _ in enumerate(self.data.chains)]


@dataclass
class TraceDialHistogram(SliceHistogram):
    """Slice histogram for trace dial plot."""

    @property
    def plot_elements(self) -> List[BaseTraceType]:
        return [self.histo([n], self.theme.palette[n], 1) for n, _ in enumerate(self.data.chains)]

    @property
    def xaxis_props(self) -> Props:
        if self.n_slc == len(self.data.slcs) - 1:
            return dict(side='top')
        else:
            return dict(visible=False)


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
        return SliceHistograms(
            x_domain=(0.5 + DialPlot.hole_size / 2, 1.0),
            y_domain=(0.5, 1.0),
            data=self.data,
            theme=self.theme,
        )

    @property
    def title(self) -> str:
        return f"Pretzel plot for {self.data.param}"

    @property
    def layout_props(self) -> Props:
        return dict(
            barmode='overlay',
            # plotting region won't be exactly square but best we can do to align histogram width with donut
            width=800, height=800,
        )

    def add_additional_titles(self, fig: go.Figure) -> None:
        histos: List[Plot] = self.histograms.plots
        annotate(fig, 14, histos[0].top_left, 'right', 'top', None, "Burn-in histogram", textangle=-90)
        annotate(fig, 14, histos[1].top_left, 'right', 'top', None, "Sample histogram", textangle=-90)

    @staticmethod
    def fig(mcmc_run: MCMCRun, theme: BackfillzTheme, verbose: bool, param: str) -> go.Figure:
        """Create a trace slice histogram."""
        slcs: List[Slice] = [Slice(0.0, 0.04), Slice(0.4, 1)]  # how to decide
        return TraceDial(
            x_domain=(0.0, 1.0),
            y_domain=(0.0, 1.0),
            data=ParameterSlices(
                slcs=slcs,
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
