from dataclasses import dataclass
import math
from typing import List, Tuple

import numpy as np
from plotly.basedatatypes import BaseTraceType  # type: ignore
import plotly.graph_objects as go  # type: ignore

from backfillz.data import Domain, MCMCRun, normalise, ParameterSlices, Props, segment, to_domain
from backfillz.plot import AggregatePlot, alpha, annotate, fresh_axis_id, LeafPlot, Plot, RootPlot
from backfillz.slice_histograms import SliceHistogram
from backfillz.theme import BackfillzTheme


def polar_plot(xs: List[float], ys: List[float]) -> Tuple[List[float], List[float]]:
    """Normalise and plot data into angular domain and then Cartesian coordinate space."""
    xs_ang = [to_domain(x, DialPlot.angular_domain) for x in normalise(xs)]
    ys_rad = [to_domain(y, DialPlot.radial_domain) for y in normalise(ys)]
    return ([math.cos(x) * ys_rad[n] for n, x in enumerate(xs_ang)],
            [math.sin(x) * ys_rad[n] for n, x in enumerate(xs_ang)])


@dataclass
class DialPlot(LeafPlot):
    """Trace dial plot on the left."""

    radial_domain: Domain = 1 / 3, 1
    angular_domain: Domain = 0.5 * math.pi, 2 * math.pi  # radians

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

    def polar_trace(self, n: int) -> go.Scatter:
        chain = self.data.chains[n]
        xs = [*range(0, len(chain))]
        ys = [*chain]
        xs_circ, ys_circ = polar_plot(xs, ys)
        return go.Scatter(
            x=xs_circ, y=ys_circ,
            line=dict(color=self.theme.palette[n]),
        )

    @property
    def polar_traces(self) -> List[go.Scatter]:
        return [self.polar_trace(n) for n, _ in enumerate(self.data.chains)]

    @property
    def donut_segment(self) -> go.Scatter:
        n_steps: int = 100
        xs = [*range(0, n_steps)] + [*range(n_steps - 1, -1, -1)]
        ys = [1.0] * n_steps + [0.0] * n_steps
        assert len(xs) == len(ys)
        xs, ys = polar_plot(xs, ys)
        return go.Scatter(
            x=xs, y=ys,
            line=dict(width=0),
            mode='lines',
            fill='toself',
            fillcolor=self.theme.mg_colour,
        )


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
        x_start, x_end = DialPlot.radial_domain
        return SliceHistograms(
            x_domain=(0.5 + x_start / 2, 0.5 + x_end / 2),
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
        burn_in_iter: int = 480  # how to decide?
        burn_in_end: float = burn_in_iter / mcmc_run.samples.num_samples
        print(burn_in_end, mcmc_run.samples.num_samples)
        return TraceDial(
            x_domain=(0.0, 1.0),
            y_domain=(0.0, 1.0),
            data=ParameterSlices(
                slcs=[(0.0, burn_in_end), (burn_in_end, 1)],
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
    def burn_in_segment(self) -> str:
        return alpha(self.theme.fg_colour, 0.3)

    @property
    def remaining_segment(self) -> str:
        return alpha(self.theme.fg_colour, 0.1)
