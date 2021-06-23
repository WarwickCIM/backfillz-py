from dataclasses import dataclass
import math
from typing import cast, List, Tuple

import numpy as np
from plotly.basedatatypes import BaseTraceType  # type: ignore
import plotly.graph_objects as go  # type: ignore

from backfillz.core import Backfillz, HistoryEntry, HistoryEvent, ParameterSlices, Props, Slice
from backfillz.plot import alpha, LeafPlot, Plot, segment, VerticalSubplots
from backfillz.slice_histograms import SliceHistogram
from backfillz.theme import BackfillzTheme


@dataclass
class DialPlot(LeafPlot):
    """Trace dial plot on the left."""

    hole_size: float = 1 / 3
    donut_domain: Tuple[float, float] = 0.5 * math.pi, 2 * math.pi

    @property
    def plot_elements(self) -> List[BaseTraceType]:
        return [self.donut] + [trace for trace in self.polar_traces]  # type conversion

    @property
    def xaxis_props(self) -> Props:
        # scaleanchor='y', scaleratio=1 would force an exact square/perfect circle, but then we can't
        # set width of histograms correctly
        return dict(visible=False)

    @property
    def yaxis_props(self) -> Props:
        return dict(range=[-1, 1], visible=False)

    @staticmethod
    def to_angular(x: float, domain: Tuple[float, float]) -> float:
        """Normalised x coordinate as angular coordinate in specified portion of unit circle."""
        start, end = domain
        return start + x * (end - start)

    @staticmethod
    def to_radial(y: float) -> float:
        """Normalised y coordinate as radial coordinate along upper 2/3 of radius."""
        return DialPlot.hole_size + y * (1 - DialPlot.hole_size)

    def normalise_iter(self, n: int) -> float:
        return n / self.data.n_iter

    def normalise_sample(self, y: float) -> float:
        return (y - self.data.min_sample) / (self.data.max_sample - self.data.min_sample)

    @staticmethod
    def polar_plot(xs: List[float], ys: List[float]) -> Tuple[List[float], List[float]]:
        xs_ang = [DialPlot.to_angular(x, DialPlot.donut_domain) for x in xs]
        xs_circ = [math.cos(x) * ys[n] for n, x in enumerate(xs_ang)]
        ys_circ = [math.sin(x) * ys[n] for n, x in enumerate(xs_ang)]
        return xs_circ, ys_circ

    def polar_trace(self, n: int) -> go.Scatter:
        chain = [(x, y) for x, y in enumerate(self.data.chains[n])]
        xs = [x / (len(chain) - 1) for x, _ in chain]                           # normalise x
        ys = [DialPlot.to_radial(self.normalise_sample(y)) for _, y in chain]   # normalise y
        xs_circ, ys_circ = DialPlot.polar_plot(xs, ys)
        return go.Scatter(
            x=xs_circ,
            y=ys_circ,
            line=dict(color=self.theme.palette[n]),
        )

    @property
    def donut(self) -> go.Scatter:
        n_segments: int = 100
        xs1 = [0, *range(0, n_segments)]
        ys1 = [DialPlot.hole_size] + [1.0] * n_segments
        assert len(xs1) == len(ys1)
        xs1 = [x / max(xs1) for x in xs1]
        xs_circ1, ys_circ1 = DialPlot.polar_plot(xs1, ys1)
        xs2 = [n_segments - 1, *range(n_segments - 1, -1, -1)]
        ys2 = [1.0] + [DialPlot.hole_size] * n_segments
        assert len(xs2) == len(ys2)
        xs2 = [x / max(xs2) for x in xs2]
        xs_circ2, ys_circ2 = DialPlot.polar_plot(xs2, ys2)
        return go.Scatter(
            x=xs_circ1 + xs_circ2,
            y=ys_circ1 + ys_circ2,
            line=dict(width=0),
            fill='toself',
            fillcolor=self.theme.mg_colour,
        )

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
class SliceHistograms(VerticalSubplots):
    """One slice histogram per slice."""

    def make_plots(self) -> List[Plot]:
        return [
            TraceDialHistogram(
                axis_id=self.axis_ids[n],
                x_domain=self.x_domain,
                y_domain=segment(self.y_domain, len(self.data.slcs), n),
                data=self.data,
                theme=self.theme,
                slc=slc,
                n_slc=n,
                row=self.row + len(self.data.slcs) - 1 - n,
                col=self.col,
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
        return [self.dial_plot, self.histograms]

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
            axis_ids=['3', '2'],
            x_domain=(0.5 + DialPlot.hole_size / 2, 1.0),
            y_domain=(0.5, 1.0),
            row=1,
            col=1,
            data=self.data,
            theme=self.theme,
        )

    @property
    def title(self) -> str:
        return f"Pretzel plot for {self.data.param}"

    def render(self) -> None:
        """Create fig and render subplots."""
        layout = go.Layout(
            title=self.title,
            titlefont=dict(size=30),
            plot_bgcolor=self.theme.bg_colour,
            showlegend=False,
            barmode='overlay',
            xaxis2=dict(anchor='y2'),
            yaxis2=dict(anchor='x2'),
            xaxis3=dict(anchor='y3'),
            yaxis3=dict(anchor='x3'),
            # plotting region won't be exactly square but best we can do to align histogram width with donut
            width=800, height=800,
        )
        fig = go.Figure(layout=layout)

        for plot in self.plots:
            plot.layout_axes(fig)

        for trace in self.dial_plot.plot_elements:
            fig.add_trace(trace)

        for histo in self.histograms.plots:
            for trace in cast(LeafPlot, histo).plot_elements:
                fig.add_trace(trace)

        fig.show(config=dict(displayModeBar=False, showAxisDragHandles=False))

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
