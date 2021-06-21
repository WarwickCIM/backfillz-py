from dataclasses import dataclass
import math
from typing import List

import numpy as np
from plotly.basedatatypes import BaseTraceType  # type: ignore
import plotly.graph_objects as go  # type: ignore

from backfillz.core import Backfillz, HistoryEntry, HistoryEvent, ParameterSlices, Props, Slice
from backfillz.plot import alpha, annotate, LeafPlotNoAxes, Plot, RootPlot, segment, Specs, VerticalSubplots
from backfillz.slice_histograms import SliceHistogram
from backfillz.theme import BackfillzTheme


@dataclass
class DialPlot(LeafPlotNoAxes):
    """Trace dial plot on the left."""

    hole_size: float = 1 / 3
    donut_start: float = 0.5 * math.pi
    donut_end: float = 2 * math.pi

    # Annoyingly, the "donut" is always drawn on top of the polar traces, so this approach won't work.
    @property
    def plot_elements(self) -> List[BaseTraceType]:
        return [self.donut] + self.polar_traces + self.polar_traces_2

    @property
    def donut(self) -> BaseTraceType:
        burn_in_end: float = self.data.slcs[0].upper
        return go.Pie(
            values=[0.25, (1 - burn_in_end) * 0.75, burn_in_end * 0.75],
            hole=DialPlot.hole_size,
            direction='clockwise',
            sort=False,
            domain=dict(x=self.x_domain, y=self.y_domain),
            marker=dict(
                colors=[
                    'rgba(0, 0, 0, 0)',
                    self.theme.bg_colour,  # not sure what colours to use here
                    self.theme.mg_colour,
                ]
            ),
            textinfo='none'
        )

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

    @property
    def polar_traces_2(self) -> List[go.Scatter]:
        thetas = [DialPlot.to_angular(self.normalise_iter(n)) for n in range(0, self.data.n_iter)]
        ys = [math.sin(theta) for theta in thetas]
        xs = [math.cos(theta) for theta in thetas]
        result = [
            go.Scatter(
                x=[DialPlot.to_angular(self.normalise_iter(n)) for n in range(0, self.data.n_iter)],
                y=chain,
                line=dict(color=self.theme.palette[n])
            )
            for n, chain in enumerate(self.data.chains)
        ]
        return []

    @property
    def polar_traces(self) -> List[go.Scatterpolar]:
        circle_segments: range = range(0, 360)
        return [
            go.Scatterpolar(
                theta=[n / self.data.n_iter * 270 for n in range(0, self.data.n_iter)],
                r=chain,
                line=dict(color=self.theme.palette[n]),
                subplot='polar',
            )
            for n, chain in enumerate(self.data.chains)
        ] + [
            go.Scatterpolar(
                theta=list(circle_segments),
                r=[(self.data.max_sample - self.data.min_sample) / 2 for _ in circle_segments],
                line=dict(color=self.theme.mg_colour),
                subplot='polar',
            ),
        ]


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
    """Two slice histograms: one for burn in, one for rest of chain."""

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
class TraceDial(RootPlot):
    """Top-level plot, for a given parameter and chain."""

    data: ParameterSlices
    theme: BackfillzTheme

    @property
    def plots(self) -> List[Plot]:
        return [self.dial_plot, self.histograms]

    @property
    def dial_plot(self) -> DialPlot:
        return DialPlot(
            # entire root plot:
            x_domain=(0.0, 1),
            y_domain=(0.0, 1),
            row=1,
            col=1,
            data=self.data,
            theme=self.theme,
        )

    @property
    def histograms(self) -> SliceHistograms:
        x_to_y: float = 0.865  # magic ratio that I don't know how to discover
        return SliceHistograms(
            axis_ids=['', '2'],
            # top-right quadrant:
            x_domain=(0.5 + (x_to_y - 0.5) * DialPlot.hole_size, x_to_y),
            y_domain=(0.5, 1.0),
            row=1,
            col=2,
            data=self.data,
            theme=self.theme,
        )

    def grid_specs(self, layout: go.Layout) -> Specs:
        return (
            [[dict(rowspan=len(self.data.slcs), type='domain'), dict()]] +  # upper quadrants
            [[None, dict()] for _ in self.data.slcs[1:]] +
            [[None, None]]                                                  # lower quadrants
        )

    @property
    def title(self) -> str:
        return f"Pretzel plot for {self.data.param}"

    def add_additional_titles(self, fig: go.Figure) -> None:
        super().add_additional_titles(fig)

        # TODO: move to more appropriate place.
        fig.update_layout(
            polar=dict(
                sector=[90, 360],
                hole=DialPlot.hole_size,
                bgcolor='rgba(0,0,0,0)',
                radialaxis=dict(showgrid=False, angle=90, tickangle=90, ticks='outside'),
                angularaxis=dict(showgrid=False, rotation=90, showticklabels=False),
                domain=dict(x=[0, 1]),
            ),
            barmode='overlay'
        ),

        histos: List[Plot] = self.histograms.plots
        annotate(fig, 16, histos[0].top_left, 'right', 'top', None, "Burn-in histogram", textangle=-90)
        annotate(fig, 16, histos[1].top_left, 'right', 'top', None, "Sample histogram", textangle=-90)

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

        TraceDial(backfillz.theme, data).render()
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
