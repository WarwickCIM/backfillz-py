from dataclasses import dataclass
import math
from typing import List

import numpy as np
from plotly.basedatatypes import BaseTraceType
import plotly.graph_objects as go

from backfillz.core import Backfillz, HistoryEntry, HistoryEvent, ParameterSlices, Slice
from backfillz.plot import LeafPlot, Plot, RootPlot, Specs
from backfillz.theme import BackfillzTheme


@dataclass
class DialPlot(LeafPlot):
    """Trace dial plot on the left."""

    hole_size: float = 1 / 3
    donut_start: float = 0.5 * math.pi
    donut_end: float = 2 * math.pi

    @property
    def plot_elements(self) -> List[BaseTraceType]:
        return self.polar_traces

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
        chain = self.data.chains[n]
        xys = [
            (math.cos(DialPlot.to_angular(self.normalise_iter(x))),
             math.sin(DialPlot.to_radial(self.normalise_sample(y))))
            for x, y in enumerate(chain)
        ]
        return go.Scatter(
            x=[x for x, _ in xys],
            y=[y for _, y in xys],
            line=dict(color=self.theme.palette[n]),
        )

    @property
    def polar_traces(self) -> List[go.Scatter]:
        return [self.polar_trace(n) for n, _ in enumerate(self.data.chains)]


@dataclass
class TraceDial(RootPlot):
    """Top-level plot, for a given parameter and chain."""

    data: ParameterSlices
    theme: BackfillzTheme

    @property
    def plots(self) -> List[Plot]:
        return [self.dial_plot]

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
