from dataclasses import dataclass
import math
from typing import List

import numpy as np
from plotly.basedatatypes import BaseTraceType
import plotly.graph_objects as go

from backfillz.core import Backfillz, HistoryEntry, HistoryEvent, ParameterSlices, Props, Slice
from backfillz.plot import LeafPlot, Plot, RootPlot, Specs
from backfillz.theme import BackfillzTheme


@dataclass
class DialPlot(LeafPlot):
    """Trace dial plot on the left."""

    hole_size: float = 1 / 3
    donut_start: float = 1.5 * math.pi
    donut_end: float = 2 * math.pi

    @property
    def plot_elements(self) -> List[BaseTraceType]:
        return [DialPlot.circle_experiment()]

    @property
    def xaxis_props(self) -> Props:
        return dict(range=[-1, 1])

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

    @staticmethod
    def circle_experiment() -> go.Scatter:
        chain = [(1, 1), (2, 1), (3, 1), (4, 1)]    # start with horizontal line
        xs = [x / len(chain) for x, _ in chain]     # normalise x coords
        ys = [y for _, y in chain]                  # y coords already normalised
        xs_ang = [DialPlot.to_angular(x) for x in xs]
        xs_circ = [math.cos(x) for x in xs_ang + [xs_ang[0]]]
        ys_circ = [math.sin(x) for x in xs_ang + [xs_ang[0]]]
        return go.Scatter(
            x=xs_circ,
            y=ys_circ,
            line=dict(color='black')
        )

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
            axis_id='',
            # entire root plot:
            x_domain=(0.0, 1),
            y_domain=(0.0, 1),
            row=1,
            col=1,
            data=self.data,
            theme=self.theme,
        )

    def grid_specs(self, layout: go.Layout) -> Specs:
        return [[dict()]]  # single cell for now

    @property
    def title(self) -> str:
        return f"Pretzel plot for {self.data.param}"

    def add_additional_titles(self, fig: go.Figure) -> None:
        super().add_additional_titles(fig)

        # TODO: move to more appropriate place.
        fig.update_layout(barmode='overlay')

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
