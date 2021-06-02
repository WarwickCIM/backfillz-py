from dataclasses import dataclass
from typing import List

import numpy as np
import plotly.graph_objects as go  # type: ignore

from backfillz.core import Backfillz, HistoryEntry, HistoryEvent, ParameterSlices, Slice
from backfillz.plot import LeafPlot, Plot, RootPlot, Specs, VerticalSubplots
from backfillz.theme import BackfillzTheme


@dataclass
class DialPlot(LeafPlot):
    """Trace dial plot on the left."""

    def render(self, fig: go.Figure) -> None:
        pass


@dataclass
class Histograms(VerticalSubplots):
    """Histograms in the top-right quadrant, one for each of the two slices."""

    def make_plots(self) -> List[Plot]:
        return []


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
            axis_ids=[None],
            # top-right quadrant:
            x_domain=(0.5, 1.0),
            y_domain=(0.5, 1.0),
            row=1,
            col=2,
            data=self.data,
            theme=self.theme,
        )

    @property
    def histograms(self) -> Histograms:
        return Histograms(
            axis_ids=[2, 3],
            x_domain=(0, 1.0),
            y_domain=(0, 1.0),
            row=1,
            col=1,
            data=self.data,
            theme=self.theme,
        )

    def grid_specs(self, layout: go.Layout) -> Specs:
        return ([
            [dict(rowspan=len(self.data.slcs)), dict()],  # upper quadrants
            [None, None]                                  # lower quadrants
        ])

    @property
    def title(self) -> str:
        return f"Trace slice histogram of {self.data.param}"

    def add_title(self, fig: go.Figure) -> None:
        pass

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
