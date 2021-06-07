from dataclasses import dataclass
from typing import List

import numpy as np
from plotly.basedatatypes import BaseTraceType  # type: ignore
import plotly.graph_objects as go  # type: ignore

from backfillz.core import Backfillz, HistoryEntry, HistoryEvent, ParameterSlices, Slice
from backfillz.plot import LeafPlotNoAxes, Plot, RootPlot, segment, Specs, VerticalSubplots
from backfillz.slice_histograms import SliceHistogram
from backfillz.theme import BackfillzTheme


@dataclass
class DialPlot(LeafPlotNoAxes):
    """Trace dial plot on the left."""

    hole_size: float = 0.3

    @property
    def plot_elements(self) -> List[BaseTraceType]:
        burn_in_end: float = self.data.slcs[0].upper
        return [go.Pie(
            values=[0.25, (1 - burn_in_end) * 0.75, burn_in_end * 0.75],
            hole=DialPlot.hole_size,
            direction='clockwise',
            sort=False,
            domain=dict(x=self.x_domain, y=self.y_domain),
            marker=dict(
                colors=[
                    'rgba(0, 204, 0, 0)',
                    'rgb(255, 255, 0)',
                    'rgb(118, 17, 195)',
                ]
            ),
            textinfo='none'
        )]


@dataclass
class TraceDialHistograms(VerticalSubplots):
    """Two slice histograms, one for burn in, one for rest of chain."""

    def make_plots(self) -> List[Plot]:
        return [
            SliceHistogram(
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
    def histograms(self) -> TraceDialHistograms:
        x_to_y: float = 0.865  # magic ratio that I don't know how to discover
        return TraceDialHistograms(
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
            [[None, None]]                                                 # lower quadrants
        )

    @property
    def title(self) -> str:
        return f"Pretzel plot for {self.data.param}"

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
