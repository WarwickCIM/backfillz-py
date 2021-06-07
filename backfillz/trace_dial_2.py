from dataclasses import dataclass
from typing import List

import numpy as np
import plotly.graph_objects as go

from backfillz.core import Backfillz, ParameterSlices, Slice
from backfillz.plot import Plot, RootPlot


# Experiment to see if I can add axes directly, without involving make_subplots.
@dataclass
class TraceDial2(RootPlot):
    """Top-level trace dial plot for a given parameter."""

    data: ParameterSlices

    @property
    def title(self) -> str:
        return f"Pretzel plot for {self.data.param}"

    @property
    def plots(self) -> List[Plot]:
        return []

    # Override render to customise layout.
    def render(self) -> None:
        """Create fig and render subplots."""
        fig: go.Figure = go.Figure(
            layout=go.Layout(
                title=self.title,
                titlefont=dict(size=30),
                plot_bgcolor=self.theme.bg_colour,
                showlegend=False,
                xaxis=dict(
                    domain=[0, 1],
                    anchor='y'
                ),
                yaxis=dict(
                    domain=[0, 1],
                    anchor='x'
                ),
                xaxis2=dict(
                    domain=[0.5, 1],
                    anchor='y2',
                ),
                yaxis2=dict(
                    domain=[0.5, 1],
                    anchor='x2'
                )
            )
        )

        for plot in self.plots:
            plot.layout_axes(fig)

        self.add_additional_titles(fig)

        for plot in self.plots:
            plot.render(fig)

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

        TraceDial2(backfillz.theme, data).render()
