from dataclasses import dataclass
from math import nan
from typing import List, Sequence

import numpy as np
from plotly.basedatatypes import BaseTraceType  # type: ignore
import plotly.graph_objects as go  # type: ignore

from backfillz.data import MCMCRun, ParameterData, segment
from backfillz.plot import AggregatePlot, fresh_axis_id, LeafPlot, Plot, RootPlot
from backfillz.theme import BackfillzTheme


@dataclass
class ParameterSteps(ParameterData):
    """Data being visualised by a spiral stream plot."""

    steps: List[int]


@dataclass
class SpiralRow(LeafPlot[ParameterSteps]):
    """Row of spiral plots for chain with index n."""

    n: int

    @property
    def plot_elements(self) -> Sequence[BaseTraceType]:
        spiral_points = [nan] * self.data.n_iter
        for step in self.data.steps:
            for i in range(0, self.data.n_iter):
                if i >= step:
                    klw = i - step
                else:
                    klw = 1

                if i <= (self.data.n_iter - step):
                    khg = i + step
                else:
                    khg = self.data.n_iter

            spiral_points[i] = np.var(self.data.chains[self.n][klw:khg])

        return []


class SpiralRows(AggregatePlot[ParameterSteps]):
    """One spiral row per chain."""

    def make_plots(self) -> Sequence[Plot[ParameterSteps]]:
        return [
            SpiralRow(
                axis_id=fresh_axis_id(),
                x_domain=self.x_domain,
                y_domain=segment(self.y_domain, len(self.data.chains), n),
                data=self.data,
                theme=self.theme,
                n=n,
            )
            for n, slc in enumerate(self.data.chains)
        ]


@dataclass
class SpiralStream(RootPlot[ParameterSteps]):
    """Spiral stream plot for a given parameter."""

    def make_plots(self) -> Sequence[Plot[ParameterSteps]]:
        return self.spiral_rows

    @property
    def spiral_rows(self) -> Sequence[SpiralRows]:
        return [
            SpiralRows(
                x_domain=(0, 1),
                y_domain=segment(self.y_domain, len(self.data.chains), n),
                data=self.data,
                theme=self.theme,
            )
            for n, _ in enumerate(self.data.chains)
        ]

    @property
    def title(self) -> str:
        return f"Spiral stream plot for {self.data.param}"

    @staticmethod
    def fig(mcmc_run: MCMCRun, theme: BackfillzTheme, verbose: bool, param: str) -> go.Figure:
        """Create a spiral stream plot."""
        steps: List[int] = [3, 8, 15]  # defaults for now
        return SpiralStream(
            x_domain=(0.0, 1.0),
            y_domain=(0.0, 1.0),
            data=ParameterSteps(mcmc_run, param, steps),
            theme=theme,
            verbose=verbose,
        ).make_fig()
