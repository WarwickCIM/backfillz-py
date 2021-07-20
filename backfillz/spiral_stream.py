from dataclasses import dataclass
from math import nan
from typing import Sequence

import numpy as np
from plotly.basedatatypes import BaseTraceType  # type: ignore
import plotly.graph_objects as go

from backfillz.data import MCMCRun, ParameterSlices, segment, Slice
from backfillz.plot import AggregatePlot, fresh_axis_id, LeafPlot, Plot, RootPlot
from backfillz.theme import BackfillzTheme


@dataclass
class SpiralRow(LeafPlot):
    """Row of spiral plots for chain with index n."""

    n: int

    @property
    def plot_elements(self) -> Sequence[BaseTraceType]:
        steps = [3, 8, 15]
        spiral_points = [nan] * self.data.n_iter
        for step in steps:
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


class SpiralRows(AggregatePlot):
    """One spiral row per chain."""

    def make_plots(self) -> Sequence[Plot]:
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
class SpiralStream(RootPlot):
    """Spiral stream plot for a given parameter."""

    def make_plots(self) -> Sequence[Plot]:
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
        return SpiralStream(
            x_domain=(0.0, 1.0),
            y_domain=(0.0, 1.0),
            data=ParameterSlices(
                slcs=[Slice(0, 1)],
                param=param,
                chains=mcmc_run.iter_chains(param),
                max_sample=np.amax(mcmc_run.samples[param]),
                min_sample=np.amin(mcmc_run.samples[param]),
            ),
            theme=theme,
            verbose=verbose
        ).make_fig()
